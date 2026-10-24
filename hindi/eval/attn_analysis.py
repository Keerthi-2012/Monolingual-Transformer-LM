"""Attention analysis: heatmaps, per-head entropy, and mean attention distance.

Entropy of head (l, h) = mean over queries of -sum_j a_ij log a_ij (nats); higher = more diffuse.
Normalized entropy divides by ln(#allowed keys) so queries early in the window are comparable.
Mean attention distance = mean over queries of sum_j a_ij (i - j): how far back the head looks.
Computation functions are separate from plotting functions.

Usage: python attn_analysis.py --ckpt runs/hindi_25m/ckpt_best.pt --out-dir ../../report/attn --name pretrained
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from common import ROOT, load_model, load_split, load_tokenizer  # noqa: E402

SENTENCES = [
    "भारत की राजधानी नई दिल्ली है और यह यमुना नदी के किनारे बसा हुआ शहर है।",
    "बच्चे सुबह जल्दी उठकर स्कूल जाते हैं और शाम को मैदान में खेलते हैं।",
    "अगर कल बारिश होगी तो हम घर पर ही रहेंगे और किताबें पढ़ेंगे।",
]
LOCAL_DIST = 5.0   # heads with mean distance below this (tokens) are called "local"


# ---------------------------------------------------------------- computation
@torch.no_grad()
def attention_stats(model, shards, device, n_seq=32, seq_len=256, batch=8):
    """Average entropy, normalized entropy and mean distance per (layer, head) over held-out windows."""
    L, H = model.cfg.n_layer, model.cfg.n_head
    ent, nent, dist = (np.zeros((L, H)) for _ in range(3))
    T = seq_len
    idx = torch.arange(T)
    d = (idx[:, None] - idx[None, :]).clamp(min=0).float().to(device)            # i - j (0 above diagonal)
    norm = torch.log(torch.arange(1, T + 1).float()).to(device)                  # ln(#allowed keys) per query
    n = 0
    for s in range(0, n_seq, batch):
        x, _ = shards.batch(list(range(s, min(s + batch, n_seq))))
        _, _, atts = model(x[:, :T].to(device), return_attn=True)
        for l, a in enumerate(atts):                                              # a: (B, H, T, T)
            a = a.float()
            e = -(a * torch.log(a + 1e-12)).sum(-1)                               # (B, H, T)
            ent[l] += e.mean((0, 2)).cpu().numpy() * len(x)
            nent[l] += (e[..., 1:] / norm[1:]).mean((0, 2)).cpu().numpy() * len(x)
            dist[l] += (a * d).sum(-1).mean((0, 2)).cpu().numpy() * len(x)
        n += len(x)
    return {"entropy": ent / n, "norm_entropy": nent / n, "mean_distance": dist / n}


@torch.no_grad()
def attention_maps(model, token_ids, device):
    """Attention weights for one sequence: array (L, H, T, T)."""
    x = torch.tensor([token_ids], device=device)
    _, _, atts = model(x, return_attn=True)
    return np.stack([a[0].float().cpu().numpy() for a in atts])


def summarize(stats):
    """Plain-dict summary incl. which heads are local (mean distance < LOCAL_DIST tokens)."""
    md = stats["mean_distance"]
    return {**{k: v.round(4).tolist() for k, v in stats.items()},
            "local_heads": int((md < LOCAL_DIST).sum()), "total_heads": int(md.size),
            "layer_mean_entropy": stats["entropy"].mean(1).round(4).tolist(),
            "layer_mean_distance": md.mean(1).round(3).tolist()}


# ---------------------------------------------------------------- plotting
def devanagari_font():
    """Name of an installed Devanagari-capable font, or None (then axes use token indices)."""
    for n in ("Nirmala UI", "Noto Sans Devanagari", "Mangal", "Lohit Devanagari", "Kohinoor Devanagari"):
        try:
            font_manager.findfont(n, fallback_to_default=False)
            return n
        except ValueError:
            continue
    return None


def plot_heatmaps(maps, labels, layers, heads, path, title):
    """Grid of heatmaps: rows = layers, columns = heads; x = key position, y = query position."""
    font = devanagari_font()
    T = len(labels)
    fig, axes = plt.subplots(len(layers), len(heads), figsize=(4.2 * len(heads), 4.0 * len(layers)), squeeze=False)
    for r, l in enumerate(layers):
        for c, h in enumerate(heads):
            ax = axes[r][c]
            im = ax.imshow(maps[l, h], cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"Layer {l}, head {h}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            if font:
                ax.set_xticks(range(T))
                ax.set_yticks(range(T))
                ax.set_xticklabels(labels, rotation=90, fontsize=6, fontname=font)
                ax.set_yticklabels(labels, fontsize=6, fontname=font)
            fig.colorbar(im, ax=ax, fraction=0.046, label="Attention weight")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_head_grid(mat, path, title, cbar_label):
    """Layer x head heatmap of a scalar statistic, annotated with values."""
    fig, ax = plt.subplots(figsize=(1.0 * mat.shape[1] + 2.5, 0.55 * mat.shape[0] + 2))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < mat.mean() else "k", fontsize=8)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- entry point
def run(ckpt, out_dir, name, tokens_dir, device, n_seq=32, seq_len=256):
    """Compute stats on held-out text and draw all figures for one checkpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(ckpt, device)
    sp = load_tokenizer()
    stats = attention_stats(model, load_split(tokens_dir, "val", model.cfg.block_size), device, n_seq, seq_len)
    (out / f"{name}_attn_stats.json").write_text(json.dumps(summarize(stats), indent=1))
    plot_head_grid(stats["entropy"], out / f"{name}_entropy.png",
                   f"Attention entropy per head ({name})", "Entropy (nats)")
    plot_head_grid(stats["mean_distance"], out / f"{name}_distance.png",
                   f"Mean attention distance per head ({name})", "Distance (tokens)")
    L, H = model.cfg.n_layer, model.cfg.n_head
    for k, sent in enumerate(SENTENCES):
        ids = sp.encode(sent)
        maps = attention_maps(model, ids, device)
        labels = [sp.id_to_piece(i).replace("▁", "") for i in ids]
        plot_heatmaps(maps, labels, [0, L // 2, L - 1], list(range(min(4, H))), out / f"{name}_heatmap_s{k}.png",
                      f"Attention heatmaps ({name}), sentence {k}")
    return summarize(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="../../report/attn")
    ap.add_argument("--name", default="pretrained")
    ap.add_argument("--tokens", default=str(ROOT / "data" / "tokens"))
    ap.add_argument("--n-seq", type=int, default=32)
    a = ap.parse_args()
    s = run(a.ckpt, a.out_dir, a.name, a.tokens, "cuda" if torch.cuda.is_available() else "cpu", a.n_seq)
    print("local heads:", s["local_heads"], "/", s["total_heads"])
    print("layer mean entropy:", s["layer_mean_entropy"])
    print("layer mean distance:", s["layer_mean_distance"])
