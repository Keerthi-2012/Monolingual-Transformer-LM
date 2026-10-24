"""Generation quality: continue held-out prefixes, score against the true continuations.

Prefix = first `--prefix-len` tokens of a held-out window (windows containing EOS are skipped);
reference = the next `--cont-len` tokens. Decodings: greedy and temperatures 0.5 / 1.0 / 1.5.
Metrics: corpus BLEU-4, chrF++ (sacrebleu), ROUGE-L F1 (own implementation), Distinct-1/2 and
repetition rate over generated token ids. Samples are written to a markdown file.

Usage: python gen_eval.py --ckpt runs/hindi_25m/ckpt_best.pt --out-prefix ../../report/gen
"""
import argparse
import json
import re

import numpy as np
import sacrebleu
import torch

from common import ROOT, load_model, load_split, load_tokenizer

DECODINGS = {"greedy": 0.0, "T=0.5": 0.5, "T=1.0": 1.0, "T=1.5": 1.5}


def make_prompts(shards, n, prefix_len, cont_len, eos_id, seed=0):
    """Pick n random held-out windows with no EOS inside; return (prefix, reference) id arrays."""
    rng = np.random.RandomState(seed)
    P, R = [], []
    for c in rng.permutation(shards.n_chunks):
        w = shards.read(int(c) * shards.T, prefix_len + cont_len)
        if eos_id not in w:
            P.append(w[:prefix_len].astype(np.int64))
            R.append(w[prefix_len:].astype(np.int64))
        if len(P) == n:
            break
    return np.stack(P), np.stack(R)


@torch.no_grad()
def generate(model, prefixes, cont_len, temperature, device, eos_id, batch=50):
    """Batched sampling; each continuation is cut at its first EOS. Returns a list of id lists."""
    outs = []
    for s in range(0, len(prefixes), batch):
        x = torch.from_numpy(prefixes[s:s + batch]).to(device)
        y = model.generate(x, cont_len, temperature=temperature)[:, x.size(1):].cpu().numpy()
        for row in y:
            row = [int(t) for t in row]
            outs.append(row[:row.index(eos_id)] if eos_id in row else row)
    return outs


def prep(text):
    """Split the danda from words so it counts as its own token for BLEU/ROUGE."""
    return re.sub(r"\s+", " ", text.replace("।", " । ")).strip()


def rouge_l(hyp, ref):
    """ROUGE-L F1 on whitespace tokens via longest common subsequence."""
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if h[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


def distinct_n(seqs, n):
    """Unique n-grams / total n-grams over all generated sequences."""
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def repetition_rate(seqs, n=4):
    """Fraction of n-grams that already occurred earlier in the same continuation."""
    rep = tot = 0
    for s in seqs:
        seen = set()
        for i in range(len(s) - n + 1):
            g = tuple(s[i:i + n])
            rep += g in seen
            tot += 1
            seen.add(g)
    return rep / max(tot, 1)


def score(sp, gens, refs):
    """All metrics for one decoding setting (pure computation, no I/O)."""
    hyp = [prep(sp.decode([int(t) for t in g])) for g in gens]
    ref = [prep(sp.decode([int(t) for t in r])) for r in refs]
    return {
        "bleu4": sacrebleu.corpus_bleu(hyp, [ref]).score,
        "chrf++": sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score,
        "rougeL": 100 * float(np.mean([rouge_l(h, r) for h, r in zip(hyp, ref)])),
        "distinct1": distinct_n(gens, 1), "distinct2": distinct_n(gens, 2),
        "repetition4": repetition_rate(gens), "avg_len_tokens": float(np.mean([len(g) for g in gens])),
    }, hyp, ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", default=str(ROOT / "data" / "tokens"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--out-prefix", default="../../report/gen")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.ckpt, dev)
    sp = load_tokenizer()
    P, R = make_prompts(load_split(a.tokens, a.split, model.cfg.block_size), a.n, a.prefix_len, a.cont_len, sp.eos_id())
    results, md = {}, ["# Generated samples (first 5 prompts per decoding)\n"]
    for name, temp in DECODINGS.items():
        torch.manual_seed(0)
        gens = generate(model, P, a.cont_len, temp, dev, sp.eos_id())
        results[name], hyp, ref = score(sp, gens, R)
        md.append(f"## {name}\n")
        for i in range(5):
            md.append(f"**Prefix:** {sp.decode([int(t) for t in P[i]])}\n\n**Generated:** {hyp[i]}\n\n**Reference:** {ref[i]}\n")
    json.dump(results, open(f"{a.out_prefix}_metrics.json", "w"), indent=1)
    open(f"{a.out_prefix}_samples.md", "w", encoding="utf-8").write("\n".join(md))
    for k, v in results.items():
        print(k, {m: round(x, 3) for m, x in v.items()})
