"""Phase 3: finetune the pretrained Hindi LM on the synthetic reasoning data, and evaluate before / after.

Training text = prompt + reasoning + "\nउत्तर: <answer>" + EOS. The loss is computed on everything after the
prompt only (prompt tokens get label -100, which F.cross_entropy ignores). Same resume-capable checkpoint
format as pretraining: rerun the same command after an interruption.

Evaluation = greedy decoding from the prompt; exact match on the text after the last "उत्तर:".
The pretrained model is evaluated first with the same prompt format (zero-shot baseline).

Usage:
    python finetune.py --pretrained ../train/runs/hindi_25m/ckpt_best.pt --out-dir runs/ft
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
from common import load_model, load_tokenizer  # noqa: E402
from train import build_optimizer, lr_at  # noqa: E402

ANSWER_TAG = "\nउत्तर:"


# ---------------------------------------------------------------- data
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def encode_example(sp, ex, with_cot=True):
    """(input ids, labels). Prompt and tail are tokenized separately so the prompt is identical at inference."""
    p = sp.encode(ex["prompt"])
    tail = sp.encode((ex["cot"] if with_cot else "") + f"{ANSWER_TAG} {ex['answer']}") + [sp.eos_id()]
    ids = p + tail
    return ids, [-100] * len(p) + tail


def collate(batch, pad_id=0):
    """Right-pad to the longest sequence; shift so position t predicts token t+1."""
    n = max(len(i) for i, _ in batch)
    x = torch.full((len(batch), n - 1), pad_id, dtype=torch.long)
    y = torch.full((len(batch), n - 1), -100, dtype=torch.long)
    for k, (ids, lab) in enumerate(batch):
        x[k, :len(ids) - 1] = torch.tensor(ids[:-1])
        y[k, :len(ids) - 1] = torch.tensor(lab[1:])
    return x, y


# ---------------------------------------------------------------- evaluation
def extract_answer(text):
    """Text after the last 'उत्तर:' (stripped), or '' if the model never produced it."""
    return text.rsplit("उत्तर:", 1)[1].strip() if "उत्तर:" in text else ""


@torch.no_grad()
def predict(model, sp, rows, device, max_new=None, batch=64):
    """Greedy continuations. Prompts are bucketed by length so each batch needs no padding."""
    max_new = max_new or MAX_NEW
    order = sorted(range(len(rows)), key=lambda i: len(sp.encode(rows[i]["prompt"])))
    enc = {i: sp.encode(rows[i]["prompt"]) for i in order}
    preds = [None] * len(rows)
    buckets = defaultdict(list)
    for i in order:
        buckets[len(enc[i])].append(i)
    for L, idxs in buckets.items():
        for s in range(0, len(idxs), batch):
            ii = idxs[s:s + batch]
            x = torch.tensor([enc[i] for i in ii], device=device)
            out = model.generate(x, min(max_new, model.cfg.block_size - L), temperature=0, eos_id=sp.eos_id())
            for i, row in zip(ii, out[:, L:].tolist()):
                row = row[:row.index(sp.eos_id())] if sp.eos_id() in row else row
                preds[i] = sp.decode(row)
    return preds


def evaluate(model, sp, rows, device):
    """Exact-match accuracy overall and by kind / question type, plus per-example records."""
    preds = predict(model, sp, rows, device)
    recs, by = [], defaultdict(list)
    for r, p in zip(rows, preds):
        ok = extract_answer(p) == r["answer"]
        cot_ok = p.strip().startswith(r["cot"].strip())          # reasoning line reproduced exactly
        recs.append({"id": r["id"], "kind": r["kind"], "qtype": r["qtype"], "gold": r["answer"], "pred": p, "ok": ok, "cot_ok": cot_ok})
        for key in ("all", r["kind"], r["qtype"]):
            by[key].append(ok)
    acc = {k: float(np.mean(v)) for k, v in sorted(by.items())}
    acc["cot_exact"] = float(np.mean([x["cot_ok"] for x in recs]))
    return acc, recs


def eval_all(model, sp, data_dir, device, out_dir, tag, n=None):
    """Evaluate val / test_iid / test_ood; write accuracy json and per-example predictions."""
    res = {}
    for split in ("val", "test_iid", "test_ood"):
        rows = load_jsonl(Path(data_dir) / f"{split}.jsonl")[:n]
        res[split], recs = evaluate(model, sp, rows, device)
        with open(Path(out_dir) / f"preds_{tag}_{split}.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
        print(tag, split, {k: round(v, 3) for k, v in res[split].items() if k in ("all", "cot_exact")}, flush=True)
    (Path(out_dir) / f"acc_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- training
def finetune(model, sp, args, device):
    """AdamW + warmup/cosine over `epochs`; atomic resumable checkpoints; returns loss history."""
    out = Path(args.out_dir)
    train_rows = load_jsonl(Path(args.data) / "train.jsonl")[:args.max_train]
    data = [encode_example(sp, r, not args.no_cot) for r in train_rows]
    steps_per_epoch = len(data) // args.batch
    tc = {"lr": args.lr, "min_lr": args.lr / 10, "warmup_steps": 50, "max_steps": steps_per_epoch * args.epochs,
          "weight_decay": 0.01, "betas": [0.9, 0.95]}
    opt = build_optimizer(model, tc)
    step, history, ck = 0, [], out / "ckpt_latest.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        step, history = s["step"], s["history"]
        print(f"resumed finetuning from step {step}", flush=True)
    model.train()
    t0 = time.time()
    while step < tc["max_steps"]:
        ep, pos = divmod(step, steps_per_epoch)
        perm = np.random.RandomState(args.seed + ep).permutation(len(data))   # order depends only on (seed, step)
        x, y = collate([data[j] for j in perm[pos * args.batch:(pos + 1) * args.batch]])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        loss = model(x.to(device), y.to(device))[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        history.append({"step": step, "epoch": step / steps_per_epoch, "train_loss": loss.item(), "lr": lr_at(step - 1, tc)})
        if step % 50 == 0:
            print(json.dumps(history[-1]), f"| {time.time() - t0:.0f}s", flush=True)
        if step % args.ckpt_every == 0 or step == tc["max_steps"]:
            tmp = f"{ck}.tmp"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "history": history,
                        "config": {**json.load(open(out / "pretrain_config.json")), "finetune": vars(args)}}, tmp)
            os.replace(tmp, ck)
    model.eval()
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="pretraining checkpoint (ckpt_best.pt)")
    ap.add_argument("--data", default=str(ROOT / "reasoning" / "data"))
    ap.add_argument("--out-dir", default="runs/ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cot", action="store_true", help="train/eval on answer only (no reasoning line)")
    ap.add_argument("--eval-n", type=int, help="limit examples per eval split (smoke tests)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--max-train", type=int, help="use only the first N training examples (smoke tests)")
    ap.add_argument("--max-new", type=int, default=96)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    globals()["MAX_NEW"] = args.max_new
    sp = load_tokenizer()
    model, cfg = load_model(args.pretrained, device)
    json.dump(cfg, open(out / "pretrain_config.json", "w"), indent=1)
    if not args.skip_baseline and not (out / "acc_pretrained.json").exists():
        eval_all(model, sp, args.data, device, out, "pretrained", args.eval_n)     # zero-shot baseline
        model, _ = load_model(args.pretrained, device)                              # fresh weights for finetuning
    hist = finetune(model, sp, args, device)
    (out / "finetune_log.jsonl").write_text("\n".join(json.dumps(h) for h in hist), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": cfg}, out / "ckpt_finetuned.pt")
    eval_all(model, sp, args.data, device, out, "finetuned", args.eval_n)
