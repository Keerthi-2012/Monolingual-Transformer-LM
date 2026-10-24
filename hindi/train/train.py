"""Pretraining loop: AdamW + warmup/cosine LR + periodic validation + resumable checkpoints.

A checkpoint (ckpt_latest.pt, written atomically) holds: model weights, optimizer state,
LR-scheduler state, AMP grad-scaler state, step, tokens seen, RNG states, and the full
config. Data order is a pure function of (seed, step), so resuming reproduces the run.

Usage:
    python train.py --config ../configs/hindi_25m.json            # starts or auto-resumes
    python train.py --config ../configs/hindi_25m.json --max-steps 50 --out-dir runs/smoke
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
from transformer import GPT, GPTConfig  # noqa: E402


class TokenShards:
    """Read-only view over uint16 token shards, addressed as one long token stream."""

    def __init__(self, token_dir, prefix, seq_len):
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in sorted(glob.glob(f"{token_dir}/{prefix}_*.bin"))]
        assert self.shards, f"no {prefix}_*.bin shards in {token_dir}"
        self.ends = np.cumsum([len(s) for s in self.shards])
        self.total, self.T = int(self.ends[-1]), seq_len
        self.n_chunks = (self.total - 1) // seq_len      # non-overlapping windows of T+1 tokens

    def read(self, start, n):
        """n tokens from global offset `start`, stitching across shard boundaries."""
        out, pos = [], start
        while n > 0:
            i = int(np.searchsorted(self.ends, pos, side="right"))
            lo = pos - (int(self.ends[i - 1]) if i else 0)
            take = min(n, len(self.shards[i]) - lo)
            out.append(np.asarray(self.shards[i][lo:lo + take]))
            pos, n = pos + take, n - take
        return np.concatenate(out)

    def batch(self, chunk_ids):
        """Return x, y (B, T): y is x shifted left by one token."""
        w = np.stack([self.read(int(c) * self.T, self.T + 1) for c in chunk_ids]).astype(np.int64)
        return torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:])


class Order:
    """Deterministic chunk order: a fresh seeded permutation per epoch, indexed by global sequence number."""

    def __init__(self, n_chunks, seed):
        self.n, self.seed, self._ep, self._perm = n_chunks, seed, -1, None

    def ids(self, first, count):
        out = []
        for g in range(first, first + count):
            ep, pos = divmod(g, self.n)
            if ep != self._ep:
                self._ep, self._perm = ep, np.random.RandomState(self.seed + ep).permutation(self.n)
            out.append(self._perm[pos])
        return out


def lr_at(step, tc):
    """Linear warmup then cosine decay to min_lr."""
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, tc["max_steps"] - tc["warmup_steps"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * min(p, 1.0)))


def build_optimizer(model, tc):
    """AdamW; weight decay only on matrices (not biases/LayerNorm), as is standard."""
    seen, decay, no_decay = set(), [], []
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": tc["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=tuple(tc["betas"]))


@torch.no_grad()
def evaluate(model, val, tc, device, amp_ctx):
    """Mean validation cross-entropy over the first eval_batches fixed batches."""
    model.eval()
    losses = []
    for b in range(tc["eval_batches"]):
        ids = [(b * tc["micro_batch"] + j) % val.n_chunks for j in range(tc["micro_batch"])]
        x, y = val.batch(ids)
        with amp_ctx:
            losses.append(model(x.to(device), y.to(device))[1].item())
    model.train()
    return float(np.mean(losses))


def save_ckpt(path, model, opt, scaler, step, tokens, cfg, best_val, history):
    """Atomic checkpoint write (tmp file + rename) so a crash never leaves a corrupt file."""
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
             "step": step, "tokens_seen": tokens, "config": cfg, "best_val": best_val, "history": history,
             "scheduler": {"last_step": step}, "rng": {"torch": torch.get_rng_state(),
                                                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def run(cfg, stop_at=None):
    """Train until max_steps (or `stop_at`, used by tests to simulate an interruption)."""
    tc, mc = cfg["train"], GPTConfig(**cfg["model"])
    out = Path(tc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc["seed"])
    amp_dtype = None
    if device == "cuda":
        # bf16 is only native on compute capability >= 8.0 (Ampere+). On a T4 (7.5) PyTorch can emulate it,
        # which is ~10x slower, so use fp16 + GradScaler there.
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8
        amp_dtype = torch.bfloat16 if (tc["dtype"] in ("auto", "bf16") and native_bf16) else torch.float16
    amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    train = TokenShards(tc["token_dir"], "train", mc.block_size)
    val = TokenShards(tc["token_dir"], "val", mc.block_size)
    order = Order(train.n_chunks, tc["seed"])
    model = GPT(mc).to(device)
    opt = build_optimizer(model, tc)
    step, tokens, best_val, history = 0, 0, float("inf"), []

    ck = out / "ckpt_latest.pt"
    if ck.exists():  # ---- resume
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        scaler.load_state_dict(s["scaler"])
        step, tokens, best_val, history = s["step"], s["tokens_seen"], s["best_val"], s["history"]
        torch.set_rng_state(s["rng"]["torch"].cpu())
        if s["rng"]["cuda"] is not None and device == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in s["rng"]["cuda"]])
        print(f"resumed from step {step}", flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"params: {model.n_params():,} | device {device} | dtype {amp_dtype}", flush=True)

    mb, ga, T = tc["micro_batch"], tc["grad_accum"], mc.block_size
    seqs_per_step = mb * ga
    model.train()
    t0, tokens0 = time.time(), tokens
    last = stop_at if stop_at is not None else tc["max_steps"]
    while step < min(last, tc["max_steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        loss_acc = 0.0
        for a in range(ga):
            x, y = train.batch(order.ids(step * seqs_per_step + a * mb, mb))
            with amp_ctx:
                loss = model(x.to(device), y.to(device))[1] / ga
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += seqs_per_step * T
        rec = {"step": step, "train_loss": loss_acc, "lr": lr_at(step - 1, tc), "grad_norm": gn, "tokens": tokens}
        if step % tc["eval_every"] == 0 or step == tc["max_steps"]:
            rec["val_loss"] = evaluate(model, val, tc, device, amp_ctx)
            if rec["val_loss"] < best_val:
                best_val = rec["val_loss"]
                torch.save({"model": model.state_dict(), "config": cfg, "step": step}, out / "ckpt_best.pt")
        history.append(rec)
        if step % tc["log_every"] == 0 or "val_loss" in rec:
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}),
                  f"| {(tokens - tokens0) / (time.time() - t0):,.0f} tok/s", flush=True)
        out_of_time = tc.get("time_limit_min") and (time.time() - t0) / 60 > tc["time_limit_min"]
        if step % tc["ckpt_every"] == 0 or step == tc["max_steps"] or step == stop_at or out_of_time:
            save_ckpt(ck, model, opt, scaler, step, tokens, cfg, best_val, history)
            (out / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in history), encoding="utf-8")
        if out_of_time:
            print(f"time limit reached at step {step}; checkpoint saved, rerun to resume", flush=True)
            break
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../configs/hindi_25m.json")
    for k in ("max_steps", "micro_batch", "grad_accum", "eval_every", "ckpt_every", "eval_batches", "time_limit_min"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float if k == "time_limit_min" else int)
    ap.add_argument("--out-dir")
    ap.add_argument("--token-dir")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    for k, v in vars(a).items():
        if k != "config" and v is not None:
            cfg["train"][k] = v
    run(cfg)
