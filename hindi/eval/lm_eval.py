"""Intrinsic LM metrics on a held-out split: cross-entropy, perplexity, bits-per-byte.

PPL = exp(mean cross-entropy per token). BPB = total NLL (bits) / total UTF-8 bytes of the target
text, which is comparable across tokenizers. The split is cut into non-overlapping windows of
`block_size` tokens; every target token is scored exactly once.

Usage: python lm_eval.py --ckpt runs/hindi_25m/ckpt_best.pt --out ../../report/lm_eval.json
"""
import argparse
import json
import math

import torch

from common import ROOT, byte_lengths, load_model, load_split, load_tokenizer


@torch.no_grad()
def evaluate_split(model, shards, blen, device, batch=16, max_chunks=None):
    """Return dict(loss, ppl, bpb, tokens, bytes) over the split."""
    n = shards.n_chunks if max_chunks is None else min(max_chunks, shards.n_chunks)
    nll, toks, nbytes = 0.0, 0, 0
    for s in range(0, n, batch):
        x, y = shards.batch(list(range(s, min(s + batch, n))))
        loss = model(x.to(device), y.to(device))[1].item()
        nll += loss * y.numel()                      # mean loss * count = summed NLL (nats)
        toks += y.numel()
        nbytes += int(blen[y.numpy()].sum())
    ce = nll / toks
    return {"loss": ce, "ppl": math.exp(ce), "bpb": nll / math.log(2) / nbytes, "tokens": toks, "bytes": nbytes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", default=str(ROOT / "data" / "tokens"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max-chunks", type=int)
    ap.add_argument("--out", default="../../report/lm_eval.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(a.ckpt, dev)
    blen = byte_lengths(load_tokenizer())
    res = {sp: evaluate_split(model, load_split(a.tokens, sp, model.cfg.block_size), blen, dev,
                              max_chunks=a.max_chunks) for sp in a.splits}
    res["params"] = model.n_params()
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
