"""Phase 1: build the tokenized pretraining corpus (~500M train tokens).

Per document: subsample OSCAR -> MinHash signature -> tokenize (in worker processes).
In the main process: near-duplicate check against everything seen so far, then
append token ids to uint16 shards (EOS after each document).

Leakage control: val/test are processed FIRST and their LSH bucket keys are kept,
so any train document near-duplicating a val/test document is dropped.

MinHash: 5-word shingles, 128 hash functions, 8 LSH bands x 16 rows
(flags pairs with Jaccard similarity above roughly 0.85).

Usage:
    python build_corpus.py --proc processed --tok ../tokenizer/models/hindi_bpe_24k.model --out tokens
"""
import argparse
import hashlib
import json
import random
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sentencepiece as spm

N_PERM, BANDS, ROWS, SHINGLE = 128, 8, 16, 5
_rng = np.random.RandomState(1234)
A = _rng.randint(1, 2**62, size=N_PERM, dtype=np.uint64) | np.uint64(1)
B = _rng.randint(0, 2**62, size=N_PERM, dtype=np.uint64)
_sp = None


def init_worker(tok_path):
    """Load the tokenizer once per worker process."""
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=tok_path)


def lsh_keys(text: str):
    """Return BANDS integer bucket keys for a document, or [] if too short to shingle."""
    w = text.split()
    if len(w) < SHINGLE + 3:
        return []
    h = np.fromiter((zlib.crc32(" ".join(w[i:i + SHINGLE]).encode()) for i in range(len(w) - SHINGLE + 1)),
                    dtype=np.uint64)
    h = np.unique(h)
    # uint64 multiply wraps modulo 2^64, which is fine as a cheap universal-style hash
    sig = (h[:, None] * A[None, :] + B[None, :]).min(axis=0)
    # blake2b, not hash(): Python's hash() of bytes is randomized per process
    return [int.from_bytes(hashlib.blake2b(bytes([b]) + sig[b * ROWS:(b + 1) * ROWS].tobytes(),
                                           digest_size=8).digest(), "big") for b in range(BANDS)]


def work(line: str):
    """Worker: parse one jsonl line, compute LSH keys and token ids."""
    d = json.loads(line)
    ids = np.asarray(_sp.encode(d["text"]) + [_sp.eos_id()], dtype=np.uint16)
    return d["src"], len(d["text"]), lsh_keys(d["text"]), ids


class ShardWriter:
    """Append uint16 token ids to fixed-size shards <prefix>_000.bin, ..."""

    def __init__(self, prefix: Path, shard_tokens: int = 100_000_000):
        self.prefix, self.cap, self.i, self.n, self.f = prefix, shard_tokens, 0, 0, None

    def write(self, ids):
        if self.f is None or self.n >= self.cap:
            if self.f:
                self.f.close()
            self.f = open(f"{self.prefix}_{self.i:03d}.bin", "wb")
            self.i, self.n = self.i + 1, 0
        ids.tofile(self.f)
        self.n += len(ids)

    def close(self):
        if self.f:
            self.f.close()


def process_split(pool, path, seen, writer, stats, name, p_oscar=1.0, token_cap=None, seed=0):
    """Stream one split; dedup against `seen`; write tokens; update stats."""
    rng = random.Random(seed)

    def lines():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if p_oscar < 1.0 and '"src": "oscar"' in line[:20] and rng.random() > p_oscar:
                    stats[f"{name}:oscar_subsampled_out"] += 1
                    continue
                yield line

    total = 0
    for src, nchar, keys, ids in pool.imap(work, lines(), chunksize=500):
        if keys and any(k in seen for k in keys):
            stats[f"{name}:{src}:near_duplicate"] += 1
            continue
        seen.update(keys)
        writer.write(ids)
        total += len(ids)
        stats[f"{name}:{src}:docs"] += 1
        stats[f"{name}:{src}:chars"] += nchar
        stats[f"{name}:{src}:tokens"] += len(ids)
        if token_cap and total >= token_cap:
            stats[f"{name}:stopped_at_cap"] = 1
            break
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="processed")
    ap.add_argument("--tok", default="../tokenizer/models/hindi_bpe_24k.model")
    ap.add_argument("--out", default="tokens")
    ap.add_argument("--p-oscar", type=float, default=0.48)
    ap.add_argument("--train-tokens", type=int, default=500_000_000)
    a = ap.parse_args()
    proc, out = Path(a.proc), Path(a.out)
    out.mkdir(exist_ok=True)
    stats, seen = Counter(), set()
    with Pool(14, initializer=init_worker, initargs=(a.tok,)) as pool:
        for name in ("val", "test", "train"):  # eval splits first: their keys block train near-dups
            w = ShardWriter(out / name)
            cap = a.train_tokens if name == "train" else None
            p = a.p_oscar if name == "train" else 1.0
            n = process_split(pool, proc / f"{name}.jsonl", seen, w, stats, name, p, cap)
            w.close()
            print(name, "tokens:", n, flush=True)
    (out / "stats_corpus.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(stats.items())), indent=1))
