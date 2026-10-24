"""Phase 1: train separate SentencePiece BPE tokenizers (several vocab sizes).

Step 1 samples sentences from the TRAIN split only (val/test never touch the
tokenizer), stratified by source. Step 2 trains one BPE model per vocab size.

Usage:
    python train_tokenizer.py --data ../data/processed/train.jsonl --out models \
        --vocab-sizes 16000 24000 32000
"""
import argparse
import json
import random
import re
from pathlib import Path

import sentencepiece as spm

# Sentence boundary: danda, double danda, ?, !, . and newlines. SentencePiece skips
# lines longer than max_sentence_length bytes, so whole documents cannot be fed in.
SENT_SPLIT = re.compile(r"(?<=[।॥?!.])\s+|\n+")
# Fraction of documents kept per source: OSCAR dominates the corpus, so it is
# down-weighted to keep news/BBC represented in the tokenizer training text.
SRC_PROB = {"oscar": 0.04, "news": 0.25, "bbc": 1.0}


def sample_sentences(train_path: Path, out_path: Path, seed: int = 0, max_len: int = 1000):
    """Write one sentence per line, sampled by source, from the train split."""
    rng = random.Random(seed)
    n = 0
    with open(train_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        for line in f:
            d = json.loads(line)
            if rng.random() > SRC_PROB[d["src"]]:
                continue
            for s in SENT_SPLIT.split(d["text"]):
                s = s.strip()
                if 5 <= len(s) <= max_len:
                    w.write(s + "\n")
                    n += 1
    return n


def train(sample: Path, out_dir: Path, vocab_size: int, max_sentences: int):
    """Train one BPE model; byte_fallback keeps unknown-token rate near zero."""
    prefix = str(out_dir / f"hindi_bpe_{vocab_size // 1000}k")
    spm.SentencePieceTrainer.train(
        input=str(sample), model_prefix=prefix, model_type="bpe", vocab_size=vocab_size,
        character_coverage=0.9995, byte_fallback=True, normalization_rule_name="identity",
        input_sentence_size=max_sentences, shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, num_threads=16,
        split_digits=True, max_sentence_length=4096,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed/train.jsonl")
    ap.add_argument("--out", default="models")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[16000, 24000, 32000])
    ap.add_argument("--max-sentences", type=int, default=4_000_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    sample = out / "tokenizer_sample.txt"
    if not sample.exists():
        print("sampled sentences:", sample_sentences(Path(a.data), sample))
    for v in a.vocab_sizes:
        train(sample, out, v, a.max_sentences)
        print("trained", v)
