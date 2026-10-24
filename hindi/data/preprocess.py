"""Phase 1: clean, deduplicate and split the raw Hindi corpora.

Pipeline (streaming, so the 9.5 GB OSCAR file never sits in memory):
    raw source -> normalize -> filter -> exact-dedup -> hash-based split -> jsonl

Each output line is {"src": <source name>, "text": <document>}.
Splits are decided by a hash of the document text, so they are deterministic
and a document can never land in two splits.

Usage:
    python preprocess.py --raw raw --out processed            # full run
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

csv.field_size_limit(2**30)  # news articles exceed the default 128 KB field limit

# ---------------------------------------------------------------- cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
# Control chars (except \n, \t), zero-width space/BOM/word-joiner/soft hyphen.
# ZWJ/ZWNJ (U+200C/D) are also dropped: they are inconsistently used in web Hindi.
JUNK_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏⁠﻿­�]")
HSPACE_RE = re.compile(r"[ \t  -  　]+")
BLANKS_RE = re.compile(r"\n{2,}")
DEVA_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
# U+0964/0965 are the danda / double danda, which belong to Devanagari block already.


def normalize(text: str) -> str:
    """NFC-normalize, strip URLs/junk characters and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = JUNK_RE.sub("", text)
    text = HSPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANKS_RE.sub("\n", text).strip()


def keep(text: str, min_chars: int, min_deva: float, max_latin: float):
    """Return (keep?, reason). Filters too-short docs and non-Hindi / garbled docs."""
    if len(text) < min_chars:
        return False, "too_short"
    n = len(text.replace(" ", "").replace("\n", ""))
    deva = len(DEVA_RE.findall(text)) / max(n, 1)
    if deva < min_deva:
        return False, "low_devanagari"
    if len(LATIN_RE.findall(text)) / max(n, 1) > max_latin:
        return False, "too_much_latin"
    # Legacy-font mojibake shows up as Latin letters glued inside Devanagari words.
    if re.search(r"[ऀ-ॿ][A-Za-z][ऀ-ॿ]", text):
        return False, "mixed_script_word"
    return True, ""


# ---------------------------------------------------------------- sources
def iter_oscar(path: Path):
    """OSCAR dump: one document per line."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            yield "oscar", line


def iter_csv(path: Path, src: str, cols):
    """Yield each listed column of every row as its own document."""
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for c in cols:
                if row.get(c):
                    yield src, row[c]


def split_of(digest: bytes, val_frac: float, test_frac: float) -> str:
    """Deterministic split from the first 4 bytes of the document hash."""
    u = int.from_bytes(digest[:4], "big") / 2**32
    return "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--oscar", default="hi_dedup.txt")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--min-deva", type=float, default=0.80)
    ap.add_argument("--max-latin", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--test-frac", type=float, default=0.005)
    a = ap.parse_args()

    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    handles = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}

    # News train/test CSVs are pooled: we re-split everything ourselves.
    streams = [
        iter_csv(raw / "train.csv", "news", ["article"]),
        iter_csv(raw / "test.csv", "news", ["article"]),
        iter_csv(raw / "bbc_hindi_articles_with_categories_cleaned.csv", "bbc", ["Content"]),
        iter_oscar(raw / a.oscar),
    ]
    seen = set()
    stats = Counter()
    chars = Counter()
    for stream in streams:
        for src, text in stream:
            stats[f"{src}:read"] += 1
            text = normalize(text)
            ok, why = keep(text, a.min_chars, a.min_deva, a.max_latin)
            if not ok:
                stats[f"{src}:drop_{why}"] += 1
                continue
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            if digest in seen:
                stats[f"{src}:drop_duplicate"] += 1
                continue
            seen.add(digest)
            sp = split_of(digest, a.val_frac, a.test_frac)
            handles[sp].write(json.dumps({"src": src, "text": text}, ensure_ascii=False) + "\n")
            stats[f"{src}:kept"] += 1
            stats[f"{src}:{sp}"] += 1
            chars[f"{src}:{sp}"] += len(text)
    for h in handles.values():
        h.close()

    report = {"counts": dict(sorted(stats.items())), "chars": dict(sorted(chars.items())), "args": vars(a)}
    (out / "stats_preprocess.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], indent=1))


if __name__ == "__main__":
    main()
