"""Phase 1: compare tokenizers on held-out (validation) text.

Reports per vocab size: chars/token, fertility (tokens per whitespace word),
unknown-token rate, byte-fallback rate, vocab usage; plus examples and a
rank-frequency plot. Computation and plotting are separate functions.

Usage:
    python eval_tokenizer.py --val ../data/processed/val.jsonl --models models --out ../../report
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm

EXAMPLES = ["भारत की राजधानी नई दिल्ली है।", "मुझे कल सुबह छह बजे उठकर स्कूल जाना है।",
            "विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने बड़ी प्रगति की है।"]


def load_docs(path: Path, n_docs: int):
    """Read the first n_docs validation documents."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if len(docs) >= n_docs:
                break
    return docs


def evaluate(sp, docs):
    """Return metrics dict and the token-frequency Counter for one tokenizer."""
    freq, chars, words, toks = Counter(), 0, 0, 0
    for d in docs:
        ids = sp.encode(d)
        freq.update(ids)
        chars += len(d)
        words += len(d.split())
        toks += len(ids)
    unk = freq[sp.unk_id()] / toks
    byte_ids = [i for i in range(sp.get_piece_size()) if sp.is_byte(i)]
    byte_rate = sum(freq[i] for i in byte_ids) / toks
    top100 = sum(c for _, c in freq.most_common(100)) / toks
    return {
        "vocab_size": sp.get_piece_size(), "tokens": toks, "chars_per_token": chars / toks,
        "fertility_tokens_per_word": toks / words, "unk_rate": unk, "byte_fallback_rate": byte_rate,
        "vocab_used_frac": len(freq) / sp.get_piece_size(), "top100_token_mass": top100,
    }, freq


def plot_rank_freq(freqs, out_path: Path):
    """Log-log rank-frequency curves, one line per vocab size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for name, fr in freqs.items():
        plt.loglog(range(1, len(fr) + 1), sorted(fr.values(), reverse=True), label=name)
    plt.title("Token rank-frequency on Hindi validation text")
    plt.xlabel("Token rank")
    plt.ylabel("Token count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="../data/processed/val.jsonl")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="../../report")
    ap.add_argument("--n-docs", type=int, default=20000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    docs = load_docs(Path(a.val), a.n_docs)
    results, freqs, examples = {}, {}, {}
    for m in sorted(Path(a.models).glob("*.model")):
        sp = spm.SentencePieceProcessor(model_file=str(m))
        results[m.stem], freqs[m.stem] = evaluate(sp, docs)
        examples[m.stem] = {s: sp.encode(s, out_type=str) for s in EXAMPLES}
    (out / "tokenizer_eval.json").write_text(
        json.dumps({"metrics": results, "examples": examples}, ensure_ascii=False, indent=1), encoding="utf-8")
    plot_rank_freq(freqs, out / "tokenizer_rank_frequency.png")
    for k, v in results.items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
