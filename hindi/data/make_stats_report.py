"""Phase 1: render report/dataset_stats.md from the preprocessing and corpus stats JSONs."""
import json
from pathlib import Path

pre = json.loads(Path("processed/stats_preprocess.json").read_text(encoding="utf-8"))["counts"]
cor = json.loads(Path("tokens/stats_corpus.json").read_text())
out = ["# Hindi dataset statistics (Model H)\n",
       "Manual-collection share: **0%** (all text is downloaded; the 20% manual requirement was skipped by choice).\n",
       "## 1. Cleaning (all raw text, before subsampling)\n",
       "| Source | Read | Kept | Too short | Low Devanagari | Too much Latin | Mojibake word | Exact duplicate |",
       "|---|---|---|---|---|---|---|---|"]
for s in ("oscar", "news", "bbc"):
    g = lambda k: pre.get(f"{s}:{k}", 0)
    out.append(f"| {s} | {g('read'):,} | {g('kept'):,} | {g('drop_too_short'):,} | {g('drop_low_devanagari'):,} "
               f"| {g('drop_too_much_latin'):,} | {g('drop_mixed_script_word'):,} | {g('drop_duplicate'):,} |")
out += ["\n## 2. Final tokenized splits (24k BPE tokenizer)\n",
        "| Split | Source | Docs | Characters | Tokens | Near-dups removed |", "|---|---|---|---|---|---|"]
tot = {}
for sp in ("train", "val", "test"):
    for s in ("oscar", "news", "bbc"):
        g = lambda k: cor.get(f"{sp}:{s}:{k}", 0)
        out.append(f"| {sp} | {s} | {g('docs'):,} | {g('chars'):,} | {g('tokens'):,} | {g('near_duplicate'):,} |")
        tot[sp] = tot.get(sp, 0) + g("tokens")
out.append("\n| Split | Total tokens |\n|---|---|")
out += [f"| {k} | {v:,} |" for k, v in tot.items()]
mix = ", ".join("%s %.1f%%" % (s, 100 * cor["train:%s:tokens" % s] / tot["train"]) for s in ("oscar", "news", "bbc"))
out.append("\nTrain mix: " + mix + ".\n")
out.append("Notes: OSCAR was randomly subsampled to 48% before near-duplicate removal "
           f"({cor['train:oscar_subsampled_out']:,} documents skipped); the build stopped at the 500M-token cap. "
           "Splits are by document hash, so no document is in two splits; train documents that near-duplicate a "
           "val/test document (MinHash, 8 bands x 16 rows, ~0.85 Jaccard) were removed.")
Path("../../report").mkdir(exist_ok=True)
Path("../../report/dataset_stats.md").write_text("\n".join(out) + "\n", encoding="utf-8")
