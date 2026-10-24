# Hindi dataset statistics (Model H)

Manual-collection share: **0%** (all text is downloaded; the 20% manual requirement was skipped by choice).

## 1. Cleaning (all raw text, before subsampling)

| Source | Read | Kept | Too short | Low Devanagari | Too much Latin | Mojibake word | Exact duplicate |
|---|---|---|---|---|---|---|---|
| oscar | 11,469,844 | 10,192,373 | 550 | 976,788 | 279,434 | 2,541 | 18,158 |
| news | 179,835 | 147,383 | 1,137 | 21,147 | 3,602 | 10 | 6,556 |
| bbc | 4,598 | 3,840 | 3 | 0 | 0 | 1 | 754 |

## 2. Final tokenized splits (24k BPE tokenizer)

| Split | Source | Docs | Characters | Tokens | Near-dups removed |
|---|---|---|---|---|---|
| train | oscar | 4,602,153 | 1,535,491,999 | 387,269,888 | 53,518 |
| train | news | 145,811 | 447,003,986 | 107,776,800 | 161 |
| train | bbc | 3,797 | 20,252,928 | 4,953,345 | 4 |
| val | oscar | 50,632 | 17,006,972 | 4,286,717 | 19 |
| val | news | 696 | 2,018,198 | 485,121 | 0 |
| val | bbc | 19 | 97,173 | 23,867 | 0 |
| test | oscar | 50,884 | 16,954,267 | 4,278,791 | 40 |
| test | news | 715 | 2,051,465 | 494,970 | 0 |
| test | bbc | 20 | 140,522 | 34,822 | 0 |

| Split | Total tokens |
|---|---|
| train | 500,000,033 |
| val | 4,795,705 |
| test | 4,808,583 |

Train mix: oscar 77.5%, news 21.6%, bbc 1.0%.

Notes: OSCAR was randomly subsampled to 48% before near-duplicate removal (5,246,805 documents skipped); the build stopped at the 500M-token cap. Splits are by document hash, so no document is in two splits; train documents that near-duplicate a val/test document (MinHash, 8 bands x 16 rows, ~0.85 Jaccard) were removed.
