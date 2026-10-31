# Monolingual Transformer LM (Hindi)

A decoder-only Transformer language model built from scratch in PyTorch for **Hindi**, with its own corpus,
tokenizer and weights. Course project for *Language Models and Agents* (Monsoon 2026). No pretrained models,
no pretrained tokenizers, no HuggingFace model classes.

**Scope:** one high-resource language (Hindi) only; the lower-resource model and the 20% manual-collection
requirement of the original handout are intentionally skipped, so the manual share is reported as 0%.

## Status

| Stage | State |
|---|---|
| Data collection, cleaning, splits, statistics | Done |
| Tokenizer (SentencePiece BPE, 24k) | Done |
| Transformer implementation (25.4M params), training loop, resume test | Done |
| Pretraining on 500M tokens | Done (final validation loss 3.3565, PPL 28.7) |
| Evaluation (PPL/BPB, BLEU/chrF/ROUGE-L, attention analysis) | Planned |
| Synthetic Hindi reasoning dataset generator | Done |
| Reasoning finetuning script (tested on CPU) | Done |
| Reasoning finetuning results, post-finetune attention comparison, final report | Needs the pretrained checkpoint |

## Repository layout

```
hindi/
  data/        preprocess.py, build_corpus.py, make_stats_report.py   (raw/processed/tokens are git-ignored)
  tokenizer/   train_tokenizer.py, eval_tokenizer.py, models/hindi_bpe_{16,24,32}k.{model,vocab}
  model/       transformer.py, test_model.py
  train/       train.py, test_resume.py
  configs/     hindi_25m.json
  eval/        lm_eval.py (PPL/BPB), gen_eval.py (BLEU/chrF++/ROUGE-L, Distinct-n, repetition), attn_analysis.py, common.py
report/        dataset_stats.md, tokenizer_eval.json, tokenizer_rank_frequency.png
```

## Data

| Source (Kaggle) | Use |
|---|---|
| [Hindi OSCAR Corpus](https://www.kaggle.com/datasets/abhishek/hindi-oscar-corpus) (`hi_dedup.txt`) | bulk web text |
| [Hindi Text Short and Large Summarization Corpus](https://www.kaggle.com/datasets/disisbig/hindi-text-short-and-large-summarization-corpus) (`train.csv`, `test.csv`) | news articles |
| [BBC Hindi News Articles](https://www.kaggle.com/datasets/aadisrivastava/bbc-hindi-news-articles-dataset-detailed/data) | news articles |

**Cleaning** (`preprocess.py`): NFC normalization; URL and zero-width/control-character removal; whitespace
collapsing; drop documents under 50 characters, under 80% Devanagari, over 10% Latin, or containing
legacy-font mojibake (Latin letter inside a Devanagari word); exact deduplication across all sources. The
news train/test CSVs are pooled and re-split. Splits are 99/0.5/0.5 (train/val/test) by document hash.

**Corpus build** (`build_corpus.py`): OSCAR is randomly subsampled to 48%; MinHash near-duplicate removal
(5-word shingles, 128 hashes, 8 bands x 16 rows, roughly 0.85 Jaccard). Val/test are processed first, so
train documents that near-duplicate them are dropped. Output is `uint16` token shards with EOS after each document.

| Split | Tokens |
|---|---|
| Train | 500,000,033 (OSCAR 77.5%, news 21.6%, BBC 1.0%) |
| Validation | 4,795,705 |
| Test | 4,808,583 |

Full tables: [report/dataset_stats.md](report/dataset_stats.md). Known limits: only ~0.85-similarity near-duplicates
are removed, and val/test are ~99% OSCAR web text.

## Tokenizer

SentencePiece BPE trained on train-split sentences only (OSCAR 4%, news 25%, BBC 100%), with byte fallback,
digit splitting and identity normalization. Evaluated on 20k validation documents:

| Vocab | Chars/token | Tokens/word | Unk rate | Byte-fallback rate |
|---|---|---|---|---|
| 16k | 3.91 | 1.293 | 0% | 0.59% |
| **24k (used)** | 4.04 | 1.250 | 0% | 0.61% |
| 32k | 4.12 | 1.227 | 0% | 0.62% |

24k was chosen because going to 32k saves under 2% of tokens but costs ~3M more embedding parameters in a
~25M-parameter model. The unk rate is 0% only because byte fallback is on. Details and examples:
`report/tokenizer_eval.json`, `report/tokenizer_rank_frequency.png`.

## Model

Decoder-only GPT written from `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` / `nn.Dropout` only
(`hindi/model/transformer.py`). Attention, the causal mask and positional embeddings are hand-written;
`nn.Transformer*` and `scaled_dot_product_attention` are not used.

| Setting | Value | Why |
|---|---|---|
| Parameters | **25,383,552** | target ~25M |
| d_model / layers / heads | 384 / 9 / 6 (d_k = 64) | 24k-vocab embeddings take 9.2M, leaving ~16M for 9 blocks |
| FFN | 384 -> 1536 -> 384, GELU | inner width 4x d_model |
| Norm | pre-norm LayerNorm + final LN | unnormalized residual path is more stable for deep stacks |
| Positions | learned absolute, 512 | simple and fast; hard-limits context to 512 tokens |
| Output head | tied with token embedding | saves 9.2M parameters |
| Dropout | 0.05 | one pass over 500M tokens (~20 tokens/param) barely overfits, so little is needed |

Checks (`python hindi/model/test_model.py`): parameter count; initial loss close to ln(24000);
changing token t+1 leaves logits at positions <= t unchanged (max difference 0.0); attention rows sum to 1
with zeros above the diagonal.

## Training

`hindi/train/train.py`: AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), 400-step linear warmup then
cosine decay (6e-4 to 6e-5), gradient clipping 1.0, 65,536 tokens per step (32 x 4 x 512), 7,630 steps (~1 epoch of
the 500M tokens), bf16/fp16 autocast on GPU, validation every 250 steps. Checkpoints (`ckpt_latest.pt`, atomic write)
contain weights, optimizer, scaler, step, tokens seen, RNG state, config and loss history; rerunning the same
command resumes automatically. Data order depends only on (seed, step), so a resumed run is identical to an
uninterrupted one (`python hindi/train/test_resume.py` verifies this).

This machine has no NVIDIA GPU (CPU speed is ~550 tokens/s), so pretraining runs on Kaggle with
[`hindi/train/kaggle_train.ipynb`](hindi/train/kaggle_train.ipynb):

1. Create a private Kaggle dataset `hindi-tokens` from the files in `hindi/data/tokens/` (`train_*.bin`, `val_000.bin`, `test_000.bin`).
2. New notebook -> File -> Import notebook -> `kaggle_train.ipynb`. Settings: Accelerator **GPU T4**, Internet **On**; add the `hindi-tokens` dataset as input.
3. Save Version -> "Save & Run All (Commit)". Training stops itself after ~10.8 h with a checkpoint.
4. To continue: open a new session of the same notebook, add the previous version's output as an input, and Save & Run All again. The notebook finds `ckpt_latest.pt` and resumes.
5. Download `runs/hindi_25m/` (checkpoints, `train_log.jsonl`, `loss_curve.png`) from the notebook output and upload to Drive.

After pretraining, [`hindi/train/kaggle_eval_finetune.ipynb`](hindi/train/kaggle_eval_finetune.ipynb) runs the whole
evaluation, finetuning and attention pipeline on Kaggle (inputs: `hindi-tokens` + the pretraining notebook's output).

## Reasoning dataset (Phase 3)

`python hindi/reasoning/generate.py --out hindi/reasoning/data` (deterministic; the jsonl files are not stored in git) writes train (60k), val (1k), test_iid (1.5k) and
test_ood (1.5k) as jsonl, with `stats.json`. Examples are built programmatically in Hindi, so answers are exact:
transitive ordering from 3-4 shuffled pairwise statements ("राम सीता से लंबा है। सीता गीता से लंबा है।"), and
comparing 2-3 stated numbers (height, age, weight, speed, price). Questions are superlative ("सबसे ... कौन है?"),
pair ("X और Y में से कौन ...?") or yes/no ("क्या X Y से ... है?"). Each example has a short reasoning line
(ordering, then the conclusion) before `उत्तर:`. Adjectives agree with the subject's gender and masculine item
nouns take the oblique form before postpositions.

Leakage control: person and item names are disjoint across train/val/test (training uses ~1,450 real and made-up names so the model must copy names, not memorize them; v1 used only ~55 and the model failed on unseen names); `test_ood` also uses statement and
question templates never seen in training; duplicate prompts are removed across all splits (64,000 unique).

### Finetuning

```bash
python hindi/reasoning/finetune.py --pretrained <runs/hindi_25m/ckpt_best.pt> --out-dir runs/ft
```
Evaluates the pretrained model zero-shot with the same prompt format, finetunes (AdamW lr 3e-4, 3 epochs, batch 32,
loss on the reasoning + answer tokens only, tokenizer fixed), then evaluates again. Metric: exact match on the text
after `उत्तर:` (greedy decoding) on val / test_iid / test_ood, broken down by task kind and question type, plus the
fraction of examples whose reasoning line is reproduced exactly. Per-example predictions are saved for error
analysis. Resumable like pretraining (rerun the same command). `--no-cot` trains on answers only.

## Evaluation

All scripts take `--ckpt` and run from `hindi/eval/` (the last cell of the Kaggle notebook runs them for you):

```bash
pip install sacrebleu matplotlib
python lm_eval.py   --ckpt <ckpt_best.pt> --out ../../report/lm_eval.json       # val/test loss, PPL, BPB
python gen_eval.py  --ckpt <ckpt_best.pt> --out-prefix ../../report/gen          # greedy, T=0.5/1.0/1.5
python attn_analysis.py --ckpt <ckpt_best.pt> --out-dir ../../report/attn --name pretrained
```

- **PPL / BPB:** non-overlapping 512-token windows over the whole split; BPB = total NLL in bits / UTF-8 bytes of the target tokens (EOS counts 0 bytes).
- **Generation:** 500 held-out prefixes (64 tokens), 64-token continuations scored against the true continuation: corpus BLEU-4 and chrF++ (sacrebleu), ROUGE-L F1, Distinct-1/2 and 4-gram repetition rate. Samples go to `gen_samples.md`.
- **Attention:** heatmaps for layers 0, middle and last (4 heads, Hindi example sentences), per-head entropy and mean attention distance on held-out text. An untrained model gives near-uniform attention (mean distance ~T/4), which is the baseline to compare against. With `--reasoning-data <jsonl>` the same statistics and heatmaps are also computed on reasoning prompts (used for the pretrained-vs-finetuned comparison).

## Reproduce (data and tokenizer)

```bash
pip install sentencepiece numpy matplotlib
# 1. download the three Kaggle datasets into hindi/data/raw/
#    (hi_dedup.txt, train.csv, test.csv, bbc_hindi_articles_with_categories_cleaned.csv)
cd hindi/data
python preprocess.py --raw raw --out processed                  # ~25 min
cd ../tokenizer
python train_tokenizer.py --data ../data/processed/train.jsonl  # ~3 min per vocab size
python eval_tokenizer.py
cd ../data
python build_corpus.py                                          # ~10 min, writes tokens/
python make_stats_report.py
```

## Large artifacts

Datasets, token shards and checkpoints are not in git. They are in this Google Drive folder:
https://drive.google.com/drive/folders/1L8NNp3xsBACBS-_swCmaLK2qst1BUlKb?usp=sharing

| File | What it is |
|---|---|
| `ckpt_best.pt` | pretrained Hindi 25M model (weights + config) |
| `ckpt_finetuned.pt` | model finetuned on the v2 synthetic reasoning data |
| `ckpt_latest.pt` | full resumable pretraining checkpoint (weights, optimizer, step, config) |

Load a checkpoint with `hindi/eval/common.py:load_model`. The tokenizer files are in `hindi/tokenizer/models/`; the token
shards can be regenerated with the commands under "Reproduce".
