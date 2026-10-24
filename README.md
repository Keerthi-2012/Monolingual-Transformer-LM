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
| Pretraining on 500M tokens | Needs a GPU (Colab/Kaggle) |
| Evaluation (PPL/BPB, BLEU/chrF/ROUGE-L, attention analysis) | Planned |
| Reasoning finetuning (synthetic Hindi comparisons) | Planned |

## Repository layout

```
hindi/
  data/        preprocess.py, build_corpus.py, make_stats_report.py   (raw/processed/tokens are git-ignored)
  tokenizer/   train_tokenizer.py, eval_tokenizer.py, models/hindi_bpe_{16,24,32}k.{model,vocab}
  model/       transformer.py, test_model.py
  train/       train.py, test_resume.py
  configs/     hindi_25m.json
  eval/                                                                  (upcoming)
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

This machine has no NVIDIA GPU (CPU speed is ~550 tokens/s), so pretraining is run on Colab/Kaggle:

```bash
# upload hindi/data/tokens/*.bin to Drive, mount it, then:
python hindi/train/train.py --config hindi/configs/hindi_25m.json --token-dir /content/drive/MyDrive/tokens        --out-dir /content/drive/MyDrive/runs/hindi_25m     # re-run the same command after a disconnect
```

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

Datasets, token shards and checkpoints are not in git. Google Drive links: _to be added_.
