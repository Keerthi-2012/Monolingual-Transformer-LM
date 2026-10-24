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
| Transformer implementation, pretraining (~25M params) | Planned |
| Evaluation (PPL/BPB, BLEU/chrF/ROUGE-L, attention analysis) | Planned |
| Reasoning finetuning (synthetic Hindi comparisons) | Planned |

## Repository layout

```
hindi/
  data/        preprocess.py, build_corpus.py, make_stats_report.py   (raw/processed/tokens are git-ignored)
  tokenizer/   train_tokenizer.py, eval_tokenizer.py, models/hindi_bpe_{16,24,32}k.{model,vocab}
  model/ train/ eval/ configs/                                          (upcoming)
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

## Reproduce (Phase 1)

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
