"""Shared helpers for evaluation: load a checkpoint, the tokenizer, and held-out token shards."""
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))
from train import TokenShards  # noqa: E402
from transformer import GPT, GPTConfig  # noqa: E402

DEFAULT_TOK = ROOT / "tokenizer" / "models" / "hindi_bpe_24k.model"


def load_model(ckpt_path, device="cpu"):
    """Load weights + the config stored inside the checkpoint. Returns (model.eval(), config dict)."""
    s = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**s["config"]["model"])).to(device)
    model.load_state_dict(s["model"])
    return model.eval(), s["config"]


def load_tokenizer(path=DEFAULT_TOK):
    return spm.SentencePieceProcessor(model_file=str(path))


def byte_lengths(sp):
    """UTF-8 byte length each token id contributes to the text (control tokens such as EOS = 0).

    Used for bits-per-byte so the denominator does not depend on how decode() treats spaces.
    """
    out = np.zeros(sp.get_piece_size(), dtype=np.int64)
    for i in range(len(out)):
        if sp.is_control(i) or sp.is_unknown(i):
            continue
        out[i] = 1 if sp.is_byte(i) else len(sp.id_to_piece(i).replace("▁", " ").encode("utf-8"))
    return out


def load_split(token_dir, split, seq_len):
    return TokenShards(str(token_dir), split, seq_len)
