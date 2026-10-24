"""Decoder-only (GPT-style) Transformer written from basic PyTorch layers.

Allowed primitives: nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout.
NOT used: nn.Transformer*, F.scaled_dot_product_attention, any HF class.
Attention, positional embeddings, and the causal mask are implemented below.

Tensor-shape legend: B=batch, T=sequence length, d=d_model, h=n_heads, dk=d/h, V=vocab.
"""
import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 24000
    block_size: int = 512      # max context length (see positional embeddings below)
    n_layer: int = 9
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536           # inner FFN width = 4 * d_model
    dropout: float = 0.05
    tie_embeddings: bool = True
    use_pos_emb: bool = True   # False = no-positional-embedding ablation (bonus)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f)["model"])

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, from first principles.

    Q = X W_Q, K = X W_K, V = X W_V are computed with one fused Linear (d -> 3d) and then
    split; this is mathematically identical to three separate (d x d) projections.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.h, self.dk = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # output projection W_O
        self.attn_drop = nn.Dropout(cfg.dropout)             # on attention probabilities
        self.resid_drop = nn.Dropout(cfg.dropout)            # on sublayer output
        # Additive causal mask M: 0 where key position <= query position, large negative above.
        # Registered as a buffer (moves with the module, not a parameter, not saved).
        m = torch.triu(torch.full((cfg.block_size, cfg.block_size), -1e9), diagonal=1)
        self.register_buffer("mask", m, persistent=False)

    def forward(self, x, return_attn: bool = False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)                          # each (B, T, d)
        # (B, T, d) -> (B, T, h, dk) -> (B, h, T, dk): each head gets its own dk-dim subspace
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        # Scaled dot-product scores (B, h, T, T). Dividing by sqrt(dk) keeps the variance of
        # q.k near 1 (it grows ~dk for unit-variance entries); otherwise large scores push
        # softmax into saturation with vanishing gradients.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.float() + self.mask[:T, :T]                    # softmax in fp32 for stability
        att = F.softmax(scores, dim=-1).to(v.dtype)                    # rows sum to 1 over allowed keys
        out = self.attn_drop(att) @ v                                  # (B, h, T, dk)
        out = out.transpose(1, 2).contiguous().view(B, T, d)           # concat heads -> (B, T, d)
        out = self.resid_drop(self.proj(out))
        return (out, att) if return_attn else out


class MLP(nn.Module):
    """Position-wise feed-forward network: d -> d_ff -> d with GELU (d_ff > d)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + MLP(LN(x)).

    Pre-norm keeps an unnormalized residual path from input to output, so gradients flow
    directly through the skip connections; this is more stable for deep stacks than post-norm.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn, self.mlp = CausalSelfAttention(cfg), MLP(cfg)

    def forward(self, x, return_attn: bool = False):
        if return_attn:
            a, att = self.attn(self.ln1(x), return_attn=True)
            x = x + a
            return x + self.mlp(self.ln2(x)), att
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Learned absolute positional embeddings: one vector per position 0..block_size-1.
        # This hard-limits the context length to block_size (no embedding exists beyond it).
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.use_pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight   # saves vocab_size * d_model parameters
        self.apply(self._init)
        for n, p in self.named_parameters():          # GPT-2 style scaled init for residual projections
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        """Trainable parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None, return_attn: bool = False):
        """idx: (B, T) token ids. Returns logits (B, T, V) and, if targets given, mean CE loss.

        Loss: cross-entropy between logits at position t and the token at position t+1;
        `targets` is the input shifted left by one (targets[:, t] = idx[:, t+1]).
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)                                           # (B, T, d)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))    # broadcast over batch
        x = self.drop(x)
        atts = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, return_attn=True)
                atts.append(a)
            else:
                x = blk(x)
        logits = self.head(self.ln_f(x))                                # (B, T, V)
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        if return_attn:
            return logits, loss, atts
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, eos_id=None):
        """Autoregressive sampling. temperature == 0 means greedy (argmax)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
