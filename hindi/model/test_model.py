"""Sanity tests for the Transformer: shapes, parameter count, causality, loss at init."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from transformer import GPT, GPTConfig


def test_param_count_and_shapes():
    cfg = GPTConfig.from_json(Path(__file__).parents[1] / "configs" / "hindi_25m.json")
    m = GPT(cfg)
    print("params:", f"{m.n_params():,}")
    assert 24e6 < m.n_params() < 26.5e6
    logits, loss = m(torch.randint(0, cfg.vocab_size, (2, 16)), torch.randint(0, cfg.vocab_size, (2, 16)))
    assert logits.shape == (2, 16, cfg.vocab_size)
    # an untrained model should score ~ln(V)
    print("init loss:", loss.item(), "ln(V):", math.log(cfg.vocab_size))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_causality():
    """Changing token t+1 must not change logits at positions <= t; it must change later ones."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, d_model=32, d_ff=64, dropout=0.0)
    m = GPT(cfg).eval()
    a = torch.randint(0, 100, (1, 20))
    b = a.clone()
    t = 9
    b[0, t + 1] = (b[0, t + 1] + 1) % 100
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[0, : t + 1], lb[0, : t + 1], atol=1e-6), "future token leaked into the past"
    assert not torch.allclose(la[0, t + 1:], lb[0, t + 1:]), "later positions should change"
    print("causality: max diff past =", (la[0, : t + 1] - lb[0, : t + 1]).abs().max().item())


def test_attention_rows():
    cfg = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, d_model=16, d_ff=32, dropout=0.0)
    m = GPT(cfg).eval()
    _, _, atts = m(torch.randint(0, 50, (2, 10)), return_attn=True)
    for a in atts:
        assert a.shape == (2, 2, 10, 10)
        assert torch.allclose(a.sum(-1), torch.ones(2, 2, 10), atol=1e-5)
        assert (torch.triu(a, diagonal=1) == 0).all()


if __name__ == "__main__":
    test_param_count_and_shapes()
    test_causality()
    test_attention_rows()
    print("all model tests passed")
