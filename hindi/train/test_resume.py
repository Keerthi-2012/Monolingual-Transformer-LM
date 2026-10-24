"""Checkpoint-resume test: an interrupted-then-resumed run must match an uninterrupted run exactly."""
import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import train

root = Path(__file__).parents[1]
cfg = json.load(open(root / "configs" / "hindi_25m.json", encoding="utf-8"))
cfg["model"].update(n_layer=2, n_head=2, d_model=64, d_ff=128, dropout=0.1)  # tiny + dropout on
cfg["train"].update(token_dir=str(root / "data" / "tokens"), micro_batch=2, grad_accum=2, max_steps=12,
                    warmup_steps=3, eval_every=6, eval_batches=2, ckpt_every=100, log_every=100)
tmp = Path(__file__).parent / "_resume_test"
shutil.rmtree(tmp, ignore_errors=True)

a = copy.deepcopy(cfg); a["train"]["out_dir"] = str(tmp / "straight")
ha = train.run(a)

b = copy.deepcopy(cfg); b["train"]["out_dir"] = str(tmp / "resumed")
train.run(b, stop_at=6)          # "crash" after step 6 (checkpoint written)
hb = train.run(b)                # fresh process state: picks up ckpt_latest.pt

la, lb = [r["train_loss"] for r in ha], [r["train_loss"] for r in hb]
print("straight:", [round(x, 5) for x in la[-4:]])
print("resumed :", [round(x, 5) for x in lb[-4:]])
assert len(la) == len(lb) == 12 and max(abs(x - y) for x, y in zip(la, lb)) < 1e-6, "resume diverged"
print("resume test passed")
shutil.rmtree(tmp, ignore_errors=True)
