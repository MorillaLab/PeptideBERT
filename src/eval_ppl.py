"""Standalone, resumable pseudo-perplexity evaluation -- split into one
call per model so each stays comfortably under the sandbox's per-command
time limit. Usage: python3 eval_ppl.py pretrained | random_init
"""
import json
import sys
import numpy as np
from model import PeptideBERT
from pretrain import load_model, pseudo_perplexity_eval, RESULTS, _release_memory

which = sys.argv[1]
assert which in ("pretrained", "random_init")

with open(f"{RESULTS}/splits.json") as f:
    d = json.load(f)
records, val_idx = d["records"], d["val_idx"]

model = PeptideBERT(seed=0)
load_model(model, f"{RESULTS}/{'pretrained_model' if which=='pretrained' else 'randominit_model'}.npz")

print(f"Evaluating {which} on {len(val_idx)} held-out sequences (max_positions=40)...")
rows = pseudo_perplexity_eval(model, records, val_idx, max_positions=40)
_release_memory()

out_path = f"{RESULTS}/pretrain_ppl_eval.json"
try:
    with open(out_path) as f:
        existing = json.load(f)
except FileNotFoundError:
    existing = {}
existing[which] = rows
with open(out_path, "w") as f:
    json.dump(existing, f, indent=1)

mean_acc = np.mean([r["accuracy"] for r in rows])
print(f"{which}: mean per-position accuracy = {mean_acc:.3f}")
