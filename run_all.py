"""Runs the full S2-PepAnalyst-BERT pipeline end to end, in order.
Usage: python3 run_all.py   (run from the project root)

Pretraining checkpoints itself every EPOCHS_PER_CALL epochs (see
src/pretrain.py) -- a workaround for the sandboxed environment this was
built in having a per-command time limit, not a requirement of the model
itself. This script just loops the resumable call until it's done, so on
a normal machine it behaves like a single ~10-12 minute run.
"""
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SRC, check=True)


def main():
    run([sys.executable, "test_tensor.py"])

    while not os.path.exists(os.path.join(RESULTS, "pretrained_model.npz")):
        run([sys.executable, "pretrain.py"])

    # split from pretrain.py's own inline eval -- doing both models in one
    # call risks the sandbox's per-command time limit; harmless to split
    # unconditionally on a normal machine too.
    if not os.path.exists(os.path.join(RESULTS, "pretrain_ppl_eval.json")):
        run([sys.executable, "eval_ppl.py", "pretrained"])
        run([sys.executable, "eval_ppl.py", "random_init"])

    for backbone in ("pretrained", "random_init"):
        for regime in ("finetune", "linear_probe"):
            run([sys.executable, "finetune.py", backbone, regime])

    run([sys.executable, "embeddings_analysis.py"])
    run([sys.executable, "explain.py"])
    run([sys.executable, "make_figures.py"])
    print("\nAll done. See ../REPORT.md, ../MANUSCRIPT.md and ../figures/*.png")


if __name__ == "__main__":
    main()
