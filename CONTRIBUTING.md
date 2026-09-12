# Contributing to PeptideBERT

Thanks for your interest — this is a small research codebase, and contributions of
any size (typo fixes, additional tests, replication runs, new baselines) are welcome.

## Ways to contribute

- **Bug reports** — open an issue with a minimal reproduction and your environment
  (Python version, `pip freeze` output).
- **Replication runs** — the manuscript flags the 30-vs-70-epoch pretraining
  comparison as untested across random seeds. Reports of what happens under a
  different seed are especially useful; please include the full config used.
- **New baselines / downstream tasks** — e.g. structural contact prediction, or
  testing PeptideBERT embeddings as a supplement to ESM-2 in an existing pipeline.
- **Code / docs** — see below.

## Development setup

```bash
git clone https://github.com/MorillaLab/PeptideBERT.git
cd PeptideBERT
pip install -r requirements.txt
python3 -m pytest tests/
```

## Before opening a PR

- Run the gradient-check test suite (`tests/test_gradients.py`) if you touch
  `src/autograd.py` or `src/model.py` — every differentiable primitive must pass
  finite-difference checks (max abs error < 3e-4).
- Keep the fixed random seed (0) intact unless your change is specifically about
  seed sensitivity; note any deviation clearly in your PR description.
- Update `results/json/` and `results/figures/` outputs if your change affects
  reported numbers, and say so explicitly in the PR — silent changes to headline
  results are the one thing we ask you not do.
- Small, focused PRs are easier to review than large ones; feel free to open a
  draft PR early for feedback.

## Code of conduct

Be respectful and constructive. Disagreements about methodology are welcome and
expected in a research repo — personal attacks are not.
