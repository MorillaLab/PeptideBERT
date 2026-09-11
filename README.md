# PeptideBERT
single-sequence language modelling learns family-defining structure in plant signalling peptides

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/deps-NumPy%20%7C%20SciPy%20%7C%20scikit--learn%20%7C%20Matplotlib-lightgrey.svg)](requirements.txt)
[![Build](https://img.shields.io/github/actions/workflow/status/MorillaLab/PeptideBERT/ci.yml?branch=main)](https://github.com/MorillaLab/PeptideBERT/actions)
[![Issues](https://img.shields.io/github/issues/MorillaLab/PeptideBERT)](https://github.com/MorillaLab/PeptideBERT/issues)
[![Repo size](https://img.shields.io/github/repo-size/MorillaLab/PeptideBERT)](https://github.com/MorillaLab/PeptideBERT)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](#citation)


## Overview

PeptideBERT is a 169,567-parameter, pre-layer-norm transformer encoder (3 layers, 4
heads, model dim 64, FFN dim 256), pretrained with masked-language modelling directly on
plant small signalling peptide (SSP) sequences (RALF, CLE, PSK, PEP) plus a synthetic
i.i.d. decoy class, and implemented from scratch in NumPy (no PyTorch/JAX dependency).
It is evaluated against an identically initialised, untrained backbone under a
{pretrained, random-init} × {fine-tuned, linear-probe} design, with masked-residue
calibration, embedding-geometry, and attention/saliency analyses as complementary lines
of evidence. See the manuscript for full methodological detail.

## Repository layout

```
PeptideBERT/
├── data/
│   ├── raw/            # Literature seed sequences (RALF, CLE) and motif references (PSK, PEP)
│   └── processed/      # Augmented 300-sequence corpus, splits, length-matched band
├── src/                # Model, autodiff engine, training, evaluation, and figure-generation code
├── checkpoints/
│   ├── pretrained_30epoch/   # Initial pretraining checkpoint (pre-extension)
│   ├── pretrained_70epoch/   # Final pretraining checkpoint used for headline results
│   └── random_init/          # Identically initialised, never-trained control backbone
├── results/
│   ├── json/            # Raw numeric outputs underlying every reported statistic
│   └── figures/         # Rendered figures (Figs. 1-6) generated from results/json
├── notebooks/           # Optional exploratory / walkthrough notebooks
├── tests/               # Gradient-check and unit tests for the from-scratch autodiff engine
├── configs/             # Hyperparameter / experiment configuration files
├── run_all.py           # Top-level entry point: reproduces every result end to end
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Reproducibility

All stochastic procedures use a fixed random seed (0). Running:

```bash
python3 run_all.py
```

is intended to regenerate the pretraining/validation/test splits, both model
checkpoints (30- and 70-epoch), all downstream fine-tuning results, and every figure
and JSON output referenced in the manuscript.

## Data and Code Availability

Code, splits, checkpoints, and raw result files underlying every figure are hosted at
https://github.com/MorillaLab/PeptideBERT.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.
