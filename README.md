<h1 align="center">PhytoPepBERT</h1>

<p align="center">
  <strong>A 170,000-parameter protein language model for plant small signalling
  peptides, trained from scratch in NumPy. Domain pretraining buys ~44&times;
  parameter efficiency against the ESM-2 scaling curve &mdash; and we measure
  exactly where that runs out.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/parameters-169%2C567-c0392b" alt="parameters">
  <img src="https://img.shields.io/badge/DL%20framework-none-success" alt="no deep-learning framework">
  <img src="https://img.shields.io/badge/trains%20on-laptop%20CPU-success" alt="CPU only">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/corpus-1%2C907%20real%20UniProtKB-2980b9" alt="corpus">
  <img src="https://img.shields.io/badge/evaluation-cluster--aware%20(70%25%20id)-2980b9" alt="homology-aware">
  <img src="https://img.shields.io/badge/6--class%20k--NN-0.852-c0392b" alt="accuracy">
  <img src="https://img.shields.io/badge/parameter%20efficiency-~44x%20vs%20ESM--2-c0392b" alt="parameter efficiency">
  <img src="https://img.shields.io/badge/vs%20ESM--2%20650M-11.6%20pts%20behind-8e44ad" alt="vs ESM-2 650M">
  <img src="https://img.shields.io/badge/leakage%20control-PASS-success" alt="substring baseline">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/hero_dark.png">
    <img src="figures/hero.png" alt="PhytoPepBERT against the ESM-2 scaling curve" width="100%">
  </picture>
</p>

<p align="center">
  <em>Companion pretraining module for
  <a href="https://www.biorxiv.org/content/10.1101/2024.08.02.606319">S²-PepAnalyst</a>
  (IHSM La Mayora).</em>
</p>

---

## The result

Tools that identify plant small signalling peptides (SSPs) represent sequences with
general-purpose protein language models — ESM-2 and TAPE — and learn only the
classifier on plant data. This repository asks what a representation learned on the
SSP sequence space *itself* is worth, measured against those generalists at four
scales.

The answer is an exchange rate, not a verdict: **169,567 domain-pretrained parameters
buy roughly what 7.5M generalist parameters buy — and generalist scale keeps paying
well past that point.**

Six-class family classification from frozen embeddings, 1,907 real UniProtKB
sequences, 5-fold k-NN grouped by 70%-identity cluster, chance = 16.7%:

| representation | parameters | full corpus | vs PhytoPepBERT | paired folds |
|---|---|---|---|---|
| **ESM-2** `t33_650M` | 651M | **0.968 ± 0.012** | −11.6 | 0W–5L |
| **ESM-2** `t30_150M` | 148M | 0.921 ± 0.026 | −7.0 | 0W–5L |
| **TAPE** `bert-base` | 92M | 0.887 ± 0.027 | −3.6 | 0W–4L |
| **ESM-2** `t12_35M` | 33.5M | 0.877 ± 0.030 | −2.5 | 0W–5L |
| **ESM-2** `t6_8M` | 7.5M | 0.861 ± 0.028 | −0.9 | 2W–3L — **tie** |
| **PhytoPepBERT** (this repo) | **169,567** | 0.852 ± 0.024 | — | — |
| dipeptide composition | 0 | 0.751 ± 0.041 | +10.1 | |
| length + composition | 0 | 0.750 ± 0.049 | +10.2 | |
| BLOSUM62 | 0 | 0.676 ± 0.041 | +17.6 | |
| **untrained backbone** (identical architecture) | 169,567 | 0.580 ± 0.043 | **+27.1** | |
| length only | 0 | 0.310 ± 0.030 | +54.2 | |
| exact-substring classifier | 0 | 0.357 | | |

Three things to read out of that table:

- **Pretraining is worth 27.1 points** over an architecturally identical, identically
  initialised network that was never trained. Same parameters, same seed, same
  evaluation — the difference is attributable to pretraining and nothing else. This
  is the best-controlled number here and nothing in the scaling analysis touches it.
- **It is not reachable by cheaper means.** Composition statistics and substitution
  matrices land 10–18 points below.
- **The efficiency is bounded.** PhytoPepBERT is level with ESM-2 at 7.5M, overtaken
  between there and 33.5M, and 11.6 points behind at 651M — which is the checkpoint
  S²-PepAnalyst actually uses. Above the smallest comparator it does not win a
  single fold. **This is not a drop-in substitute for a production embedding, and we
  do not claim it is.**

Two asymmetries both cut against us, and both are measured rather than assumed.
PhytoPepBERT was pretrained on 77% of the sequences the evaluation scores, worth ~5
accuracy points (BLOSUM62 control on the same subsets: −1.5 points), so the true
crossing point is *lower* than observed. And the checkpoint beating it hardest is the
one in production use, so the gap is the operationally relevant one rather than a
worst case we constructed.

**Where the margin is large:** with the backbone frozen, pretraining is worth
**15.3 points** on downstream classification (82.9% vs 67.6%). With the whole network
trainable it is worth 1.8 (85.4% vs 83.6%). A pipeline that embeds once and reuses
the representation sits in the frozen regime.

**Who this is for.** If your constraint is compute — no GPU, no framework, no model
download — a model that trains on a laptop CPU and recovers the performance of a
generalist two orders of magnitude larger is a usable tool. If your constraint is
accuracy, the generalist curve has not flattened by 651M parameters and you should
embed with it.

## What makes the evaluation trustworthy

Four guards, each of which *lowers* the headline numbers. They are the reason the
numbers mean something.

**1. Real sequences only.** Every sequence is a retrieved UniProtKB entry, with its
accession recorded in `results/splits.json`. Nothing is generated.

**2. Homology-aware cross-validation.** Sequences are clustered at 70% global
alignment identity (Needleman–Wunsch via `parasail`) and whole clusters are assigned
to folds. Verified by outcome, not by assumption: no validation sequence retains a
>70%-identity relative in training. Ignoring clusters inflates accuracy by **3.8 to
8.6 points** depending on representation.

**3. A real negative class.** `NONSSP` is length-matched plant proteins drawn from the
same organisms *and* the same Swiss-Prot/TrEMBL review status as the positives, so
neither taxonomy nor curation status correlates with the label. Not random strings.

**4. A reported leakage control.** `src/substring_baseline.py` searches for any
substring present in ≥90% of a class and 0% of every other. If several exist, the
labels are recoverable by `grep` and no accuracy on that corpus means anything.

```
$ python3 src/substring_baseline.py
corpus: 1907 sequences, 6 classes, chance = 0.167

  CLE        none
  NONSSP     none
  PEP        none
  PSK        'DYIYTQ'  (6 aa, 90% of family, 0% elsewhere)
  RALF       none
  RGF        none

  families with an exact discriminator: 1/5
  zero-parameter substring classifier accuracy: 0.357

  PASS with note -- PSK carries an invariant motif.
```

`DYIYTQ` is the sulfated pentapeptide that *defines* PSK — real biology, and a useful
positive control: a representation that cannot separate PSK is failing at something
easy. The control exits non-zero on failure, so it can gate CI.

> **Why this control exists.** An earlier version of this project used a corpus built
> by *in silico* mutagenesis from four seed sequences, with each family's conserved
> motif held invariant. A five-line substring classifier scored **100%** on it —
> above every learned method benchmarked, including a protein language model. No
> accuracy measured on that corpus could support any claim about representation
> quality. None of its results appear here. The failure mode is easy to reproduce and
> hard to notice, which is why the control now ships and runs.

---

## Quickstart

```bash
git clone https://github.com/MorillaLab/PhytoPepBERT && cd PhytoPepBERT
pip install -r requirements.txt
python3 run_all.py                       # ~1 h on a laptop CPU
```

`run_all.py` skips pretraining if `results/pretrained_model.npz` exists, so it is safe
to re-run. The released corpus and weights let you reproduce every figure without
retraining.

### Build the corpus from scratch

```bash
pip install requests biopython parasail
python3 build_real_corpus.py --contact you@example.org \
    --include-unreviewed --max-per-organism 3 \
    --families RALF,PSK,CLE,PEP,RGF --max-len 178 \
    --out results/splits.json
```

Fetches from UniProtKB, deduplicates, clusters, splits by cluster, and **refuses to
write a corpus that fails QC** (duplicates, homology leakage across the split, a
length shortcut, exact-substring discriminators, undersized classes, or a
positive/negative imbalance in organism or review status).

### The two head-to-head arms

```bash
python3 src/head_to_head_blosum.py            # non-learned control
python3 src/head_to_head_esm2.py --sweep      # 8M, 35M, 150M, 650M
python3 src/head_to_head_tape.py              # the second PLM the tool uses
```

Both PLM scripts run embedding and analysis as **separate processes**. PyTorch links
LLVM's OpenMP while NumPy/scikit-learn link Intel's via MKL; both in one process
segfaults on macOS. Once embeddings are cached, torch is never loaded again. The
sweep caches each checkpoint separately and skips completed ones, so it is safe to
interrupt — the 650M is a ~2.5 GB download and by far the slowest arm on CPU.

---

## Repository layout

```
build_real_corpus.py         UniProt retrieval + clustering + QC gates
run_all.py                   end-to-end driver
src/
  tensor.py                  reverse-mode autograd, pure NumPy
  model.py                   3-layer pre-norm transformer, 169,567 params
  test_tensor.py             finite-difference gradient checks (max err < 3e-4)
  corpus.py                  single source of truth for the corpus
  pretrain.py                masked-language pretraining, resumable
  eval_ppl.py                exact pseudo-perplexity   [pretrained|random_init]
  finetune.py                downstream classification [backbone] [regime]
  embeddings_analysis.py     frozen-embedding k-NN + t-SNE
  explain.py                 saliency + attention-to-motif
  head_to_head_blosum.py     non-learned control
  head_to_head_esm2.py       ESM-2 comparison, two-phase
  substring_baseline.py      leakage control
  head_to_head_tape.py       TAPE comparison, two-phase
  make_figures_real.py       all five figures, from results/*.json
  make_readme_hero.py        README banner (light + dark variants)
results/                     corpus, weights, every JSON behind every figure
figures/                     generated
MANUSCRIPT.tex / .pdf        the paper
```

---

## Reproducibility notes

**Pin scikit-learn.** `StratifiedGroupKFold`'s fold-balancing changed between releases,
so fold composition and per-fold accuracies are version-dependent. The numbers above
were produced with the version pinned in `requirements.txt`. Means move by ~1–2 points
across versions; conclusions do not.

**Pretraining is not bit-reproducible across interruption patterns.** It checkpoints
every 8 epochs and re-seeds its masking RNG at each call, so exact weights depend on
where the run was interrupted. Trained weights are released so no downstream result
depends on reproducing them.

**Dependencies are split by arm.** The model, pretraining, fine-tuning, embedding
analysis and explainability need only NumPy/SciPy/scikit-learn/Matplotlib — no
deep-learning framework. Corpus construction adds `requests`, `biopython`, `parasail`.
The ESM-2 arm adds `torch` and `fair-esm` plus model-hub access.

---

## Known limitations

- **2% Swiss-Prot.** The corpus is overwhelmingly TrEMBL, so annotation quality is
  uneven and the attention-to-motif analysis rests on the ~2% of entries carrying
  curated `PEPTIDE`/`PROPEP` features. Read it as suggestive.
- **Class imbalance.** PEP has 88 sequences against 400 for the largest classes.
- **Undertrained.** Validation masked-residue accuracy was still rising at epoch 70.
- **Name-based family assignment.** CLE, PEP and RGF are retrieved by curated protein
  name because Pfam does not model them as domains. GOLVEN/RGF/CLEL entries are
  excluded from CLE explicitly and promoted to their own class; other misassignments
  may remain.
- **The scaling curve is four points.** We resolve that the crossing lies between
  7.5M and 33.5M but not where within it, and one corpus cannot show the exchange
  rate generalises to other peptide families. We also did not vary PhytoPepBERT's
  own size.
- **The length-matched band is weak** (n=228, three classes under 15 members). Reported
  for continuity, not as support.

---

## Citation

```bibtex
@article{morilla2026phytopepbert,
  title   = {What does domain-specific pretraining buy? A 170,000-parameter
             plant peptide language model against the ESM-2 scaling curve},
  author  = {Morilla, Ian},
  year    = {2026},
  note    = {Preprint}
}
```

> **On the name.** Not to be confused with
> [PeptideBERT](https://github.com/ChakradharG/PeptideBERT) (Guntuboina et al.,
> *J. Phys. Chem. Lett.* 2023), which fine-tunes ProtBERT for hemolysis, solubility
> and non-fouling prediction. Different model, different task, no relation.

Built in the spirit of [NucleicBERT](https://doi.org/10.1038/s42256-026-01295-9)
(Upadhyay, Herold, Götz & Schug, *Nat. Mach. Intell.*, 2026) — a small domain language
model evaluated honestly against the baselines it has to beat.
