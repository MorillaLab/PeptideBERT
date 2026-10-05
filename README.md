<div align="center">
  <img src="assets/banner.svg" alt="PeptideBERT banner" width="720"/>
</div>

<h1 align="center">PeptideBERT</h1>
<h3 align="center">Single-sequence language modelling learns family-defining structure in plant signalling peptides</h3>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.1101/TODO"><img alt="Preprint" src="https://img.shields.io/badge/preprint-bioRxiv-b31b1b.svg"/></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-green.svg"/></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue.svg"/>
  <img alt="NumPy only" src="https://img.shields.io/badge/deps-NumPy%20%7C%20SciPy%20%7C%20sklearn%20%7C%20Matplotlib-orange.svg"/>
  <img alt="Zero GPU required" src="https://img.shields.io/badge/GPU-not%20required-lightgrey.svg"/>
</p>

<p align="center">
  Vomo-Donfack, K.L. &nbsp;·&nbsp; Aberbache, M. &nbsp;·&nbsp; Cantez, A. &nbsp;·&nbsp; Hozsu, A. &nbsp;·&nbsp; Ginot, G. &nbsp;·&nbsp; Doblas, V.G. &nbsp;·&nbsp; Morilla, I.
  <br/>
  <em>Instituto de Hortofruticultura Subtropical y Mediterránea (IHSM) — MLiMO Lab &nbsp;·&nbsp; LAGA, Université Sorbonne Paris Nord</em>
</p>

---

**169,567 parameters. Zero deep-learning dependencies. Four peptide families + one synthetic decoy. Fully reproducible from a single command.** PeptideBERT asks whether a small transformer pretrained *directly on plant signalling-peptide sequences*, with no multiple-sequence alignment and no structural supervision, learns anything a matched untrained backbone does not. It does — and this repository shows exactly where it helps, where it ties a 33-year-old substitution matrix, and where it fails, reported as found.

---

## Why this exists

Tools like S²-PepAnalyst (*Plant Biotechnol. J.*, 2026) classify plant SSPs using ESM-2 embeddings pretrained on generic protein databases — representations optimised for protein sequence space at large, not for the sub-hundred-residue precursors, non-canonical secretion signals and post-translational modification chemistry that define plant peptides as a class. This project asks whether a representation pretrained on the domain itself adds anything. It is built as a transparent feasibility study rather than a finished tool: two methodological failure modes were found and corrected over the course of the work (a length-confounded corpus; an under-converged pretraining run), and both are reported rather than hidden.

---

## Headline results

### Precursor-scale model (MAX\_LEN = 180, trained on 33–155 aa precursors)

| Metric | Pretrained | BLOSUM62 proxy | ESM-2 | Random-init | Length + comp. | Dipeptide | Length only | Chance |
|---|---|---|---|---|---|---|---|---|
| kNN family acc., full corpus (n=300) | **98.0%** | 93.0% | 95.0% | 82.0% | 90.7% | 80.0% | 36.0% | 20% |
| kNN family acc., length-matched (n=148) | **97.2%** | 93.2% | 95.9% | 83.1% | 91.8% | 81.1% | 24.4% | 20% |
| Masked-residue acc. (pseudo-PPL eval) | **18.0%** | — | - | 5.1% | — | — | — | ~5% |
| Linear-probe family acc. (12 epochs) | **93.3%** | — | - | 90.0% | — | — | — | — |
| Masked positions with saliency > mean | **14 / 14** | — | - | — | — | — | — | — |

BLOSUM62 = mean-pooled per-residue BLOSUM62 rows (Henikoff & Henikoff 1992), loaded from Biopython — the closest available proxy for ESM-2's role as a generic, non-domain-pretrained representation. Real ESM-2 weights were unreachable from the build environment; a one-function drop-in for the literal comparison is in `src/head_to_head_blosum.py`.

### Mature-peptide model (MAX\_LEN = 32, trained on 10–25 aa mature peptides)

| Metric | Mature-scale model | BLOSUM62 |
|---|---|---|
| Distance ratio (candidates ÷ reference) | **0.86×** (in-distribution) | 1.00× (by construction) |
| Active vs. inactive pairwise ordering, 7 lit. peptides (held-out) | **12 / 12** | 12 / 12 |
| Spearman ρ vs. 3-level activity ordinal | 0.84 (p=0.019) | 0.90 (p=0.006) |
| Active-group mean distance | 2.16 | 1.35 |
| Reduced/inactive-group mean distance | 6.31 | 2.08 |
| Active-vs-rest gap / active-group s.d. | **5.13** | 2.15 |

All four canonical active peptides (AtPep1, CLV3, TDIF, GrCLE1-1) were withheld from training before the screening test. Sequences in training are flagged programmatically; the leaky first-pass run is kept in `src/screen_mature_candidates.py` so the difference is auditable.

---

## Tokenisation — what `[CLS]` and `[SEP]` mean

Both tokens come from BERT (Devlin et al. 2019) and serve structural rather than predictive roles.

**`[CLS]`** (classification token) is prepended to every sequence at position 0. Because self-attention lets every position attend to every other, this token's final-layer representation aggregates whole-sequence information — making it a natural read-out for sequence-level tasks. In the family-classification head, `hidden[:, 0, :]` is fed directly to the linear classifier.

**`[SEP]`** (separator token) marks the end of the sequence, appended after the last residue and before any `[PAD]` tokens. In the original BERT it separates two sentence segments; here every input is a single peptide, so it serves only as a fixed end-of-sequence anchor.

`[MASK]` and `[PAD]` do the real work during pretraining: `[MASK]` marks positions the model must predict; `[PAD]` fills unused context length and is excluded from attention via the attention mask.

---

## Architecture

```
Input sequence  ──► [CLS] + tokenised residues + [SEP] + [PAD]…
                          │
                   Embedding layer
                   (token emb. 26×64  +  positional emb. MAX_LEN×64)
                          │
              ┌── 3 × Pre-LN Transformer encoder layer  ────────┐
              │   Multi-head self-attention (4 heads, d=16 each)│
              │   Feed-forward (64→256→64, GELU)                │
              └─────────────────────────────────────────────────┘
                          │
              ┌── MLM head (pretraining) ──► 26-way logits per position
              │
              └── [CLS] pool ──► Linear classifier (downstream)

Parameters: 169,567   Context: 180 tokens (precursor) / 32 tokens (mature)
```

The autograd engine (`src/tensor.py`, ≈160 lines) is built from scratch over NumPy. Every differentiable primitive is validated against finite-difference gradient checks (`tests/test_tensor.py`, max |error| < 3×10⁻⁴ across all ops) before use. Porting to PyTorch later is a one-import swap — the model code is written to make that explicit.

---

## Quickstart

```bash
git clone https://github.com/MorillaLab/PeptideBERT.git
cd PeptideBERT
pip install -r requirements.txt

# Reproduces both pretraining runs (30- and 70-epoch),
# all four fine-tuning/linear-probe regimes,
# the BLOSUM62 head-to-head,
# the tomato screening stress-test (both the negative result and the fix),
# and all nine figures — one fixed seed (seed=0).
python3 run_all.py
```

`requirements.txt`: `numpy  scipy  scikit-learn  matplotlib  biopython`
No PyTorch, JAX, CUDA, or model-hub access required.

---

## Repository layout

```
PeptideBERT/
├── run_all.py                    # Orchestrator: runs the full pipeline end to end
├── requirements.txt
├── MANUSCRIPT.tex / .md / .pdf   # Paper source (LaTeX), Markdown mirror, compiled PDF
├── REPORT.md                     # Lab-notebook companion (shorter, informal)
│
├── src/
│   ├── tensor.py                 # NumPy autograd engine
│   ├── model.py                  # PeptideBERT transformer (precursor + mature configs)
│   │
│   ├── peptide_data.py           # Precursor-scale corpus (33–155 aa, 4 families + decoy)
│   ├── pretrain.py               # MLM pretraining with checkpointing (EPOCHS_PER_CALL=8)
│   ├── eval_ppl.py               # Standalone pseudo-perplexity evaluator (avoids OOM)
│   ├── finetune.py               # Downstream family classification (4 regimes)
│   ├── embeddings_analysis.py    # kNN + t-SNE + length-matched control
│   ├── explain.py                # Saliency mapping + attention-to-motif analysis
│   │
│   ├── mature_peptide_data.py    # Mature-peptide corpus (10–25 aa, 9 verified seeds)
│   ├── mature_pipeline.py        # Mature-model pretraining + kNN evaluation
│   ├── screen_tomato_candidates.py   # Precursor-scale screening test (NEGATIVE result)
│   ├── screen_mature_candidates.py   # First-pass mature screening (has training leaks —
│   │                                 #   kept for auditability; use screen_mature_heldout.py)
│   ├── screen_mature_heldout.py  # Leak-free screening test (4 actives held out of training)
│   │
│   ├── head_to_head_blosum.py    # BLOSUM62 proxy comparison + ESM-2 reference stub
│   ├── make_figures.py           # Figs 1–6 (precursor-scale pipeline)
│   ├── make_fig7.py              # Fig 7 (tomato screening negative result)
│   ├── make_fig8.py              # Fig 8 (mature-scale screening: before/after)
│   └── make_fig9.py              # Fig 9 (BLOSUM62 head-to-head)
│
├── tests/
│   └── test_tensor.py            # Finite-difference gradient checks on all autograd ops
│
├── configs/
│   ├── pretrain.yaml             # Pretraining hyperparameters
│   ├── finetune.yaml             # Fine-tuning / linear-probe hyperparameters
│   └── architecture.yaml         # Model dimensions (d_model, n_heads, n_layers…)
│
├── figures/                      # All nine publication figures (PNG, 140 dpi)
├── results/                      # JSON outputs underlying every figure (reproducible)
└── assets/                       # Banner and README graphics
```

---

## Method in five moves

**1 — Corpus with a length-matched control.** Four plant SSP families (RALF, CLE, PSK, PEP) expanded from verified mature seeds by conservative point substitution *plus indel-based length variation*, so that families overlap substantially in length (33–155 aa) and a classifier given only sequence length achieves 24.4% within the evaluation band — statistically indistinguishable from the 20% chance level for five classes. Without the indels, length was a near-perfect free classification cue; the control is how we know it is not one after the fix.

**2 — Pretraining and its convergence.** BERT-style masked language modelling (15% masking, 80/10/10), Adam, 70 epochs total. An initial 30-epoch run was extended on a concrete, checkable observation (loss still decreasing, no plateau), then re-run on the same held-out splits to verify the extension changed the downstream picture.

**3 — Controlled downstream evaluation.** Four regimes crossing {pretrained, random-init backbone} × {full fine-tuning, linear probe with frozen backbone}. The linear probe is the more discriminating of the two comparisons because it cannot compensate for a weaker starting representation with additional gradient updates; after 70 epochs of pretraining, it favours the pretrained backbone (93.3% versus 90.0%), while full fine-tuning still marginally favours random-init (96.7% versus 93.3%).

**4 — Explainability.** Gradient-based saliency (L2 norm of ∂loss/∂embedding per position, computed on an independent leaf tensor so gradients are position-specific, not vocabulary-pooled). Attention-to-conserved-motif heatmap, re-indexed per augmented sequence so indel-shifted motifs are tracked correctly. After pretraining: all 14 masked positions show above-mean saliency; attention variance across heads increases by an order of magnitude with heads specialising toward and away from conserved motifs.

**5 — Translational stress-test: tomato screening.** The precursor-scale model was scored against 18-residue mature systemin variants — and failed unambiguously. Candidates landed 3.3× further from the training manifold than held-out sequences, all collapsed to the same family prediction, and the deliberately disrupted negative control ranked best of six. This is documented as a scope limit, not a screening result. Rebuilding at mature-peptide scale (MAX_LEN=32, 9 verified seeds, 10–25 aa) fixed the failure: under the same kNN protocol with four canonical actives withheld from training, 12 of 12 known-active versus known-inactive literature pairs are ranked correctly.

---

## Data

All peptide seeds are individually sourced and tagged:

| Family | Kind | Seed | Source |
|---|---|---|---|
| RALF | **REAL** | AtRALF1 precursor, 120 aa | UniProt Q9SRY3 / TAIR AT1G02900 |
| CLE | **REAL** | AtCLV3 precursor, 96 aa | UniProt Q9XF04 |
| PSK | TEMPLATE | ~80 aa scaffold around YIYTQ motif | Yang et al. 2001; Srivastava et al. 2008 |
| PEP | TEMPLATE | Scaffold around AtPep1 23-mer | Huffaker et al. 2006 PNAS |
| CLE (mature) | **REAL** | CLV3 12-mer, TDIF/CLE41, 4× *Globodera* CLE mimics | Kondo 2006; Ito 2006; US 8,569,578 |
| PEP (mature) | **REAL** | AtPep1 (23 aa), AtPep1(9–23) | Huffaker 2006; IJMS PMC11117541 |
| SYSTEMIN (mature) | **REAL** | Tomato systemin, 18 aa | UniProt P27058; Pearce et al. 1991 |
| DECOY | SYNTHETIC | i.i.d. uniform over 20 amino acids | — |

TEMPLATE scaffolds embed real, literature-reported conserved motifs in illustrative (not accession-matched) flanking sequence. The `kind` field is propagated through every record in the corpus JSON and should be consulted before citing any result as if it were derived solely from verified accession sequences.

---

## Figures

| Fig. | Content |
|---|---|
| 1 | MLM pretraining curve — loss and accuracy over 70 epochs (30-epoch extension motivated by inspection; see manuscript Discussion for why the plateau matters) |
| 2 | Held-out masked-residue accuracy vs. pseudo-perplexity, coloured by peptide family (pretrained versus random-init; DECOY correctly near-chance for both) |
| 3 | t-SNE projections of pooled embeddings — random-init backbone, pretrained backbone, dipeptide-composition baseline |
| 4 | Gradient-based saliency at masked positions across a held-out RALF sequence (all 14 masked positions above mean saliency) |
| 5 | Attention-to-conserved-motif heatmap by layer and head (pretrained versus random-init; variance increases 10× after pretraining) |
| 6 | Three-panel downstream summary: fine-tuning curves (4 regimes) · kNN vs. all baselines, full corpus · same, length-matched band — length-only collapses to 24.4% (chance=20%) in the matched band |
| 7 | **Negative result** — the precursor-scale model cannot triage 18-aa mature peptides: candidates 3.3× out-of-distribution; the deliberately disrupted control ranks best; reported as a scope limit |
| 8 | **Mature-scale rebuild** — three-panel before/after: (a) distribution ratio (b) disrupted control vs. conservative variants (c) 7 literature-labelled peptides, 12/12 pairwise ordering correct, canonical actives withheld from training |
| 9 | **S²-PepAnalyst head-to-head** — PeptideBERT vs. BLOSUM62 proxy: clear win at corpus scale (98.0% vs. 93.0%); honest tie on the 7-peptide screening test (12/12 both; PeptideBERT larger gap, BLOSUM62 higher Spearman ρ); ESM-2 reference stub in `src/head_to_head_blosum.py` |

---

## Honest accounting — what this study does and does not show

**Shows convincingly (corpus scale, n=300/148):** domain-specific pretraining produces embeddings that separate SSP families more cleanly than an untrained backbone, a BLOSUM62 evolutionary proxy, and hand-engineered composition statistics, even with sequence length controlled.

**Shows suggestively (screening scale, n=7):** the mature-scale model ranks known-active literature peptides closer to its training distribution than known-inactive ones, including against held-out sequences not used in any stage of training. The same metric ties BLOSUM62.

**Does not show:** that PeptideBERT beats ESM-2 (weight downloads blocked; one-function replacement documented for anyone with model-hub access). That any specific candidate peptide is bioactive (distance to training manifold is a plausibility signal, not a predicted activity). That the 30→70-epoch finding replicates across random seeds (single run; reported as a documented instance, not a general rule).

---

## Roadmap

- [ ] Port `tensor.py` autograd to PyTorch/JAX — the model code is written to make this a one-import change, no structural refactor needed
- [ ] Replace PSK and PEP template scaffolds with real curated sequences from PlantPepDB or an S²-PepAnalyst training-set export
- [ ] Run the literal ESM-2 comparison — swap `get_esm2_embedding()` in `src/head_to_head_blosum.py` on a machine with model-hub access; no other change required
- [ ] Structural downstream task — contact/interface-prediction using AlphaFold-modelled RALF–LRX complexes (SlLRX5–SlRALF5/10, see Montano et al. 2026); data sourced, framing under discussion
- [ ] Replicate the 30-vs-70-epoch pretraining comparison across seeds to determine how seed-dependent the convergence finding is
- [ ] Expand the mature-peptide literature test set beyond 7 labelled peptides to separate the "tied at n=7" picture from a real sample-size issue
- [x] Length-matched control — corpus rebuilt with indel augmentation; control confirmed (24.4% length-only in matched band, p≈chance)
- [x] Pretraining convergence — 30-epoch run extended to 70 epochs; linear-probe discrepancy resolved
- [x] Translational stress-test (tomato screening) — precursor-scale failure documented (Fig. 7); mature-scale fix built and validated (Fig. 8)
- [x] Generic-representation head-to-head — BLOSUM62 proxy comparison run and documented on both the main corpus and the screening test (Fig. 9); one-function ESM-2 stub in place

---

## Citation

```bibtex
@article{vomodonfack2026peptidebert,
  title   = {PeptideBERT: single-sequence language modelling learns family-defining structure in plant signalling peptides},
  author  = {Vomo-Donfack, K.L. and Aberbache, M. and Cantez, A. and Hozsu, A. and Ginot, G. and Doblas, V.G. and Morilla, I.},
  year    = {2026},
  url     = {https://github.com/MorillaLab/PeptideBERT}
}
```

If you use the S²-PepAnalyst tool or dataset that this work builds on, please also cite:

```bibtex
@article{vomodonfack2026s2pepanalyst,
  title   = {S²-PepAnalyst: a web tool for predicting plant small signalling peptides},
  author  = {Vomo-Donfack, K.L. and Abaach, M. and Luna, A.M. and Ginot, G. and Doblas, V.G. and Morilla, I.},
  journal = {Plant Biotechnology Journal},
  volume  = {24},
  number  = {5},
  pages   = {3244--3260},
  year    = {2026},
  url     = {https://www.s2-pepanalyst.uma.es}
}
```

---

## Contributing

Issues and pull requests are welcome. Before opening a PR, run `python3 tests/test_tensor.py` — all gradient checks must pass. If you are adding a new downstream task or evaluation, follow the `kind` tagging convention in `src/peptide_data.py` so data provenance stays visible throughout.

## License

MIT — see `LICENSE`.
