"""Fig. 9: the S2-PepAnalyst head-to-head, PeptideBERT vs ESM-2, with
BLOSUM62 as a non-learned control.

Two decisions worth stating, because both are interpretive:

(1) Panel (b) plots distance DIVIDED BY each representation's own
    in-distribution reference distance, not raw distance. The three
    embedding spaces have incompatible scales (reference distances 2.66,
    11.69 and 1.99), so a shared raw axis would squash BLOSUM62 into a
    sliver and imply comparisons the manuscript explicitly disclaims.
    Normalising by the reference puts "1.0 = as far as a held-out
    in-distribution sequence" on one axis for all three.

(2) The annotation reports the margin by which each method holds its
    12/12 pairwise score -- min(non-active) - max(active), in
    active-group sd. All three score 12/12, but BLOSUM62 holds it by
    0.01 sd and ESM-2 by 1.66. A headline tie that fragile should be
    visible in the figure, not buried.

Usage: python3 make_fig9_esm2.py   (edit RESULTS / ESM2_JSON below)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RESULTS = "results"
FIGDIR = "figures"
ESM2_JSON = f"{RESULTS}/esm2_head_to_head__esm2_t6_8M_UR50D.json"
BLOSUM_JSON = f"{RESULTS}/blosum_head_to_head.json"

PB_RED, ESM_BLUE, BL_PURPLE, GREY = "#c0392b", "#2980b9", "#8e44ad", "#7f8c8d"
ORD3 = {"ACTIVE": 0, "PARTIAL": 1, "REDUCED": 1, "INACTIVE": 2}
NON = ("INACTIVE", "REDUCED", "PARTIAL")

plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

emb = json.load(open(f"{RESULTS}/embeddings_analysis.json"))
pbs = json.load(open(f"{RESULTS}/mature_screening_heldout.json"))
esm = json.load(open(ESM2_JSON))
blo = json.load(open(BLOSUM_JSON))

# ---------------------------------------------------------------- panel a
rows = [
    ("PeptideBERT",       emb["knn_table"]["pretrained"],              emb["knn_table_length_matched"]["pretrained"],              PB_RED),
    ("ESM-2\n(8M)",       esm["main_full"],                            esm["main_length_matched"],                                 ESM_BLUE),
    ("BLOSUM62",          blo["main_full"],                            blo["main_length_matched"],                                 BL_PURPLE),
    ("length +\ncomp.",   emb["knn_table"]["length_plus_composition"], emb["knn_table_length_matched"]["length_plus_composition"], GREY),
    ("untrained\nbackbone", emb["knn_table"]["random_init"],           emb["knn_table_length_matched"]["random_init"],              GREY),
    ("dipeptide\ncomp.",  emb["knn_table"]["dipeptide_composition"],   emb["knn_table_length_matched"]["dipeptide_composition"],    GREY),
    ("length\nonly",      emb["knn_table"]["length_only"],             emb["knn_table_length_matched"]["length_only"],              GREY),
]
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
ax = axes[0]
x = np.arange(len(rows)); w = 0.36
full = [r[1]["mean"] for r in rows];  full_e = [r[1]["std"] for r in rows]
lm   = [r[2]["mean"] for r in rows];  lm_e   = [r[2]["std"] for r in rows]
cols = [r[3] for r in rows]
ax.bar(x - w/2, full, w, yerr=full_e, capsize=3, color=cols, alpha=0.55,
       error_kw=dict(lw=1, ecolor="#444"), label="full corpus (n=300)")
ax.bar(x + w/2, lm, w, yerr=lm_e, capsize=3, color=cols,
       error_kw=dict(lw=1, ecolor="#444"), label="length-matched (n=148)")
ax.axhline(0.2, color="black", linestyle=":", linewidth=1)
ax.text(len(rows) - 1.35, 0.215, "chance (1/5)", fontsize=7.5, ha="right", color="#333")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel("5-fold kNN accuracy (family, $k$=5)")
ax.set_ylim(0, 1.12); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("a. Main corpus: PeptideBERT leads ESM-2 by 3.0 points\n"
             "(full) and 1.3 points (length-matched)", fontsize=10.5)

# ---------------------------------------------------------------- panel b
def table(cands, ref):
    return {c["name"]: c["mean_dist"] / ref for c in cands}, \
           {c["name"]: c["expected"] for c in cands}

pb_lit = [c for c in pbs["candidates"] if c["group"] == "literature"]
series = [
    ("PeptideBERT", pb_lit,                        pbs["reference_mean_dist"],                 PB_RED,    "o"),
    ("ESM-2 (8M)",  esm["mature_screening"]["candidates"], esm["mature_screening"]["reference_mean_dist"], ESM_BLUE,  "s"),
    ("BLOSUM62",    blo["mature_screening"]["candidates"], blo["mature_screening"]["reference_mean_dist"], BL_PURPLE, "^"),
]
norm = {}; exp = {}
for tag, cands, ref, _, _ in series:
    norm[tag], exp = table(cands, ref)

names = sorted(norm["ESM-2 (8M)"], key=lambda n: norm["ESM-2 (8M)"][n])
y = np.arange(len(names))[::-1]

ax = axes[1]
for i, n in enumerate(names):
    vals = [norm[t][n] for t, *_ in series]
    ax.plot([min(vals), max(vals)], [y[i], y[i]], color="#bbb", lw=1, zorder=1)
for tag, _, _, col, mk in series:
    ax.scatter([norm[tag][n] for n in names], y, s=78, marker=mk, color=col,
               label=tag, zorder=3, edgecolors="white", linewidths=0.6)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
ax.text(0.985, 5.5, "in-distribution reference", fontsize=7.5, color="#333",
        rotation=90, va="center", ha="right")
ax.set_xlim(0.25, 3.55)
ax.set_yticks(y)
ax.set_yticklabels([f"{n.replace('_',' ')}  ({exp[n]})" for n in names], fontsize=8)
ax.set_xlabel("mean distance to 5 nearest training sequences\n"
              "($\\times$ that representation's own in-distribution reference)")

ann = []
for tag, cands, _, _, _ in series:
    a = np.array([c["mean_dist"] for c in cands if c["expected"] == "ACTIVE"])
    b = np.array([c["mean_dist"] for c in cands if c["expected"] in NON])
    rho = spearmanr([c["mean_dist"] for c in cands], [ORD3[c["expected"]] for c in cands])[0]
    ann.append(f"{tag}: $\\rho$={rho:.3f}, margin={(b.min()-a.max())/a.std():.2f} sd")
ax.set_title("b. Literature screening (n=7): all three score 12/12,\n"
             "but not with the same confidence", fontsize=10.5, pad=12)
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax.text(0.985, 0.60, "\n".join(ann), transform=ax.transAxes, fontsize=7.8,
        va="top", ha="right", bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="#ccc", lw=0.8))

fig.suptitle("Fig. 9 | S$^2$-PepAnalyst head-to-head: PeptideBERT vs ESM-2 (esm2_t6_8M_UR50D), "
             "BLOSUM62 as non-learned control", y=1.03, fontsize=11)
fig.text(0.5, -0.07,
         "PeptideBERT's advantage is clear at full-corpus scale (panel a, ahead in 4 of 5 folds, behind in none) and unresolved\n"
         "length-matched (ahead in 3, level in 1, behind in 1). On the literature test all three order every active/inactive pair\n"
         "correctly, but the margin holding that result spans two orders of magnitude: 1.66 sd for ESM-2, 0.32 for PeptideBERT,\n"
         "0.01 for BLOSUM62 -- whose apparent parity rests on 0.003 distance units.",
         ha="center", fontsize=8.2, style="italic")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig9_esm2_head_to_head.png", bbox_inches="tight")
print(f"wrote {FIGDIR}/fig9_esm2_head_to_head.png")
