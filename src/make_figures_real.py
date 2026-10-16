"""
Figure set for the real-corpus results.

Reads the result JSONs and regenerates every figure, so re-running any
analysis and re-running this script keeps the figures consistent with the
artefacts. Nothing is hardcoded from a previous run.

    python3 src/make_figures_real.py

Four figures, chosen for what the real results actually show:

  fig1  pretraining: loss and masked-residue accuracy over 70 epochs, plus
        per-family masked accuracy. The headline here is that validation is
        still rising at epoch 70 and the negatives sit lowest.
  fig2  representation comparison: the central result. Every feature space
        on one axis, with the untrained backbone marked as the reference
        the 27-point margin is measured against, and the paired fold
        record against ESM-2 beside it.
  fig3  fine-tuning: all four regimes, showing that pretraining is worth
        15 points frozen and 2 points end to end.
  fig4  what the model attends to: t-SNE of the embedding space and the
        attention-to-motif heatmaps.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")

PB, ESM, BL, GREY, DARK = "#c0392b", "#2980b9", "#8e44ad", "#95a5a6", "#2c3e50"
FAMCOL = {"RALF": "#c0392b", "CLE": "#2980b9", "PSK": "#27ae60",
          "PEP": "#e67e22", "RGF": "#8e44ad", "NONSSP": "#7f8c8d",
          "DECOY": "#7f8c8d"}

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- fig 1
def fig1():
    curve, ppl = load("pretrain_curve.json"), load("pretrain_ppl_eval.json")
    if not curve:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ep = [c["epoch"] for c in curve]

    ax[0].plot(ep, [c["train_loss"] for c in curve], color=PB, lw=1.6, label="train")
    ax[0].plot(ep, [c["val_loss"] for c in curve], color=DARK, lw=1.6, ls="--", label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("masked-LM cross-entropy")
    ax[0].legend(frameon=False); ax[0].set_title("a. Pretraining loss")

    tr = np.array([c["train_acc"] for c in curve]); va = np.array([c["val_acc"] for c in curve])
    ax[1].plot(ep, tr, color=PB, lw=1.6, label="train")
    ax[1].plot(ep, va, color=DARK, lw=1.6, ls="--", label="validation")
    # the trend over the last third, to make "still rising" visible rather than asserted
    z = np.polyfit(ep[-25:], va[-25:], 1)
    ax[1].plot(ep[-25:], np.polyval(z, ep[-25:]), color="#e67e22", lw=1.2, ls=":",
               label=f"last 25 epochs: {z[0]*100:+.3f} pts/epoch")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("masked-residue accuracy")
    ax[1].legend(frameon=False, fontsize=7.5)
    ax[1].set_title("b. Still improving at epoch 70")

    if ppl:
        fams = sorted({r["family"] for r in ppl["pretrained"]})
        neg = [f for f in fams if f in ("NONSSP", "DECOY")]
        fams = [f for f in fams if f not in neg] + neg
        def by(k):
            d = {}
            for r in ppl[k]:
                d.setdefault(r["family"], []).append(r["accuracy"])
            return [np.mean(d[f]) for f in fams]
        x = np.arange(len(fams)); w = 0.38
        ax[2].bar(x - w/2, by("pretrained"), w, color=[FAMCOL.get(f, GREY) for f in fams],
                  label="pretrained")
        ax[2].bar(x + w/2, by("random_init"), w, color=[FAMCOL.get(f, GREY) for f in fams],
                  alpha=0.35, label="untrained")
        ax[2].axhline(1/21, color="black", ls=":", lw=1)
        ax[2].text(len(fams)-0.5, 1/21 + 0.004, "chance", fontsize=7, ha="right")
        ax[2].set_xticks(x); ax[2].set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
        ax[2].set_ylabel("masked-residue accuracy")
        ax[2].legend(frameon=False, fontsize=8)
        ax[2].set_title("c. Elevated on every peptide family,\nlowest on real non-SSP proteins")
    save(fig, "fig1_pretraining.png")


# ---------------------------------------------------------------- fig 2
def fig2():
    emb, esm, blo = load("embeddings_analysis.json"), None, load("blosum_head_to_head.json")
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith("esm2_head_to_head__"):
            esm = load(f)
    if not emb:
        return
    kt = emb["knn_table"]
    rows = []
    if esm:
        rows.append(("ESM-2 (8M)\nzero-shot", esm["main_full"]["mean"], esm["main_full"]["std"], ESM))
    rows.append(("PeptideBERT\n169,567 params", kt["pretrained"]["mean"], kt["pretrained"]["std"], PB))
    rows.append(("dipeptide\ncomposition", kt["dipeptide_composition"]["mean"], kt["dipeptide_composition"]["std"], GREY))
    rows.append(("length +\ncomposition", kt["length_plus_composition"]["mean"], kt["length_plus_composition"]["std"], GREY))
    if blo:
        rows.append(("BLOSUM62", blo["main_full"]["mean"], blo["main_full"]["std"], BL))
    rows.append(("untrained\nbackbone", kt["random_init"]["mean"], kt["random_init"]["std"], "#bdc3c7"))
    rows.append(("length\nonly", kt["length_only"]["mean"], kt["length_only"]["std"], "#bdc3c7"))

    n_cls = len(emb["family_names"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2),
                           gridspec_kw={"width_ratios": [1.65, 1]})
    x = np.arange(len(rows))
    ax[0].bar(x, [r[1] for r in rows], 0.62, yerr=[r[2] for r in rows], capsize=3,
              color=[r[3] for r in rows], error_kw=dict(lw=1, ecolor="#555"))
    for i, r in enumerate(rows):
        ax[0].text(i, r[1] + r[2] + 0.018, f"{r[1]:.3f}", ha="center", fontsize=7.5)
    ax[0].axhline(1/n_cls, color="black", ls=":", lw=1)
    ax[0].text(len(rows)-0.4, 1/n_cls + 0.012, f"chance (1/{n_cls})", fontsize=7.5, ha="right")
    # mark the margin the paper's central claim rests on
    pb_v = kt["pretrained"]["mean"]; ri_v = kt["random_init"]["mean"]
    i_pb = [i for i, r in enumerate(rows) if r[3] == PB][0]
    i_ri = [i for i, r in enumerate(rows) if "untrained" in r[0]][0]
    ax[0].annotate("", xy=(i_ri, ri_v), xytext=(i_ri, pb_v),
                   arrowprops=dict(arrowstyle="<->", color=PB, lw=1.3))
    ax[0].text(i_ri + 0.42, (pb_v + ri_v)/2, f"{(pb_v-ri_v)*100:+.1f} pts\nfrom pretraining",
               ha="left", va="center", fontsize=7.5, color=PB)
    ax[0].set_xlim(-0.7, len(rows) - 0.15)
    ax[0].set_xticks(x); ax[0].set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax[0].set_ylabel(f"{n_cls}-class kNN accuracy (cluster-aware 5-fold)")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("a. Pretraining beats every non-learned representation,\n"
                    "and ties a generalist model 44x larger")

    if esm and kt["pretrained"].get("folds"):
        p = np.array(kt["pretrained"]["folds"]); z = np.array(esm["main_full"]["folds"])
        f = np.arange(1, len(p)+1)
        ax[1].plot(f, p, "o-", color=PB, lw=1.6, ms=7, label="PeptideBERT")
        ax[1].plot(f, z, "s-", color=ESM, lw=1.6, ms=7, label="ESM-2 (8M)")
        for i in range(len(p)):
            ax[1].plot([f[i], f[i]], [p[i], z[i]], color="#ccc", lw=1, zorder=0)
        w = int((p > z).sum()); l = int((p < z).sum())
        ax[1].set_xticks(f); ax[1].set_xlabel("cross-validation fold (identical splits)")
        ax[1].set_ylabel("accuracy")
        ax[1].legend(frameon=False, fontsize=8, loc="lower right")
        ax[1].set_title(f"b. Paired fold by fold: PeptideBERT wins {w}, loses {l}\n"
                        f"mean difference {np.mean(p-z)*100:+.1f} points")
    save(fig, "fig2_representation_comparison.png")


# ---------------------------------------------------------------- fig 3
def fig3():
    ft = load("finetune_results.json")
    if not ft:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    style = {"pretrained_finetune": (PB, "-", "pretrained, full fine-tune"),
             "pretrained_linear_probe": (PB, "--", "pretrained, linear probe"),
             "random_init_finetune": ("#7f8c8d", "-", "untrained, full fine-tune"),
             "random_init_linear_probe": ("#7f8c8d", "--", "untrained, linear probe")}
    for k, (c, ls, lab) in style.items():
        if k not in ft:
            continue
        v = [e["val_acc"] for e in ft[k]["curve"]]
        ax[0].plot(range(1, len(v)+1), v, color=c, ls=ls, lw=1.8, label=lab)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation accuracy")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_title("a. Downstream classification, four regimes")

    pairs = [("linear probe", "pretrained_linear_probe", "random_init_linear_probe"),
             ("full fine-tune", "pretrained_finetune", "random_init_finetune")]
    x = np.arange(len(pairs)); w = 0.36
    pre = [ft[p]["final_val_acc"] for _, p, _ in pairs]
    rnd = [ft[r]["final_val_acc"] for _, _, r in pairs]
    ax[1].bar(x - w/2, pre, w, color=PB, label="pretrained")
    ax[1].bar(x + w/2, rnd, w, color="#bdc3c7", label="untrained")
    for i, (a_, b_) in enumerate(zip(pre, rnd)):
        ax[1].text(i, max(a_, b_) + 0.02, f"{(a_-b_)*100:+.1f} pts",
                   ha="center", fontsize=9, color=PB if a_ > b_ else DARK)
    ax[1].set_xticks(x); ax[1].set_xticklabels([p[0] for p in pairs])
    ax[1].set_ylabel("final validation accuracy"); ax[1].set_ylim(0, 1.0)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("b. Pretraining matters most\nwhen the backbone is frozen")
    save(fig, "fig3_finetuning.png")


# ---------------------------------------------------------------- fig 4
def fig4():
    emb, ex = load("embeddings_analysis.json"), load("explain_results.json")
    if not emb:
        return
    have_ex = ex is not None
    fig, ax = plt.subplots(1, 3 if have_ex else 2, figsize=(13 if have_ex else 9, 4.0))
    fam = np.array(emb["family"])
    for j, key in enumerate(["random_init", "pretrained"]):
        P = np.array(emb["projections"][key])
        for f in emb["family_names"]:
            m = fam == f
            ax[j].scatter(P[m, 0], P[m, 1], s=7, alpha=0.65,
                          color=FAMCOL.get(f, GREY), label=f, linewidths=0)
        ax[j].set_xticks([]); ax[j].set_yticks([])
        acc = emb["knn_table"][key]["mean"]
        ax[j].set_title(("a. Untrained backbone" if j == 0 else "b. After pretraining")
                        + f"\nkNN {acc:.3f}")
    ax[1].legend(frameon=False, fontsize=7, markerscale=1.6, loc="best", ncol=2)

    if have_ex:
        A = np.array(ex["attention_to_motifs_pretrained"])
        R = np.array(ex["attention_to_motifs_random"])
        im = ax[2].imshow(A, cmap="Reds", vmin=0, vmax=max(A.max(), R.max()))
        for i in range(A.shape[0]):
            for j2 in range(A.shape[1]):
                ax[2].text(j2, i, f"{A[i,j2]:.2f}", ha="center", va="center",
                           fontsize=7, color="white" if A[i, j2] > 0.4 else "black")
        ax[2].set_xlabel("head"); ax[2].set_ylabel("layer")
        ax[2].set_xticks(range(A.shape[1])); ax[2].set_yticks(range(A.shape[0]))
        ax[2].set_title(f"c. Attention to annotated motifs\n"
                        f"untrained: {R.mean():.3f} $\\pm$ {R.std():.3f} (flat)")
        fig.colorbar(im, ax=ax[2], fraction=0.046)
    save(fig, "fig4_embedding_geometry.png")


if __name__ == "__main__":
    print("regenerating figures from results/ ...")
    fig1(); fig2(); fig3(); fig4()
    print("done")
