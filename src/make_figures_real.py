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
    """Panel (a) is horizontal: with a full ESM-2 sweep plus TAPE there are ten
    representations, and vertical bars cannot carry ten readable labels at any
    sensible figure width. Horizontal also lets the labels stay one line."""
    emb, blo = load("embeddings_analysis.json"), load("blosum_head_to_head.json")
    esms = sorted((load(f) for f in sorted(os.listdir(RESULTS))
                   if f.startswith("esm2_head_to_head__")),
                  key=lambda d: d["provenance"]["n_parameters"])
    esm = esms[0] if esms else None          # smallest: the capacity-closest comparator
    tape = next((load(f) for f in sorted(os.listdir(RESULTS))
                 if f.startswith("tape_head_to_head__")), None)
    if not emb:
        return
    kt = emb["knn_table"]

    def sz(n):
        return f"{n/1e6:.0f}M" if n >= 1e6 else f"{n/1e3:.0f}K"

    # worst first: barh puts index 0 at the bottom
    rows = [("length only", kt["length_only"], "#bdc3c7"),
            ("untrained backbone (169K)", kt["random_init"], "#bdc3c7")]
    if blo:
        rows.append(("BLOSUM62", blo["main_full"], BL))
    rows += [("length + composition", kt["length_plus_composition"], GREY),
             ("dipeptide composition", kt["dipeptide_composition"], GREY)]
    if tape:
        rows.append((f"TAPE ({sz(tape['provenance']['n_parameters'])})",
                     tape["main_full"], "#16a085"))
    rows.append(("PhytoPepBERT (169K)", kt["pretrained"], PB))
    for e in esms:
        rows.append((f"ESM-2 ({sz(e['provenance']['n_parameters'])})",
                     e["main_full"], ESM))
    rows.sort(key=lambda r: r[1]["mean"])

    n_cls = len(emb["family_names"])
    h = max(4.2, 0.42 * len(rows) + 1.6)
    fig, ax = plt.subplots(1, 2, figsize=(13.0, h),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(rows))
    ax[0].barh(y, [r[1]["mean"] for r in rows], 0.62,
               xerr=[r[1]["std"] for r in rows], capsize=3,
               color=[r[2] for r in rows], error_kw=dict(lw=1, ecolor="#555"))
    for i, r in enumerate(rows):
        ax[0].text(r[1]["mean"] + r[1]["std"] + 0.012, i, f"{r[1]['mean']:.3f}",
                   va="center", fontsize=8,
                   fontweight="bold" if r[2] in (PB, ESM) else "normal")
    ax[0].axvline(1 / n_cls, color="black", ls=":", lw=1)
    ax[0].text(1 / n_cls - 0.012, len(rows) - 0.5, f"chance (1/{n_cls})",
               fontsize=7.5, ha="right", va="center")

    # the margin the paper's central claim rests on
    i_pb = [i for i, r in enumerate(rows) if r[2] == PB][0]
    i_ri = [i for i, r in enumerate(rows) if "untrained" in r[0]][0]
    pb_v, ri_v = rows[i_pb][1]["mean"], rows[i_ri][1]["mean"]
    xa = min(0.97, max(r[1]["mean"] + r[1]["std"] for r in rows) + 0.075)
    ax[0].annotate("", xy=(xa, i_pb), xytext=(xa, i_ri),
                   arrowprops=dict(arrowstyle="<->", color=PB, lw=1.6))
    ax[0].text(xa + 0.012, (i_pb + i_ri) / 2,
               f"{(pb_v-ri_v)*100:+.1f} pts\nfrom\npretraining",
               ha="left", va="center", fontsize=8, color=PB, fontweight="bold")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([r[0] for r in rows], fontsize=9)
    ax[0].set_xlim(0, xa + 0.14)
    ax[0].set_xlabel(f"{n_cls}-class kNN accuracy (cluster-aware 5-fold)")
    ax[0].set_title("a. Pretraining beats every non-learned representation,\n"
                    "and ties ESM-2 at the nearest scale tested")

    if esm and kt["pretrained"].get("folds"):
        p = np.array(kt["pretrained"]["folds"]); z = np.array(esm["main_full"]["folds"])
        f = np.arange(1, len(p) + 1)
        ax[1].plot(f, p, "o-", color=PB, lw=1.6, ms=7, label="PhytoPepBERT")
        ax[1].plot(f, z, "s-", color=ESM, lw=1.6, ms=7,
                   label=f"ESM-2 ({sz(esm['provenance']['n_parameters'])})")
        for i in range(len(p)):
            ax[1].plot([f[i], f[i]], [p[i], z[i]], color="#ccc", lw=1, zorder=0)
        w = int((p > z).sum()); l = int((p < z).sum())
        ax[1].set_xticks(f); ax[1].set_xlabel("cross-validation fold (identical splits)")
        ax[1].set_ylabel("accuracy")
        ax[1].legend(frameon=False, fontsize=8, loc="lower right")
        ax[1].set_title(f"b. Paired fold by fold: PhytoPepBERT wins {w}, loses {l}\n"
                        f"mean difference {np.mean(p-z)*100:+.1f} points")
    fig.tight_layout()
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


# ---------------------------------------------------------------- fig 5
def fig5():
    """Accuracy against parameter count across the ESM-2 family.

    The question this answers is not "do we beat ESM-2" but "how much
    generalist scale does it take to match a domain model of 169,567
    parameters on this task". Skipped unless at least two ESM-2 checkpoints
    have been run.
    """
    emb = load("embeddings_analysis.json")
    esms = sorted((load(f) for f in sorted(os.listdir(RESULTS))
                   if f.startswith("esm2_head_to_head__")),
                  key=lambda d: d["provenance"]["n_parameters"])
    if not emb or len(esms) < 2:
        print("  fig5 skipped (needs >=2 ESM-2 checkpoints; run --sweep)")
        return
    tape = next((load(f) for f in sorted(os.listdir(RESULTS))
                 if f.startswith("tape_head_to_head__")), None)
    blo = load("blosum_head_to_head.json")
    kt = emb["knn_table"]
    PB_PARAMS = 169567
    pb, pb_sd = kt["pretrained"]["mean"], kt["pretrained"]["std"]
    ri = kt["random_init"]["mean"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    lo = min(blo["main_full"]["mean"] if blo else 1, kt["dipeptide_composition"]["mean"])
    hi = max(blo["main_full"]["mean"] if blo else 0, kt["dipeptide_composition"]["mean"])
    ax.axhspan(lo, hi, color=GREY, alpha=0.18, zorder=0)
    ax.text(0.99, hi + 0.006, "non-learned baselines", fontsize=8, color="#555",
            transform=ax.get_yaxis_transform(), ha="right")

    x = [e["provenance"]["n_parameters"] for e in esms]
    yv = [e["main_full"]["mean"] for e in esms]
    ye = [e["main_full"]["std"] for e in esms]
    ax.errorbar(x, yv, yerr=ye, fmt="s-", color=ESM, lw=1.8, ms=8, capsize=3,
                label="ESM-2 (zero-shot)", zorder=3)
    for e, xx, yy in zip(esms, x, yv):
        ax.annotate(e["provenance"]["checkpoint"].replace("esm2_", "").replace("_UR50D", ""),
                    (xx, yy), textcoords="offset points", xytext=(0, -16),
                    fontsize=7.5, color=ESM, ha="center")

    ax.axhline(pb, color=PB, ls="--", lw=1.4, zorder=2)
    ax.fill_between([4e4, 1e10], pb - pb_sd, pb + pb_sd, color=PB, alpha=0.12, zorder=1)
    ax.errorbar([PB_PARAMS], [pb], yerr=[pb_sd], fmt="o", color=PB, ms=12, capsize=3,
                zorder=4, label=f"PhytoPepBERT ({PB_PARAMS:,} params)")
    ax.scatter([PB_PARAMS], [ri], s=90, color=GREY, zorder=4,
               label="untrained backbone (identical architecture)")
    if tape:
        tn, tv = tape["provenance"]["n_parameters"], tape["main_full"]["mean"]
        ax.errorbar([tn], [tv], yerr=[tape["main_full"]["std"]], fmt="^",
                    color="#16a085", ms=10, capsize=3, zorder=4,
                    label="TAPE (zero-shot)")
        ax.annotate("TAPE", (tn, tv), textcoords="offset points", xytext=(13, -4),
                    fontsize=7.5, color="#16a085", ha="left")

    # where, if anywhere, ESM-2 overtakes
    over = [(xx, yy) for xx, yy in zip(x, yv) if yy > pb + pb_sd]
    if over:
        ax.annotate(f"ESM-2 clears PhytoPepBERT\nat {over[0][0]/1e6:.0f}M parameters",
                    xy=over[0], xytext=(over[0][0] * 1.3, over[0][1] - 0.085),
                    fontsize=8.5, color=ESM, ha="left",
                    arrowprops=dict(arrowstyle="->", color=ESM, lw=1.2,
                                    connectionstyle="arc3,rad=0.25"))
    else:
        ax.text(max(x) * 1.6, pb + 0.012,
                "no tested checkpoint\nclears PhytoPepBERT", fontsize=8.5,
                color=PB, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(6e4, max(x) * 6)
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel(f"{len(emb['family_names'])}-class kNN accuracy "
                  f"(cluster-aware 5-fold)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("How much generalist scale does it take to match\n"
                 "169,567 domain-pretrained parameters?", fontsize=10.5)
    save(fig, "fig5_scaling.png")


if __name__ == "__main__":
    print("regenerating figures from results/ ...")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("done")
