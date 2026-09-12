"""Generates all reproducible figures from the JSON results written by
pretrain.py / finetune.py / embeddings_analysis.py / explain.py.
Run this last, after all four have completed.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/home/morillalab/s2pepanalyst/results"
FIGDIR = "/home/morillalab/s2pepanalyst/figures"
FAMILY_COLORS = {"RALF": "#c0392b", "CLE": "#2980b9", "PSK": "#27ae60",
                  "PEP": "#8e44ad", "DECOY": "#7f8c8d"}
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                      "axes.spines.right": False})


def fig1_pretrain_curve():
    curve = json.load(open(f"{RESULTS}/pretrain_curve.json"))
    epochs = [c["epoch"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(epochs, [c["train_loss"] for c in curve], label="train", color="#2c3e50")
    axes[0].plot(epochs, [c["val_loss"] for c in curve], label="val", color="#e67e22")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MLM loss (cross-entropy)")
    axes[0].set_title("a. Pretraining loss"); axes[0].legend(frameon=False)
    axes[1].plot(epochs, [c["train_acc"] for c in curve], label="train", color="#2c3e50")
    axes[1].plot(epochs, [c["val_acc"] for c in curve], label="val", color="#e67e22")
    axes[1].axhline(1 / 20, color="gray", linestyle="--", linewidth=1, label="chance (1/20)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("masked-token accuracy")
    axes[1].set_title("b. Pretraining accuracy"); axes[1].legend(frameon=False)
    fig.suptitle("Fig. 1 | Masked-language-model pretraining on plant SSP sequences", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig1_pretrain_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig2_accuracy_vs_perplexity():
    d = json.load(open(f"{RESULTS}/pretrain_ppl_eval.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, key, title in [(axes[0], "random_init", "a. Non-pretrained model"),
                            (axes[1], "pretrained", "b. Pretrained model")]:
        rows = d[key]
        for fam in FAMILY_COLORS:
            fam_rows = [r for r in rows if r["family"] == fam]
            if not fam_rows:
                continue
            ax.scatter([r["pseudo_perplexity"] for r in fam_rows], [r["accuracy"] for r in fam_rows],
                       c=FAMILY_COLORS[fam], s=32, edgecolor="white", linewidth=0.5, label=fam, alpha=0.9)
        ax.set_xlabel(r"$PPL_{pseudo}$"); ax.set_title(title)
    axes[0].set_ylabel("held-out per-position accuracy")
    axes[1].legend(frameon=False, loc="center right", fontsize=8)
    fig.suptitle("Fig. 2 | Pretraining validation: accuracy vs pseudo-perplexity, by family", y=1.04, fontsize=11)
    fig.text(0.5, -0.04, "DECOY = synthetic i.i.d.-random sequences: genuinely unpredictable, so the pretrained\n"
                          "model is correctly low-accuracy/high-perplexity there too, not just uniformly \"confident\".",
              ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig2_accuracy_vs_perplexity.png", bbox_inches="tight")
    plt.close(fig)


def fig3_embedding_projections():
    d = json.load(open(f"{RESULTS}/embeddings_analysis.json"))
    fam = d["family"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    titles = {"random_init": "a. Random-init backbone", "pretrained": "b. Pretrained PeptideBERT",
              "dipeptide_composition": "c. Dipeptide-composition baseline"}
    for ax, key in zip(axes, ["random_init", "pretrained", "dipeptide_composition"]):
        pts = np.array(d["projections"][key])
        for f in FAMILY_COLORS:
            mask = [ff == f for ff in fam]
            ax.scatter(pts[mask, 0], pts[mask, 1], s=16, color=FAMILY_COLORS[f], label=f, alpha=0.85, edgecolor="none")
        ax.set_title(titles[key]); ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title="family")
    fig.suptitle("Fig. 3 | t-SNE projection of pooled sequence embeddings, coloured by family", y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig3_embedding_projections.png", bbox_inches="tight")
    plt.close(fig)


def fig4_saliency():
    d = json.load(open(f"{RESULTS}/explain_results.json"))["saliency"]
    sal = np.array(d["saliency_scores"])
    mp = [p + 1 for p in d["mask_positions"]]  # +1 for leading [CLS]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(range(len(sal)), sal, color="#2c3e50", linewidth=1.1)
    ax.scatter(mp, sal[mp], color="#e74c3c", zorder=5, s=45, label="[MASK] positions", edgecolor="white", linewidth=0.6)
    ax.axhline(sal.mean(), color="gray", linestyle="--", linewidth=1, label=f"mean saliency ({sal.mean():.3f})")
    ax.set_xlabel("token position (incl. [CLS]/[SEP])"); ax.set_ylabel("saliency  " r"$\|\partial\mathcal{L}/\partial e_i\|_2$")
    ax.set_title("Fig. 4 | Saliency for masked-token prediction (example RALF-family sequence)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig4_saliency.png", bbox_inches="tight")
    plt.close(fig)


def fig5_attention_to_motifs():
    d = json.load(open(f"{RESULTS}/explain_results.json"))
    pre = np.array(d["attention_to_motifs_pretrained"])
    rnd = np.array(d["attention_to_motifs_random"])
    vmax = max(pre.max(), rnd.max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, mat, title in [(axes[0], rnd, "a. Random-init backbone"), (axes[1], pre, "b. Pretrained PeptideBERT")]:
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xlabel("head"); ax.set_title(title)
        ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(mat.shape[0]))
    axes[0].set_ylabel("layer")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("avg. attention mass on conserved motif positions")
    fig.suptitle("Fig. 5 | Attention directed at each family's conserved functional motif", y=1.03, fontsize=11)
    fig.savefig(f"{FIGDIR}/fig5_attention_to_motifs.png", bbox_inches="tight")
    plt.close(fig)


def fig6_downstream():
    ft = json.load(open(f"{RESULTS}/finetune_results.json"))
    emb = json.load(open(f"{RESULTS}/embeddings_analysis.json"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    styles = {"pretrained_finetune": ("#c0392b", "-", "pretrained, fine-tuned"),
              "pretrained_linear_probe": ("#c0392b", "--", "pretrained, linear probe"),
              "random_init_finetune": ("#7f8c8d", "-", "random-init, fine-tuned"),
              "random_init_linear_probe": ("#7f8c8d", "--", "random-init, linear probe")}
    for key, (color, ls, label) in styles.items():
        curve = ft[key]["curve"]
        axes[0].plot([c["epoch"] for c in curve], [c["val_acc"] for c in curve],
                     color=color, linestyle=ls, label=label, linewidth=1.6)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("held-out family-classification accuracy")
    axes[0].set_title("a. Downstream family classification\n(4 training regimes)")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].set_ylim(0.4, 1.05)

    order = ["pretrained", "random_init", "length_plus_composition", "dipeptide_composition", "length_only"]
    labels = ["PeptideBERT\n(pretrained)", "PeptideBERT\n(random-init)", "length +\ncomposition",
              "dipeptide\ncomposition", "length\nonly"]
    colors = ["#c0392b", "#7f8c8d", "#f1c40f", "#3498db", "#95a5a6"]

    means = [emb["knn_table"][k]["mean"] for k in order]
    stds = [emb["knn_table"][k]["std"] for k in order]
    axes[1].bar(labels, means, yerr=stds, color=colors, capsize=4)
    axes[1].axhline(0.2, color="black", linestyle=":", linewidth=1, label="chance (1/5 classes)")
    axes[1].set_ylabel("5-fold kNN accuracy (family, k=5)")
    axes[1].set_title("b. Embedding-space kNN vs. baselines\n(full corpus, n=300)")
    axes[1].set_ylim(0, 1.05); axes[1].legend(frameon=False, fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    lo, hi = emb["length_matched_band"]
    n_matched = emb["knn_table_length_matched"]["pretrained"]["n"]
    means_m = [emb["knn_table_length_matched"][k]["mean"] for k in order]
    stds_m = [emb["knn_table_length_matched"][k]["std"] for k in order]
    axes[2].bar(labels, means_m, yerr=stds_m, color=colors, capsize=4)
    axes[2].axhline(0.2, color="black", linestyle=":", linewidth=1, label="chance (1/5 classes)")
    axes[2].set_ylabel("5-fold kNN accuracy (family, k=5)")
    axes[2].set_title(f"c. Same, length-matched control\n({lo}-{hi} aa only, n={n_matched})")
    axes[2].set_ylim(0, 1.05); axes[2].legend(frameon=False, fontsize=8)
    plt.setp(axes[2].get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("Fig. 6 | Downstream evaluation, trivial baselines, and the length-matched control", y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig6_downstream_and_baselines.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pretrain_curve()
    fig2_accuracy_vs_perplexity()
    fig3_embedding_projections()
    fig4_saliency()
    fig5_attention_to_motifs()
    fig6_downstream()
    print("All figures written to", FIGDIR)
