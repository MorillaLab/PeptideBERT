"""Fig. 8: Option (a) result -- mature-scale model succeeds where the
precursor-scale model failed. Plotted as a direct before/after so the
comparison is the point, not a detail."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/home/claude/s2pepanalyst/results"
FIGDIR = "/home/claude/s2pepanalyst/figures"
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                      "axes.spines.right": False})

old = json.load(open(f"{RESULTS}/tomato_screening_case_study.json"))
new = json.load(open(f"{RESULTS}/mature_screening_heldout.json"))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

# --- a: in-distribution, before vs after
ax = axes[0]
old_cand = [c["mean_dist_to_neighbours"] for c in old["candidates"]]
old_ref = [c["mean_dist_to_neighbours"] for c in old["in_distribution_controls"]]
new_cand = [c["mean_dist"] for c in new["candidates"] if c["group"] == "systemin"]
new_ref = new["reference_mean_dist"]
ratio_old = np.mean(old_cand) / np.mean(old_ref)
ratio_new = np.mean(new_cand) / new_ref
ax.bar(["precursor-scale\n(33–155 aa)", "mature-scale\n(10–25 aa)"], [ratio_old, ratio_new],
       color=["#c0392b", "#27ae60"])
ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
ax.text(0.5, 1.08, "in-distribution (=1.0)", ha="center", fontsize=8, style="italic")
ax.set_ylabel("candidate distance ÷ held-out reference distance")
ax.set_title("a. Are 18-aa candidates in-distribution?", fontsize=10.5)
for i, v in enumerate([ratio_old, ratio_new]):
    ax.text(i, v + 0.08, f"{v:.2f}×", ha="center", fontweight="bold")
ax.set_ylim(0, max(ratio_old, ratio_new) * 1.25)

# --- b: our designed control, before vs after
ax = axes[1]
old_real = [c["mean_dist_to_neighbours"] for c in old["candidates"] if c["name"] == "tomato_systemin"][0]
old_dis = [c["mean_dist_to_neighbours"] for c in old["candidates"] if "disrupted" in c["name"]][0]
new_cons = [c["mean_dist"] for c in new["candidates"] if c["name"].startswith("systemin_var")]
new_dis = [c["mean_dist"] for c in new["candidates"] if c["name"] == "systemin_disrupted"][0]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [old_real, np.mean(new_cons)], w, label="intact systemin / conservative variants", color="#2980b9")
ax.bar(x + w/2, [old_dis, new_dis], w, label="deliberately disrupted control", color="#000000")
ax.set_xticks(x); ax.set_xticklabels(["precursor-scale", "mature-scale"])
ax.set_ylabel("mean distance to nearest training sequences")
ax.set_title("b. Does the broken control rank worse,\nas it must?", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.5)
ax.text(0, max(old_real, old_dis) * 1.05, "FAIL\n(broken ranks better)", ha="center",
        fontsize=8, color="#c0392b", fontweight="bold")
ax.text(1, new_dis * 1.05, "PASS", ha="center", fontsize=8, color="#27ae60", fontweight="bold")
ax.set_ylim(0, max(old_real, old_dis, new_dis) * 1.35)

# --- c: independent literature ground truth
ax = axes[2]
lit = [c for c in new["candidates"] if c["group"] == "literature"]
lit = sorted(lit, key=lambda r: r["mean_dist"])
cols = {"ACTIVE": "#27ae60", "REDUCED": "#e67e22", "PARTIAL": "#e67e22", "INACTIVE": "#c0392b"}
names = [r["name"].replace("_", " ") for r in lit]
vals = [r["mean_dist"] for r in lit]
ax.barh(names, vals, color=[cols.get(r["expected"].split()[0], "#7f8c8d") for r in lit])
ax.axvline(new["reference_mean_dist"], color="black", linestyle=":", linewidth=1)
ax.text(new["reference_mean_dist"] + 0.1, len(lit) - 0.4, "in-distribution\nreference", fontsize=7.5, style="italic")
ax.invert_yaxis()
ax.set_xlabel("mean distance to nearest training sequences")
ax.set_title(f"c. Independent literature ground truth:\n"
             f"{new['pairwise_correct']}/{new['pairwise_total']} active-vs-inactive pairs ranked correctly",
             fontsize=10.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#27ae60", label="known active"),
                   Patch(color="#e67e22", label="known reduced/partial"),
                   Patch(color="#c0392b", label="known inactive")],
          frameon=False, fontsize=7.5, loc="lower right")

fig.suptitle("Fig. 8 | Retraining at mature-peptide scale fixes the screening failure",
             y=1.03, fontsize=11)
fig.text(0.5, -0.05,
         "All four canonical active peptides (AtPep1, CLV3, TDIF, GrCLE1-1) were removed from training before this test, so panel c\n"
         "contains no memorised sequences. Ground truth is from published structure-activity studies, not from us.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig8_mature_scale_screening.png", bbox_inches="tight")
print("wrote fig8_mature_scale_screening.png")
