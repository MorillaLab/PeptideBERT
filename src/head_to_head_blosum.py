"""
BLOSUM62 control for the S2-PepAnalyst head-to-head.

WHAT THIS IS NOW. The head-to-head against ESM-2 is run for real, in
head_to_head_esm2.py, against the esm2_t6_8M_UR50D checkpoint. This script
no longer stands in for it. It contributes a different and still useful
thing: a NON-LEARNED control. BLOSUM62 encodes genuine evolutionary and
physicochemical amino-acid similarity but has no learned sequence context
at all, so it calibrates how much of a language model's performance needs
a pretrained transformer rather than substitution statistics. It is also
different in kind from the dipeptide-composition and length+composition
baselines, which are counting statistics with no substitution information.

(An earlier version of this file described itself as a proxy for ESM-2,
on the grounds that ESM-2 weights were unreachable from the environment
it was written in. That is no longer true and the framing has been
removed, along with a get_esm2_embedding() stub that raised
NotImplementedError. Use head_to_head_esm2.py.)

Representation vs. representation under the same held-out, cluster-aware
kNN protocol used everywhere else in this project. This isolates the
representation as the variable rather than also reproducing
GeoTop/CNN/RL, and is not a claim about S2-PepAnalyst's pipeline output.

MATURE-SCALE ARM. Experiment 2 scores literature peptides against the
mature corpus, which is GENERATED, not retrieved: of its 320 sequences,
9 are real seeds, 231 are augmented from them and 80 are synthetic
decoys. Its numbers are therefore not comparable with the precursor-scale
results above, which come from real UniProt sequences. The script detects
this and skips the arm unless you pass --include-generated-mature, and
when you do, it writes the result under a key that says so.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from Bio.Align import substitution_matrices

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BLOSUM62 = substitution_matrices.load("BLOSUM62")
_AA20 = list("ACDEFGHIKLMNPQRSTVWY")
HOLDOUT_SEQS = {"ATKVKAKQRGKEKVSSGRPGQHN", "RTVPSGPDPLHH",
                "HEVPSGPNPISN", "RVIPGGPDPLHN"}


def blosum_embedding(seq):
    """Per-residue BLOSUM62 row against the 20 standard amino acids,
    mean-pooled over the sequence -- the same pooling used for every
    PeptideBERT and ESM-2 embedding here, so the comparison differs only
    in the representation."""
    rows = [[float(BLOSUM62[c][a]) for a in _AA20] if c in _AA20 else [0.0] * 20
            for c in seq]
    return np.mean(rows, axis=0) if rows else np.zeros(20)


def knn_cv(X, y, k=5, n_splits=5, seed=0, groups=None):
    """Cluster-aware, matching embeddings_analysis.py. Scoring one
    representation with leaky folds and another without would make the
    comparison meaningless."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    import corpus as _C
    if groups is None:
        groups = _C.clusters()
    accs = []
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        c = KNeighborsClassifier(n_neighbors=k).fit((X[tr] - mu) / sd, y[tr])
        accs.append(c.score((X[te] - mu) / sd, y[te]))
    return float(np.mean(accs)), float(np.std(accs)), [float(a) for a in accs]


def mature_provenance():
    """Returns (is_generated, summary). A corpus is 'generated' if any
    record is marked AUGMENTED or SYNTHETIC."""
    p = os.path.join(RESULTS, "mature_pretrain.json")
    if not os.path.exists(p):
        return None, "mature_pretrain.json absent"
    kinds = Counter(r.get("kind", "UNKNOWN") for r in json.load(open(p))["records"])
    generated = any(k in kinds for k in ("AUGMENTED", "SYNTHETIC"))
    return generated, dict(kinds)


def experiment_2(out, label):
    heldout = json.load(open(os.path.join(RESULTS, "mature_screening_heldout.json")))
    lit = [c for c in heldout["candidates"] if c["group"] == "literature"]
    meta = json.load(open(os.path.join(RESULTS, "mature_pretrain.json")))
    recs = meta["records"]
    tr = [i for i in meta["train_idx"] if recs[i]["seq"] not in HOLDOUT_SEQS]
    train_seqs = [recs[i]["seq"] for i in tr]
    train_lab = [recs[i]["family"] for i in tr]

    train_emb = np.array([blosum_embedding(s) for s in train_seqs])
    mu, sd = train_emb.mean(0), train_emb.std(0) + 1e-8
    train_std = (train_emb - mu) / sd

    def score(seq):
        e = (blosum_embedding(seq) - mu) / sd
        dist = np.linalg.norm(train_std - e[None, :], axis=1)
        nn = np.argsort(dist)[:5]
        return dict(predicted_family=Counter(train_lab[i] for i in nn).most_common(1)[0][0],
                    mean_dist=float(dist[nn].mean()))

    ref = float(np.mean([score(recs[i]["seq"])["mean_dist"] for i in meta["val_idx"]]))
    print(f"  in-distribution reference mean_dist: {ref:.3f}")
    scored = [dict(name=c["name"], expected=c["expected"], seq=c["seq"],
                   in_training=c["seq"] in train_seqs, **score(c["seq"]))
              for c in sorted(lit, key=lambda r: r["name"])]
    for r in sorted(scored, key=lambda x: x["mean_dist"]):
        print(f"    d={r['mean_dist']:6.3f}  {r['name']:20s} {r['expected']:10s}"
              f"{' [LEAK]' if r['in_training'] else ''}")
    clean = [r for r in scored if not r["in_training"]]
    a = np.array([r["mean_dist"] for r in clean if r["expected"] == "ACTIVE"])
    b = np.array([r["mean_dist"] for r in clean
                  if r["expected"] in ("INACTIVE", "REDUCED", "PARTIAL")])
    pairs = [(x, z) for x in a for z in b]
    gap, asd = float(b.mean() - a.mean()), float(a.std(ddof=0))
    print(f"    active {a.mean():.3f}  non-active {b.mean():.3f}  "
          f"gap {gap:.3f} = {gap/asd:.3f} sd   pairwise {sum(1 for x,z in pairs if x<z)}/{len(pairs)}")
    out[label] = dict(reference_mean_dist=ref, active_mean=float(a.mean()),
                      non_active_mean=float(b.mean()), gap=gap, active_sd_ddof0=asd,
                      gap_in_active_sd=gap / asd,
                      margin_in_active_sd=float((b.min() - a.max()) / asd),
                      pairwise_correct=int(sum(1 for x, z in pairs if x < z)),
                      pairwise_total=len(pairs), candidates=scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-generated-mature", action="store_true",
                    help="run Experiment 2 even though the mature corpus is "
                         "generated rather than retrieved. Result is written "
                         "under 'mature_screening_GENERATED_CORPUS'.")
    a = ap.parse_args()
    out = {}

    print("=" * 70)
    print("EXPERIMENT 1: precursor-scale corpus, full + length-matched")
    print("=" * 70)
    import corpus as C
    d = C.load()
    records = d["records"]
    fam2id = C.fam2id()
    y = np.array([fam2id[r["family"]] for r in records])
    X = np.array([blosum_embedding(r["seq"]) for r in records])

    m, s, folds = knn_cv(X, y)
    out["main_full"] = dict(mean=m, std=s, n=len(records), folds=folds)
    print(f"  BLOSUM62, full corpus (n={len(records)}): {m:.4f} +/- {s:.4f}")

    emb = json.load(open(os.path.join(RESULTS, "embeddings_analysis.json")))
    band = emb["length_matched_indices"]
    m2, s2, folds2 = knn_cv(X[band], y[band], groups=C.clusters()[band])
    out["main_length_matched"] = dict(mean=m2, std=s2, n=len(band), folds=folds2)
    print(f"  BLOSUM62, length-matched (n={len(band)}): {m2:.4f} +/- {s2:.4f}")
    print("\n  already-computed on the same splits:")
    for k in ["pretrained", "random_init", "dipeptide_composition",
              "length_only", "length_plus_composition"]:
        print(f"    {k:25s} full={emb['knn_table'][k]['mean']:.3f}  "
              f"matched={emb['knn_table_length_matched'][k]['mean']:.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: mature-peptide literature screening")
    print("=" * 70)
    generated, summary = mature_provenance()
    if generated is None:
        print(f"  skipped: {summary}")
    elif generated and not a.include_generated_mature:
        print(f"  SKIPPED. The mature corpus is generated, not retrieved: {summary}.")
        print("  Its distances are set by how the augmentation was built, so they")
        print("  are not comparable with the real-sequence results above and must")
        print("  not be reported alongside them. Rebuild it with real sequences,")
        print("  or pass --include-generated-mature to run it as a labelled control.")
        out["mature_screening_SKIPPED"] = dict(reason="generated corpus", kinds=summary)
    else:
        if generated:
            print(f"  WARNING: generated corpus ({summary}). Writing under "
                  f"'mature_screening_GENERATED_CORPUS'.")
        experiment_2(out, "mature_screening_GENERATED_CORPUS" if generated
                     else "mature_screening")

    p = os.path.join(RESULTS, "blosum_head_to_head.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
