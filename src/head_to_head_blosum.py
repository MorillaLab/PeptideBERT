"""
The S2-PepAnalyst head-to-head -- and an honest account of why it is a
head-to-head against a PROXY, not against S2-PepAnalyst's actual ESM-2
pipeline.

WHAT WAS ACTUALLY ATTEMPTED: real ESM-2 weight downloads were tried
against both hosts that serve them (dl.fbaipublicfiles.com, the
fair-esm package's default; and huggingface.co, the alternate route via
transformers). Both returned HTTP 403 (host_not_allowed) from this
sandbox's egress proxy -- confirmed directly with curl, independent of
any torch/disk issue. There is no route to real ESM-2 weights from this
environment. That is a property of the sandbox, not a property of
whether the comparison is worth doing.

WHAT THIS SCRIPT DOES INSTEAD: compares PeptideBERT against a BLOSUM62
pooled-embedding baseline, playing the specific methodological role
"a generic, evolutionarily-informed representation with no plant-SSP-
specific training" -- which is the property of ESM-2 that actually
matters for the scientific question (does domain-specific pretraining
beat a generic representation?), even though BLOSUM62 is obviously not
ESM-2 itself. BLOSUM62 is the right proxy for that specific role: it is
a real, standard, universally-used representation (loaded here from
Biopython's own substitution_matrices, not transcribed by hand), encodes
genuine evolutionary/physicochemical amino acid similarity, and --
crucially -- is DIFFERENT IN KIND from the dipeptide-composition and
length+composition baselines already in the manuscript, which are
counting statistics with no substitution information at all.

This is deliberately NOT presented as "beats S2-PepAnalyst" anywhere.
The comparison is representation vs. representation, held-out kNN
throughout (same protocol used for every other baseline in this
project), which isolates the representation as the variable rather than
also having to reproduce GeoTop/CNN/RL. See the manuscript's Methods
and Discussion for the full framing, including exactly what a real
ESM-2 comparison would require and how to run it (get_esm2_embedding()
below is written as the single function to replace).
"""
import json
import numpy as np
from Bio.Align import substitution_matrices

RESULTS = "/home/morillalab/s2pepanalyst/results"
BLOSUM62 = substitution_matrices.load("BLOSUM62")
_AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def blosum_embedding(seq):
    """Per-residue: BLOSUM62 row against the 20 standard amino acids
    (20-dim). Sequence: mean-pooled over residues, matching the pooling
    convention used for every PeptideBERT embedding in this project, so
    the comparison differs only in the representation, not in how it is
    reduced to a fixed-length vector."""
    rows = []
    for c in seq:
        if c in _AA20:
            rows.append([float(BLOSUM62[c][a]) for a in _AA20])
        else:
            rows.append([0.0] * 20)
    return np.mean(rows, axis=0) if rows else np.zeros(20)


def get_esm2_embedding(seq):
    """NOT IMPLEMENTED HERE -- this is the function to fill in for a real
    ESM-2 comparison, on a machine with access to
    huggingface.co/facebook/esm2_* or dl.fbaipublicfiles.com/fair-esm.
    Minimal reference implementation for the smallest checkpoint
    (esm2_t6_8M_UR50D), given a working `fair-esm` install:

        import torch, esm
        _model, _alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        _model.eval()
        _bc = _alphabet.get_batch_converter()
        def get_esm2_embedding(seq):
            _, _, toks = _bc([("q", seq)])
            with torch.no_grad():
                out = _model(toks, repr_layers=[6])
            reps = out["representations"][6][0, 1:len(seq)+1]
            return reps.mean(0).numpy()

    Swap this in for blosum_embedding() everywhere below and every other
    line of this experiment (kNN protocol, held-out splits, literature
    ground truth) is unchanged.
    """
    raise NotImplementedError(
        "ESM-2 weights are not reachable from this sandbox (see module "
        "docstring). Fill in with the reference snippet above on a "
        "machine with model-hub access."
    )


def knn_cv(X, y, k=5, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    accs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        c = KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr])
        accs.append(c.score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    out = {}

    print("=" * 70)
    print("EXPERIMENT 1: main corpus (precursor-scale), full + length-matched")
    print("=" * 70)
    with open(f"{RESULTS}/splits.json") as f:
        d = json.load(f)
    records = d["records"]
    fam2id = {f: i for i, f in enumerate(["RALF", "CLE", "PSK", "PEP", "DECOY"])}
    y = np.array([fam2id[r["family"]] for r in records])
    X_blosum = np.array([blosum_embedding(r["seq"]) for r in records])

    m, s = knn_cv(X_blosum, y)
    print(f"BLOSUM62 baseline, full corpus (n={len(records)}):        {m:.3f} +/- {s:.3f}")
    out["main_full"] = dict(mean=m, std=s, n=len(records))

    with open(f"{RESULTS}/embeddings_analysis.json") as f:
        emb = json.load(f)
    band = emb["length_matched_indices"]
    m2, s2 = knn_cv(X_blosum[band], y[band])
    print(f"BLOSUM62 baseline, length-matched (n={len(band)}):        {m2:.3f} +/- {s2:.3f}")
    out["main_length_matched"] = dict(mean=m2, std=s2, n=len(band))
    print("\n  (for reference, already-computed numbers on the same splits:)")
    for k in ["pretrained", "random_init", "dipeptide_composition", "length_only", "length_plus_composition"]:
        print(f"    {k:25s} full={emb['knn_table'][k]['mean']:.3f}  "
              f"matched={emb['knn_table_length_matched'][k]['mean']:.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: mature-peptide screening (leak-free literature test)")
    print("=" * 70)
    with open(f"{RESULTS}/mature_screening_heldout.json") as f:
        heldout = json.load(f)
    cands = heldout["candidates"]
    lit = [c for c in cands if c["group"] == "literature"]

    meta = json.load(open(f"{RESULTS}/mature_pretrain.json"))
    all_records = meta["records"]
    HOLDOUT_SEQS = {"ATKVKAKQRGKEKVSSGRPGQHN", "RTVPSGPDPLHH", "HEVPSGPNPISN", "RVIPGGPDPLHN"}
    train_idx = [i for i in meta["train_idx"] if all_records[i]["seq"] not in HOLDOUT_SEQS]
    train_seqs = [all_records[i]["seq"] for i in train_idx]
    train_lab = [all_records[i]["family"] for i in train_idx]

    train_emb = np.array([blosum_embedding(s) for s in train_seqs])
    mu, sd = train_emb.mean(0), train_emb.std(0) + 1e-8
    train_std = (train_emb - mu) / sd

    def score(seq):
        e = (blosum_embedding(seq) - mu) / sd
        dist = np.linalg.norm(train_std - e[None, :], axis=1)
        nn = np.argsort(dist)[:5]
        from collections import Counter
        fam = Counter(train_lab[i] for i in nn).most_common(1)[0][0]
        return dict(predicted_family=fam, mean_dist=float(dist[nn].mean()))

    val_idx = [i for i in meta["val_idx"]]
    val_d = [score(all_records[i]["seq"])["mean_dist"] for i in val_idx]
    ref = float(np.mean(val_d))
    print(f"In-distribution reference mean_dist (BLOSUM62): {ref:.2f}")

    print("\nLiterature ground truth, ranked by BLOSUM62 distance:")
    scored_lit = []
    for c in sorted(lit, key=lambda r: r["name"]):
        sc = score(c["seq"])
        leaked = c["seq"] in train_seqs
        scored_lit.append(dict(name=c["name"], expected=c["expected"], seq=c["seq"],
                               in_training=leaked, **sc))
    for r in sorted(scored_lit, key=lambda x: x["mean_dist"]):
        tag = " [LEAK]" if r["in_training"] else ""
        print(f"   d={r['mean_dist']:5.2f}  {r['name']:20s} expected={r['expected']:10s}{tag}")

    clean = [r for r in scored_lit if not r["in_training"]]
    act = [r["mean_dist"] for r in clean if r["expected"] == "ACTIVE"]
    inact = [r["mean_dist"] for r in clean if r["expected"] in ("INACTIVE", "REDUCED", "PARTIAL")]
    pairs = [(a, i) for a in act for i in inact]
    correct = sum(1 for a, i in pairs if a < i)
    print(f"\n   known-ACTIVE mean d={np.mean(act):.2f}   known-REDUCED/INACTIVE mean d={np.mean(inact):.2f}")
    print(f"   pairwise ordering correct: {correct}/{len(pairs)} ({correct/len(pairs)*100:.0f}%)"
          f"   [PeptideBERT (leak-free): {heldout['pairwise_correct']}/{heldout['pairwise_total']}]")

    out["mature_screening"] = dict(
        reference_mean_dist=ref, active_mean=float(np.mean(act)), inactive_mean=float(np.mean(inact)),
        pairwise_correct=correct, pairwise_total=len(pairs), candidates=scored_lit)

    with open(f"{RESULTS}/blosum_head_to_head.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {RESULTS}/blosum_head_to_head.json")


if __name__ == "__main__":
    main()
