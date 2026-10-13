"""
ESM-2 head-to-head, instrumented.

Drop-in replacement for the Colab script. Three things it adds over that
version, each one closing a gap found by auditing the manuscript against
the repository:

  1. PER-FOLD ACCURACIES. The original stored only mean and sd, which
     makes a paired fold-by-fold comparison against PeptideBERT
     impossible to recover. They are now written to the JSON.

  2. SEPARATION + RANK STATISTICS, COMPUTED IN CODE. The manuscript's
     "gap of 4.16 distance units, 5.13 active-group standard deviations"
     figures exist nowhere in the repository -- they were derived by
     hand. separation_stats() below computes them, and is applied to
     PeptideBERT and BLOSUM62 through the SAME code path so all three
     numbers are reproducible and mutually consistent. ddof=0 is not a
     stylistic choice: it is what reproduces the published 5.13 and 2.15.

  3. PROVENANCE. Checkpoint name, embedding width, layer count and
     fair-esm version go into the JSON, so the question "which ESM-2 was
     this?" is answerable from the artefact alone.

Also: embeddings are batched (the original embedded one sequence at a
time, which is tolerable at 8M and painful at 650M), cached to disk, and
run on GPU when one is available.

Set CHECKPOINTS to a list to sweep model sizes. Each checkpoint writes
its own JSON, so results never overwrite each other.

Local use: see the LOCAL RUN notes at the bottom of this file.
"""
import json
import os
import numpy as np
import torch
import esm
from collections import Counter
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
RESULTS = "/content/drive/MyDrive"      # local: "/path/to/s2pepanalyst/results"
CACHE = os.path.join(RESULTS, "esm2_cache")

# The manuscript as written reports the first of these. Add the others to
# test whether the head-to-head result is an artefact of checkpoint size.
CHECKPOINTS = [
    "esm2_t6_8M_UR50D",
    # "esm2_t12_35M_UR50D",
    # "esm2_t30_150M_UR50D",
    # "esm2_t33_650M_UR50D",
]

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOLDOUT_SEQS = {"ATKVKAKQRGKEKVSSGRPGQHN", "RTVPSGPDPLHH",
                "HEVPSGPNPISN", "RVIPGGPDPLHN"}
ORD3 = {"ACTIVE": 0, "PARTIAL": 1, "REDUCED": 1, "INACTIVE": 2}
NON_ACTIVE = ("INACTIVE", "REDUCED", "PARTIAL")


# --------------------------------------------------------------------
# EMBEDDING
# --------------------------------------------------------------------
def load_esm2(name):
    model, alphabet = getattr(esm.pretrained, name)()
    model.eval().to(DEVICE)
    meta = dict(
        checkpoint=name,
        embed_dim=int(model.embed_dim),
        num_layers=int(model.num_layers),
        n_parameters=int(sum(p.numel() for p in model.parameters())),
        fair_esm_version=getattr(esm, "__version__", "unknown"),
        torch_version=torch.__version__,
        device=DEVICE,
    )
    return model, alphabet.get_batch_converter(), meta


def embed_many(seqs, model, bc, layer, batch_size=BATCH_SIZE):
    """Mean-pool final-layer representations over residue positions only.

    Identical in result to the one-sequence-at-a-time version: the slice
    1:len(seq)+1 drops BOS and stops before EOS, so right-padding
    introduced by batching is never read.
    """
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        _, _, toks = bc([(f"s{j}", s) for j, s in enumerate(chunk)])
        with torch.no_grad():
            reps = model(toks.to(DEVICE), repr_layers=[layer])["representations"][layer]
        for j, s in enumerate(chunk):
            out.append(reps[j, 1:len(s) + 1].mean(0).float().cpu().numpy())
    return np.asarray(out)


def cached_embeddings(tag, seqs, model, bc, layer):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}.npy")
    if os.path.exists(path):
        X = np.load(path)
        if X.shape[0] == len(seqs):
            print(f"  [cache] {tag}: {X.shape}")
            return X
        print(f"  [cache] {tag}: stale ({X.shape[0]} rows, expected {len(seqs)}) -- recomputing")
    X = embed_many(seqs, model, bc, layer)
    np.save(path, X)
    print(f"  [computed] {tag}: {X.shape}")
    return X


# --------------------------------------------------------------------
# STATISTICS
# --------------------------------------------------------------------
def knn_cv(X, y, k=5, n_splits=5, seed=0):
    """Returns (mean, sd, per_fold). Standardisation is applied to the
    whole matrix before splitting -- NOT refit per fold. That is what the
    repository's other knn_cv implementations do, and the comparison is
    only meaningful if every representation is treated identically."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    accs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed).split(X, y):
        accs.append(KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr]).score(X[te], y[te]))
    accs = np.asarray(accs, dtype=float)
    return float(accs.mean()), float(accs.std()), [float(a) for a in accs]


def separation_stats(candidates):
    """Group separation + rank correlation on the literature screening set.

    ddof=0 on the active-group sd reproduces the manuscript's published
    5.13 (PeptideBERT) and 2.15 (BLOSUM62); ddof=1 gives 4.44 and 1.86
    and matches neither. Candidates flagged in_training are dropped --
    the repository flags them but the hand-derived numbers did not always
    exclude them.
    """
    clean = [c for c in candidates if not c.get("in_training", False)]
    a = np.array([c["mean_dist"] for c in clean if c["expected"] == "ACTIVE"])
    b = np.array([c["mean_dist"] for c in clean if c["expected"] in NON_ACTIVE])
    gap = float(b.mean() - a.mean())
    sd = float(a.std(ddof=0))
    rho, p = spearmanr([c["mean_dist"] for c in clean],
                       [ORD3[c["expected"]] for c in clean])
    pairs = [(x, z) for x in a for z in b]
    return dict(
        n_active=len(a), n_non_active=len(b),
        active_mean=float(a.mean()), non_active_mean=float(b.mean()),
        gap=gap, active_sd_ddof0=sd, gap_in_active_sd=gap / sd,
        spearman_rho=float(rho), spearman_p=float(p),
        pairwise_correct=int(sum(1 for x, z in pairs if x < z)),
        pairwise_total=len(pairs),
    )


def comparator_stats():
    """Recompute the PeptideBERT and BLOSUM62 screening statistics through
    the identical code path, so the three-way comparison in the manuscript
    is reproducible rather than hand-assembled."""
    out = {}
    pb_path = os.path.join(RESULTS, "mature_screening_heldout.json")
    if os.path.exists(pb_path):
        pb = json.load(open(pb_path))["candidates"]
        out["PeptideBERT"] = separation_stats([c for c in pb if c.get("group") == "literature"])
    bl_path = os.path.join(RESULTS, "blosum_head_to_head.json")
    if os.path.exists(bl_path):
        out["BLOSUM62"] = separation_stats(json.load(open(bl_path))["mature_screening"]["candidates"])
    return out


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def run(checkpoint):
    print("=" * 70)
    print(f"CHECKPOINT: {checkpoint}   device={DEVICE}")
    print("=" * 70)
    model, bc, meta = load_esm2(checkpoint)
    layer = meta["num_layers"]
    print(f"  embed_dim={meta['embed_dim']}  layers={layer}  params={meta['n_parameters']:,}")
    out = {"provenance": meta}

    # ---- Experiment 1: main corpus -------------------------------------
    records = json.load(open(os.path.join(RESULTS, "splits.json")))["records"]
    fam2id = {f: i for i, f in enumerate(["RALF", "CLE", "PSK", "PEP", "DECOY"])}
    y = np.array([fam2id[r["family"]] for r in records])
    X = cached_embeddings(f"{checkpoint}__main", [r["seq"] for r in records], model, bc, layer)

    m, s, folds = knn_cv(X, y)
    out["main_full"] = dict(mean=m, std=s, n=len(records), folds=folds)
    print(f"\n  full corpus (n={len(records)}):     {m:.4f} +/- {s:.4f}   folds={np.round(folds,4).tolist()}")

    band = json.load(open(os.path.join(RESULTS, "embeddings_analysis.json")))["length_matched_indices"]
    m2, s2, folds2 = knn_cv(X[band], y[band])
    out["main_length_matched"] = dict(mean=m2, std=s2, n=len(band), folds=folds2)
    print(f"  length-matched (n={len(band)}):   {m2:.4f} +/- {s2:.4f}   folds={np.round(folds2,4).tolist()}")

    # ---- Experiment 2: mature screening --------------------------------
    heldout = json.load(open(os.path.join(RESULTS, "mature_screening_heldout.json")))
    lit = [c for c in heldout["candidates"] if c["group"] == "literature"]
    meta_m = json.load(open(os.path.join(RESULTS, "mature_pretrain.json")))
    recs = meta_m["records"]
    train_idx = [i for i in meta_m["train_idx"] if recs[i]["seq"] not in HOLDOUT_SEQS]
    train_seqs = [recs[i]["seq"] for i in train_idx]
    train_lab = [recs[i]["family"] for i in train_idx]

    train_emb = cached_embeddings(f"{checkpoint}__mature_train", train_seqs, model, bc, layer)
    mu, sd = train_emb.mean(0), train_emb.std(0) + 1e-8
    train_std = (train_emb - mu) / sd

    def score_vec(e):
        e = (e - mu) / sd
        dist = np.linalg.norm(train_std - e[None, :], axis=1)
        nn = np.argsort(dist)[:5]
        fam = Counter(train_lab[i] for i in nn).most_common(1)[0][0]
        return dict(predicted_family=fam, mean_dist=float(dist[nn].mean()))

    val_seqs = [recs[i]["seq"] for i in meta_m["val_idx"]]
    val_emb = cached_embeddings(f"{checkpoint}__mature_val", val_seqs, model, bc, layer)
    ref = float(np.mean([score_vec(e)["mean_dist"] for e in val_emb]))
    print(f"\n  in-distribution reference mean_dist: {ref:.3f}")

    lit = sorted(lit, key=lambda r: r["name"])
    lit_emb = cached_embeddings(f"{checkpoint}__literature", [c["seq"] for c in lit], model, bc, layer)
    scored = [dict(name=c["name"], expected=c["expected"], seq=c["seq"],
                   in_training=c["seq"] in train_seqs, **score_vec(e))
              for c, e in zip(lit, lit_emb)]
    for r in sorted(scored, key=lambda x: x["mean_dist"]):
        print(f"    d={r['mean_dist']:6.3f}  {r['name']:20s} {r['expected']:9s}"
              f"{' [LEAK]' if r['in_training'] else ''}")

    st = separation_stats(scored)
    out["mature_screening"] = dict(reference_mean_dist=ref, candidates=scored, **st)
    print(f"\n    gap={st['gap']:.3f} units = {st['gap_in_active_sd']:.3f} active-group sd (ddof=0)")
    print(f"    rho={st['spearman_rho']:.3f}  P={st['spearman_p']:.4f}"
          f"   pairwise {st['pairwise_correct']}/{st['pairwise_total']}")

    out["comparators"] = comparator_stats()
    for k, v in out["comparators"].items():
        print(f"    [{k:12s}] gap={v['gap']:.3f} = {v['gap_in_active_sd']:.3f} sd   "
              f"rho={v['spearman_rho']:.3f}  pairwise {v['pairwise_correct']}/{v['pairwise_total']}")

    path = os.path.join(RESULTS, f"esm2_head_to_head__{checkpoint}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n  wrote {path}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    for ckpt in CHECKPOINTS:
        run(ckpt)

# --------------------------------------------------------------------
# LOCAL RUN
# --------------------------------------------------------------------
# $ python3 -m venv .venv && source .venv/bin/activate
# $ pip install "fair-esm>=2.0" torch numpy scipy scikit-learn
# $ # then set RESULTS above to your local results/ directory and run:
# $ python3 head_to_head_esm2.py
#
# Weights land in ~/.cache/torch/hub/checkpoints (8M checkpoint ~31 MB,
# 650M ~2.5 GB). CPU is fine for 8M; 650M wants a GPU or patience.
#
# NOTE on Python version: the Colab script nests double quotes inside
# f-strings, e.g. f"{emb["knn_table"][k]["mean"]:.3f}". That is legal only
# on Python 3.12+ (PEP 701). On 3.11 or earlier it is a SyntaxError. This
# file avoids the construction entirely, so it runs on 3.9+.
