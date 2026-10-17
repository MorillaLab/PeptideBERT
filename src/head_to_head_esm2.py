#!/usr/bin/env python3
"""
ESM-2 head-to-head, in two phases.

WHY TWO PHASES: on macOS with conda, PyTorch links LLVM's OpenMP (libomp)
while numpy/scipy/scikit-learn link Intel's (libiomp, via MKL). Both in one
process segfaults. The usual advice is KMP_DUPLICATE_LIB_OK=TRUE, which only
suppresses the abort -- the two runtimes still fight, and the documented
failure modes include silently wrong results. Rather than risk that in a
benchmark, this script never loads them together:

    phase "embed"    imports torch + esm, writes .npy caches, exits
    phase "analyze"  imports sklearn + scipy, reads the caches, no torch

Running it plainly does both, with the embed phase in a subprocess that exits
before the analysis starts. Once the caches exist the embed phase is skipped
entirely, so re-runs never touch torch.

It also carries the fixes the real corpus needs: families derived from the
corpus rather than hardcoded, and cluster-aware cross-validation.

    python3 src/head_to_head_esm2.py
    python3 src/head_to_head_esm2.py --phase analyze     # caches must exist
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "esm2_cache")
CHECKPOINTS = ["esm2_t6_8M_UR50D"]

# --sweep runs these in order of size. Batch sizes drop as the model grows so
# the 650M fits comfortably in RAM on a laptop; it is ~2.5 GB to download and
# the slowest arm by a wide margin on CPU.
SWEEP = [
    ("esm2_t6_8M_UR50D",   16),
    ("esm2_t12_35M_UR50D", 16),
    ("esm2_t30_150M_UR50D", 8),
    ("esm2_t33_650M_UR50D", 4),
]
BATCH = dict(SWEEP)

HOLDOUT_SEQS = {"ATKVKAKQRGKEKVSSGRPGQHN", "RTVPSGPDPLHH",
                "HEVPSGPNPISN", "RVIPGGPDPLHN"}
ORD3 = {"ACTIVE": 0, "PARTIAL": 1, "REDUCED": 1, "INACTIVE": 2}
NON_ACTIVE = ("INACTIVE", "REDUCED", "PARTIAL")
NEGATIVE_CANDIDATES = ("NONSSP", "DECOY", "NEGATIVE")


# ---------------------------------------------------------------- shared
def load_corpus():
    with open(os.path.join(RESULTS, "splits.json")) as f:
        return json.load(f)


def family_names(recs):
    fams = sorted({r["family"] for r in recs})
    neg = [f for f in fams if f in NEGATIVE_CANDIDATES]
    return [f for f in fams if f not in neg] + neg


def mature_available():
    """Present AND built from retrieved sequences. The mature corpus shipped
    with this project is generated (9 real seeds, 231 augmented, 80 synthetic
    decoys), so its distances reflect how the augmentation was constructed
    rather than anything about the peptides, and must not be reported beside
    the real-sequence results. Matches the gate in head_to_head_blosum.py."""
    from collections import Counter as _C2
    if not all(os.path.exists(os.path.join(RESULTS, f)) for f in
               ("mature_screening_heldout.json", "mature_pretrain.json")):
        return False
    if os.environ.get("INCLUDE_GENERATED_MATURE"):
        return True
    kinds = _C2(r.get("kind", "UNKNOWN") for r in
                json.load(open(os.path.join(RESULTS, "mature_pretrain.json")))["records"])
    if any(k in kinds for k in ("AUGMENTED", "SYNTHETIC")):
        print(f"  mature arm SKIPPED: corpus is generated, not retrieved "
              f"({dict(kinds)}). Set INCLUDE_GENERATED_MATURE=1 to run it anyway.")
        return False
    return True


def cache_path(ckpt, tag):
    return os.path.join(CACHE, f"{ckpt}__{tag}.npy")


def required_tags():
    tags = ["main"]
    if mature_available():
        tags += ["mature_train", "mature_val", "literature"]
    return tags


def caches_complete(ckpt):
    return (os.path.exists(os.path.join(CACHE, f"{ckpt}__meta.json"))
            and all(os.path.exists(cache_path(ckpt, t)) for t in required_tags()))


# ---------------------------------------------------------------- phase 1
def embed_phase(ckpt):
    """Only this function imports torch. It runs in its own process."""
    import torch
    import esm

    os.makedirs(CACHE, exist_ok=True)
    model, alphabet = getattr(esm.pretrained, ckpt)()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    bc = alphabet.get_batch_converter()
    layer = int(model.num_layers)
    meta = dict(checkpoint=ckpt, embed_dim=int(model.embed_dim), num_layers=layer,
                n_parameters=int(sum(p.numel() for p in model.parameters())),
                fair_esm_version=getattr(esm, "__version__", "unknown"),
                torch_version=torch.__version__, device=device)
    print(f"  embed_dim={meta['embed_dim']}  layers={layer}  "
          f"params={meta['n_parameters']:,}  device={device}")

    def embed(seqs, batch_size=BATCH.get(ckpt, 16)):
        out = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            _, _, toks = bc([(f"s{j}", s) for j, s in enumerate(chunk)])
            with torch.no_grad():
                reps = model(toks.to(device), repr_layers=[layer])["representations"][layer]
            for j, s in enumerate(chunk):
                out.append(reps[j, 1:len(s) + 1].mean(0).float().cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"      {min(i + batch_size, len(seqs))}/{len(seqs)}", end="\r")
        return np.asarray(out)

    def cache(tag, seqs):
        p = cache_path(ckpt, tag)
        if os.path.exists(p) and np.load(p).shape[0] == len(seqs):
            print(f"    [cache] {tag}: {np.load(p).shape}")
            return
        X = embed(seqs)
        np.save(p, X)
        print(f"    [computed] {tag}: {X.shape}        ")

    d = load_corpus()
    cache("main", [r["seq"] for r in d["records"]])

    if mature_available():
        meta_m = json.load(open(os.path.join(RESULTS, "mature_pretrain.json")))
        recs = meta_m["records"]
        tr = [i for i in meta_m["train_idx"] if recs[i]["seq"] not in HOLDOUT_SEQS]
        cache("mature_train", [recs[i]["seq"] for i in tr])
        cache("mature_val", [recs[i]["seq"] for i in meta_m["val_idx"]])
        lit = sorted([c for c in json.load(open(os.path.join(
            RESULTS, "mature_screening_heldout.json")))["candidates"]
            if c["group"] == "literature"], key=lambda r: r["name"])
        cache("literature", [c["seq"] for c in lit])
    else:
        print("    mature arm not run (absent or generated corpus)")

    json.dump(meta, open(os.path.join(CACHE, f"{ckpt}__meta.json"), "w"), indent=1)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print("  embeddings cached; torch process exiting")


# ---------------------------------------------------------------- phase 2
def knn_cv(X, y, groups, k=5, n_splits=5, seed=0):
    """Cluster-aware. Homologues in different folds inflate accuracy by
    3.8-8.6 points on this corpus, so grouping is not optional."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    accs = []
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        c = KNeighborsClassifier(n_neighbors=k).fit((X[tr] - mu) / sd, y[tr])
        accs.append(c.score((X[te] - mu) / sd, y[te]))
    accs = np.asarray(accs, float)
    return float(accs.mean()), float(accs.std()), [float(a) for a in accs]


def separation_stats(cands):
    from scipy.stats import spearmanr
    clean = [c for c in cands if not c.get("in_training", False)]
    a = np.array([c["mean_dist"] for c in clean if c["expected"] == "ACTIVE"])
    b = np.array([c["mean_dist"] for c in clean if c["expected"] in NON_ACTIVE])
    gap, sd = float(b.mean() - a.mean()), float(a.std(ddof=0))
    rho, p = spearmanr([c["mean_dist"] for c in clean],
                       [ORD3[c["expected"]] for c in clean])
    pairs = [(x, z) for x in a for z in b]
    return dict(active_mean=float(a.mean()), non_active_mean=float(b.mean()),
                gap=gap, active_sd_ddof0=sd, gap_in_active_sd=gap / sd,
                margin_in_active_sd=float((b.min() - a.max()) / sd),
                spearman_rho=float(rho), spearman_p=float(p),
                pairwise_correct=int(sum(1 for x, z in pairs if x < z)),
                pairwise_total=len(pairs))


def analyze_phase(ckpt):
    """No torch anywhere below this line."""
    meta = json.load(open(os.path.join(CACHE, f"{ckpt}__meta.json")))
    out = {"provenance": meta}
    d = load_corpus()
    recs = d["records"]
    fams = family_names(recs)
    f2i = {f: i for i, f in enumerate(fams)}
    y = np.array([f2i[r["family"]] for r in recs])
    groups = np.array(d["clusters"]) if "clusters" in d else np.arange(len(recs))
    if "clusters" not in d:
        print("  WARNING: no clusters in splits.json; folds are not homology-aware")
    X = np.load(cache_path(ckpt, "main"))
    print(f"\n  corpus: {len(recs)} sequences, {len(fams)} classes {fams}")
    print(f"  clusters: {len(set(groups.tolist()))}")

    m, s, folds = knn_cv(X, y, groups)
    out["main_full"] = dict(mean=m, std=s, n=len(recs), folds=folds)
    print(f"\n  full corpus:    {m:.4f} +/- {s:.4f}   folds={np.round(folds,4).tolist()}")

    ea = os.path.join(RESULTS, "embeddings_analysis.json")
    if os.path.exists(ea):
        band = json.load(open(ea)).get("length_matched_indices")
        if band:
            m2, s2, f2 = knn_cv(X[band], y[band], groups[band])
            out["main_length_matched"] = dict(mean=m2, std=s2, n=len(band), folds=f2)
            print(f"  length-matched: {m2:.4f} +/- {s2:.4f}   folds={np.round(f2,4).tolist()}")
            print("\n  already-computed numbers on the same splits:")
            t = json.load(open(ea))["knn_table"]
            for k2 in t:
                print(f"    {k2:25s} {t[k2]['mean']:.3f}")
    else:
        print("  (run embeddings_analysis.py first for the length-matched band "
              "and the comparison table)")

    if mature_available():
        lit = sorted([c for c in json.load(open(os.path.join(
            RESULTS, "mature_screening_heldout.json")))["candidates"]
            if c["group"] == "literature"], key=lambda r: r["name"])
        meta_m = json.load(open(os.path.join(RESULTS, "mature_pretrain.json")))
        mrecs = meta_m["records"]
        tr = [i for i in meta_m["train_idx"] if mrecs[i]["seq"] not in HOLDOUT_SEQS]
        train_seqs = [mrecs[i]["seq"] for i in tr]
        train_lab = [mrecs[i]["family"] for i in tr]
        TE = np.load(cache_path(ckpt, "mature_train"))
        mu, sd = TE.mean(0), TE.std(0) + 1e-8
        TS = (TE - mu) / sd

        def score(vec):
            from collections import Counter
            e = (vec - mu) / sd
            dist = np.linalg.norm(TS - e[None, :], axis=1)
            nn = np.argsort(dist)[:5]
            return dict(predicted_family=Counter(train_lab[i] for i in nn).most_common(1)[0][0],
                        mean_dist=float(dist[nn].mean()))

        ref = float(np.mean([score(v)["mean_dist"]
                             for v in np.load(cache_path(ckpt, "mature_val"))]))
        LE = np.load(cache_path(ckpt, "literature"))
        scored = [dict(name=c["name"], expected=c["expected"], seq=c["seq"],
                       in_training=c["seq"] in train_seqs, **score(e))
                  for c, e in zip(lit, LE)]
        print(f"\n  in-distribution reference: {ref:.3f}")
        for r in sorted(scored, key=lambda x: x["mean_dist"]):
            print(f"    d={r['mean_dist']:6.3f}  {r['name']:20s} {r['expected']:9s}"
                  f"{' [LEAK]' if r['in_training'] else ''}")
        st = separation_stats(scored)
        out["mature_screening"] = dict(reference_mean_dist=ref, candidates=scored, **st)
        print(f"    gap={st['gap']:.3f} = {st['gap_in_active_sd']:.3f} sd   "
              f"margin={st['margin_in_active_sd']:.3f} sd   "
              f"rho={st['spearman_rho']:.3f}  "
              f"pairwise {st['pairwise_correct']}/{st['pairwise_total']}")
        if os.environ.get("INCLUDE_GENERATED_MATURE"):
            print("    NOTE: generated corpus, run under an explicit override.")
            print("          Not comparable with the real-sequence results above.")

    p = os.path.join(RESULTS, f"esm2_head_to_head__{ckpt}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {p}")


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["embed", "analyze", "both"], default="both")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="run every checkpoint in SWEEP, smallest first. The 650M "
                         "downloads ~2.5 GB and is slow on CPU; each checkpoint is "
                         "cached and written separately, so it is safe to interrupt "
                         "and resume.")
    a = ap.parse_args()
    if a.checkpoint:
        ckpts = [a.checkpoint]
    elif a.sweep:
        ckpts = [c for c, _ in SWEEP]
    else:
        ckpts = CHECKPOINTS

    for ckpt in ckpts:
        print("=" * 70)
        print(f"CHECKPOINT: {ckpt}")
        print("=" * 70)
        if a.phase == "embed":
            embed_phase(ckpt)
            continue
        if a.phase == "both":
            if caches_complete(ckpt):
                print("  embeddings already cached; skipping torch entirely")
            else:
                print("  embedding in a subprocess (keeps torch out of this process)")
                rc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                     "--phase", "embed", "--checkpoint", ckpt]).returncode
                if rc != 0:
                    sys.exit(
                        f"\nThe embedding phase failed (exit {rc}).\n"
                        f"  - missing packages?   pip install fair-esm torch\n"
                        f"  - no network?         the checkpoint downloads on first use\n"
                        f"  - segfault here?      that is torch alone, not the OpenMP\n"
                        f"                        clash this script works around; try\n"
                        f"                        OMP_NUM_THREADS=1 python3 {os.path.basename(__file__)} "
                        f"--phase embed\n")
        if not caches_complete(ckpt):
            sys.exit(f"caches missing for {ckpt}; run --phase embed first")
        analyze_phase(ckpt)

    if len(ckpts) > 1:
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        rows = []
        for c in ckpts:
            f = os.path.join(RESULTS, f"esm2_head_to_head__{c}.json")
            if os.path.exists(f):
                d = json.load(open(f))
                rows.append((c, d["provenance"]["n_parameters"],
                             d["provenance"]["embed_dim"], d["main_full"]["mean"],
                             d["main_full"]["std"]))
        ea = os.path.join(RESULTS, "embeddings_analysis.json")
        pb = json.load(open(ea))["knn_table"]["pretrained"]["mean"] if os.path.exists(ea) else None
        print(f"  {'checkpoint':24s} {'params':>12s} {'dim':>5s} {'accuracy':>16s}")
        for c, n, d_, m, sd in rows:
            flag = ""
            if pb is not None:
                flag = "  <- overtakes PhytoPepBERT" if m > pb else "  (at or below PhytoPepBERT)"
            print(f"  {c:24s} {n:12,d} {d_:5d}   {m:.4f} +/- {sd:.4f}{flag}")
        if pb is not None:
            print(f"  {'PhytoPepBERT':24s} {169567:12,d} {64:5d}   {pb:.4f}")


if __name__ == "__main__":
    main()
