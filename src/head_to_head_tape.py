#!/usr/bin/env python3
"""
TAPE head-to-head, in two phases.

WHY TAPE. S2-PepAnalyst does not embed with ESM-2 alone. It runs two protein
language models in parallel -- ESM-2 (1280-dim) and TAPE (768-dim) -- reshapes
each into an image, scores both through GeoTop, and uses reinforcement learning
to select between them per iteration, concatenating them for the final
representation. A head-to-head that claims to compare against "the
representation the tool actually uses" therefore has to cover both.

TAPE here is the `bert-base` transformer of Rao et al. (2019): 12 layers,
768-dim, ~38M parameters, pretrained on Pfam. It sits between PhytoPepBERT
(169,567) and ESM-2 650M on the parameter axis, and is a different pretraining
corpus and model generation, so it also tests whether the parity result is
specific to ESM.

Pooling, folds and statistics are identical to head_to_head_esm2.py. Only the
embedding function differs, which is the point.

TWO PHASES, same reason as the ESM-2 arm: PyTorch links LLVM's OpenMP while
NumPy/scikit-learn link Intel's via MKL, and both in one process segfaults on
macOS. Embedding runs in a subprocess that exits before analysis starts.

    pip install tape_proteins torch
    python3 src/head_to_head_tape.py
    python3 src/head_to_head_tape.py --phase analyze   # caches must exist
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "tape_cache")
MODEL = "bert-base"
NEGATIVE_CANDIDATES = ("NONSSP", "DECOY", "NEGATIVE")


def load_corpus():
    with open(os.path.join(RESULTS, "splits.json")) as f:
        return json.load(f)


def family_names(recs):
    fams = sorted({r["family"] for r in recs})
    neg = [f for f in fams if f in NEGATIVE_CANDIDATES]
    return [f for f in fams if f not in neg] + neg


def cache_path(tag):
    return os.path.join(CACHE, f"tape_{MODEL}__{tag}.npy")


def caches_complete():
    return (os.path.exists(os.path.join(CACHE, f"tape_{MODEL}__meta.json"))
            and os.path.exists(cache_path("main")))


# ---------------------------------------------------------------- phase 1
def embed_phase():
    """Only this function imports torch. It runs in its own process."""
    import torch
    from tape import ProteinBertModel, TAPETokenizer

    os.makedirs(CACHE, exist_ok=True)
    model = ProteinBertModel.from_pretrained(MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tok = TAPETokenizer(vocab="iupac")
    meta = dict(model=f"tape/{MODEL}",
                embed_dim=int(model.config.hidden_size),
                num_layers=int(model.config.num_hidden_layers),
                n_parameters=int(sum(p.numel() for p in model.parameters())),
                torch_version=torch.__version__, device=device)
    print(f"  embed_dim={meta['embed_dim']}  layers={meta['num_layers']}  "
          f"params={meta['n_parameters']:,}  device={device}")

    def embed(seqs, batch_size=8):
        """Mean-pool final-layer representations over residue positions only.

        TAPETokenizer.encode wraps the sequence as [CLS] X [SEP], exactly as
        fair-esm does, so the slice 1:len(seq)+1 drops both and right-padding
        introduced by batching is never read -- identical to the ESM-2 arm."""
        out = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            ids = [tok.encode(s) for s in chunk]
            width = max(len(x) for x in ids)
            batch = np.zeros((len(chunk), width), dtype=np.int64)
            mask = np.zeros((len(chunk), width), dtype=np.int64)
            for j, x in enumerate(ids):
                batch[j, :len(x)] = x
                mask[j, :len(x)] = 1
            with torch.no_grad():
                seq_out = model(torch.from_numpy(batch).to(device),
                                input_mask=torch.from_numpy(mask).to(device))[0]
            for j, s in enumerate(chunk):
                out.append(seq_out[j, 1:len(s) + 1].mean(0).float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"      {min(i + batch_size, len(seqs))}/{len(seqs)}", end="\r")
        return np.asarray(out)

    d = load_corpus()
    seqs = [r["seq"] for r in d["records"]]
    p = cache_path("main")
    if os.path.exists(p) and np.load(p).shape[0] == len(seqs):
        print(f"    [cache] main: {np.load(p).shape}")
    else:
        X = embed(seqs)
        np.save(p, X)
        print(f"    [computed] main: {X.shape}        ")

    json.dump(meta, open(os.path.join(CACHE, f"tape_{MODEL}__meta.json"), "w"), indent=1)
    print("  embeddings cached; torch process exiting")


# ---------------------------------------------------------------- phase 2
def knn_cv(X, y, groups, k=5, n_splits=5, seed=0):
    """Cluster-aware, identical to embeddings_analysis.py and the ESM-2 arm."""
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


def analyze_phase():
    """No torch anywhere below this line."""
    meta = json.load(open(os.path.join(CACHE, f"tape_{MODEL}__meta.json")))
    out = {"provenance": meta}
    d = load_corpus()
    recs = d["records"]
    fams = family_names(recs)
    y = np.array([{f: i for i, f in enumerate(fams)}[r["family"]] for r in recs])
    groups = np.array(d["clusters"]) if "clusters" in d else np.arange(len(recs))
    if "clusters" not in d:
        print("  WARNING: no clusters in splits.json; folds are not homology-aware")
    X = np.load(cache_path("main"))
    print(f"\n  corpus: {len(recs)} sequences, {len(fams)} classes {fams}")

    m, s, folds = knn_cv(X, y, groups)
    out["main_full"] = dict(mean=m, std=s, n=len(recs), folds=folds)
    print(f"  full corpus:    {m:.4f} +/- {s:.4f}   folds={np.round(folds,4).tolist()}")

    ea = os.path.join(RESULTS, "embeddings_analysis.json")
    if os.path.exists(ea):
        emb = json.load(open(ea))
        band = emb.get("length_matched_indices")
        if band:
            m2, s2, f2 = knn_cv(X[band], y[band], groups[band])
            out["main_length_matched"] = dict(mean=m2, std=s2, n=len(band), folds=f2)
            print(f"  length-matched: {m2:.4f} +/- {s2:.4f}")
        print("\n  on the same folds:")
        for k2, v in emb["knn_table"].items():
            print(f"    {k2:25s} {v['mean']:.3f}")
        for f in sorted(os.listdir(RESULTS)):
            if f.startswith("esm2_head_to_head__"):
                e = json.load(open(os.path.join(RESULTS, f)))
                print(f"    {'ESM-2 ' + e['provenance']['checkpoint']:25s} "
                      f"{e['main_full']['mean']:.3f}  "
                      f"({e['provenance']['n_parameters']:,} params)")
        b = os.path.join(RESULTS, "blosum_head_to_head.json")
        if os.path.exists(b):
            print(f"    {'BLOSUM62':25s} {json.load(open(b))['main_full']['mean']:.3f}")
        print(f"    {'TAPE ' + MODEL:25s} {m:.3f}  ({meta['n_parameters']:,} params)")

    p = os.path.join(RESULTS, f"tape_head_to_head__{MODEL}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {p}")


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["embed", "analyze", "both"], default="both")
    a = ap.parse_args()
    print("=" * 70)
    print(f"TAPE {MODEL}  (Rao et al. 2019)")
    print("=" * 70)
    if a.phase == "embed":
        embed_phase()
        return
    if a.phase == "both":
        if caches_complete():
            print("  embeddings already cached; skipping torch entirely")
        else:
            print("  embedding in a subprocess (keeps torch out of this process)")
            rc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                 "--phase", "embed"]).returncode
            if rc != 0:
                sys.exit(
                    f"\nThe embedding phase failed (exit {rc}).\n"
                    f"  - missing packages?  pip install tape_proteins torch\n"
                    f"  - tape_proteins is from 2019 and pins old dependencies; if it\n"
                    f"    will not install cleanly, try it in its own virtualenv with\n"
                    f"    an older torch, then run --phase analyze from your main env\n"
                    f"    (the cached .npy is all the analysis needs).\n")
    if not caches_complete():
        sys.exit("caches missing; run --phase embed first")
    analyze_phase()


if __name__ == "__main__":
    main()
