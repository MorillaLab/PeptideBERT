"""
Embedding geometry analysis -- the peptide analogue of NucleicBERT's Fig. 2b
/ Extended Data Table 1: does the pretrained backbone organise sequences
into family-coherent regions beyond what's recoverable from trivial
composition statistics alone?

Baselines mirror the original paper's logic, adapted from a 4-letter
nucleotide alphabet to 20 amino acids:
  - dipeptide composition (400-dim)  -- stands in for their 6-mer baseline;
    a hexapeptide baseline (20^6 possible motifs) would be far too sparse
    at this corpus size, so dipeptide (20^2) is the length-appropriate
    analogue, not an arbitrary substitution.
  - length only (1-dim)
  - length + single-amino-acid composition (21-dim) -- direct analogue of
    their "length + ACGU%" baseline.

Outputs: ../results/embeddings_analysis.json (kNN accuracy table + 2D
t-SNE coordinates for the three main views: pretrained / random-init /
dipeptide).
"""
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from model import PeptideBERT, MAX_LEN
from peptide_data import encode, AMINO_ACIDS, PAD_ID, CLS_ID, SEP_ID
from pretrain import load_model, _release_memory

RESULTS = "/home/claude/s2pepanalyst/results"
FAMILY_NAMES = ["RALF", "CLE", "PSK", "PEP", "DECOY"]
FAM2ID = {f: i for i, f in enumerate(FAMILY_NAMES)}
SEED = 0


def load_records():
    with open(f"{RESULTS}/splits.json") as f:
        return json.load(f)["records"]


def model_embeddings(model, records, batch_size=20):
    """Mean-pool the final-layer hidden state over valid (non-special,
    non-pad) positions -- one embedding vector per sequence."""
    embs = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        ids_list, attn_list, valid_masks = [], [], []
        for r in batch:
            ids, attn = encode(r["seq"], MAX_LEN)
            ids_list.append(ids); attn_list.append(attn)
            valid_masks.append(((ids != PAD_ID) & (ids != CLS_ID) & (ids != SEP_ID)).astype(np.float64))
        ids_b = np.stack(ids_list); attn_b = np.stack(attn_list)
        hidden, _ = model.encode(ids_b, attn_b)
        h = hidden.data  # (B,T,D) -- read-only numpy view, no need for graph here
        vm = np.stack(valid_masks)[:, :, None]  # (B,T,1)
        pooled = (h * vm).sum(axis=1) / np.clip(vm.sum(axis=1), 1, None)
        embs.append(pooled)
        del hidden
    _release_memory()
    return np.concatenate(embs, axis=0)


def dipeptide_features(seqs):
    idx = {a: i for i, a in enumerate(AMINO_ACIDS)}
    feats = np.zeros((len(seqs), len(AMINO_ACIDS) ** 2))
    for row, s in enumerate(seqs):
        n = 0
        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            if a in idx and b in idx:
                feats[row, idx[a] * len(AMINO_ACIDS) + idx[b]] += 1
                n += 1
        if n > 0:
            feats[row] /= n
    return feats


def length_and_composition_features(seqs):
    idx = {a: i for i, a in enumerate(AMINO_ACIDS)}
    feats = np.zeros((len(seqs), 1 + len(AMINO_ACIDS)))
    for row, s in enumerate(seqs):
        feats[row, 0] = len(s)
        for c in s:
            if c in idx:
                feats[row, 1 + idx[c]] += 1.0 / len(s)
    return feats


def knn_cv_accuracy(X, y, k=5, n_splits=5, seed=SEED):
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for train_i, test_i in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_i], y[train_i])
        accs.append(clf.score(X[test_i], y[test_i]))
    return float(np.mean(accs)), float(np.std(accs))


def length_matched_band(records, lo=75, hi=105):
    """Indices of records whose sequence length falls in [lo, hi] -- the
    peptide analogue of NucleicBERT's Extended Data Table 1 Setting C
    ('5 classes, length-matched'). With indel augmentation, family length
    distributions now overlap substantially (see peptide_data.py's own
    diagnostic printout), so this band is chosen to give a reasonable,
    non-trivial n from every class rather than to flatter any one model."""
    return [i for i, r in enumerate(records) if lo <= len(r["seq"]) <= hi]


def main():
    records = load_records()
    seqs = [r["seq"] for r in records]
    y = np.array([FAM2ID[r["family"]] for r in records])

    pretrained = PeptideBERT(seed=SEED); load_model(pretrained, f"{RESULTS}/pretrained_model.npz")
    random_init = PeptideBERT(seed=SEED); load_model(random_init, f"{RESULTS}/randominit_model.npz")

    print("Computing pooled embeddings...")
    emb_pre = model_embeddings(pretrained, records)
    emb_rnd = model_embeddings(random_init, records)
    dipep = dipeptide_features(seqs)
    length_only = np.array([[len(s)] for s in seqs], dtype=np.float64)
    length_comp = length_and_composition_features(seqs)

    print("Running 5-fold kNN cross-validation for each feature space (all classes, full corpus)...")
    knn_table = {}
    for name, X in [("pretrained", emb_pre), ("random_init", emb_rnd),
                     ("dipeptide_composition", dipep), ("length_only", length_only),
                     ("length_plus_composition", length_comp)]:
        mean_acc, std_acc = knn_cv_accuracy(X, y, k=5)
        knn_table[name] = dict(mean=mean_acc, std=std_acc)
        print(f"  {name:25s} kNN acc = {mean_acc:.3f} +/- {std_acc:.3f}")

    band = length_matched_band(records, lo=75, hi=105)
    band_lengths_by_fam = {}
    for i in band:
        band_lengths_by_fam.setdefault(records[i]["family"], 0)
        band_lengths_by_fam[records[i]["family"]] += 1
    print(f"\nLength-matched band [75,105] aa: n={len(band)}  per-class counts={band_lengths_by_fam}")
    y_band = y[band]
    print("Running 5-fold kNN cross-validation WITHIN the length-matched band "
          "(this is the control that actually tests whether pretraining adds\n"
          "anything once length can no longer act as a shortcut)...")
    knn_table_matched = {}
    for name, X in [("pretrained", emb_pre), ("random_init", emb_rnd),
                     ("dipeptide_composition", dipep), ("length_only", length_only),
                     ("length_plus_composition", length_comp)]:
        X_band = X[band]
        mean_acc, std_acc = knn_cv_accuracy(X_band, y_band, k=5, n_splits=5)
        knn_table_matched[name] = dict(mean=mean_acc, std=std_acc, n=len(band))
        print(f"  {name:25s} kNN acc = {mean_acc:.3f} +/- {std_acc:.3f}  (n={len(band)})")

    print("Running t-SNE projections (pretrained / random-init / dipeptide)...")
    proj = {}
    for name, X in [("pretrained", emb_pre), ("random_init", emb_rnd), ("dipeptide_composition", dipep)]:
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        coords = TSNE(n_components=2, perplexity=25, random_state=SEED, init="pca").fit_transform(Xs)
        proj[name] = coords.tolist()

    out = dict(knn_table=knn_table, knn_table_length_matched=knn_table_matched,
               length_matched_band=[75, 105], length_matched_indices=band,
               projections=proj, family=[FAMILY_NAMES[i] for i in y],
               family_names=FAMILY_NAMES, lengths=[len(s) for s in seqs])
    with open(f"{RESULTS}/embeddings_analysis.json", "w") as f:
        json.dump(out, f)
    print("Done. Wrote embeddings_analysis.json")


if __name__ == "__main__":
    main()
