"""Pretrain + evaluate a mature-peptide PeptideBERT (MAX_LEN=32), then run
the tomato screening test that the precursor-scale model failed.

Single script (rather than the multi-file layout of the precursor
pipeline) because at MAX_LEN=32 the whole thing runs in a couple of
minutes -- no checkpointing gymnastics needed.

Usage: python3 mature_pipeline.py
"""
import json
import numpy as np
from collections import Counter

from tensor import Tensor
from model import PeptideBERT, Adam
import mature_peptide_data as M
from pretrain import _release_memory

RESULTS = "/home/claude/s2pepanalyst/results"
SEED = 0
N_EPOCHS = 60
BATCH = 24
LR = 3e-3
FAMS = ["CLE", "PEP", "SYSTEMIN", "DECOY"]
FAM2ID = {f: i for i, f in enumerate(FAMS)}


def encode_batch(records, idxs, rng=None, mlm=False):
    ids_l, attn_l, lab_l = [], [], []
    for i in idxs:
        ids, attn = M.encode(records[i]["seq"])
        if mlm:
            ids, lab = M.mask_tokens(ids, rng)
            lab_l.append(lab)
        ids_l.append(ids); attn_l.append(attn)
    out = [np.stack(ids_l), np.stack(attn_l)]
    if mlm:
        out.append(np.stack(lab_l))
    return out


def mlm_loss(model, ids_b, attn_b, lab_b):
    hidden, _ = model.encode(ids_b, attn_b)
    flat = lab_b.reshape(-1)
    mask = (flat != -1).astype(np.float64)
    targets = np.where(flat == -1, 0, flat)
    logits = model.mlm_logits(hidden)
    loss = logits.cross_entropy(targets, mask=mask)
    acc = loss.aux["correct"].sum() / max(mask.sum(), 1.0)
    return loss, float(acc)


def pooled_embeddings(model, records, idxs=None, batch=40):
    idxs = list(range(len(records))) if idxs is None else idxs
    embs = []
    for s in range(0, len(idxs), batch):
        chunk = idxs[s:s + batch]
        ids_b, attn_b = encode_batch(records, chunk)
        valid = ((ids_b != M.PAD_ID) & (ids_b != M.CLS_ID) & (ids_b != M.SEP_ID)).astype(np.float64)
        hidden, _ = model.encode(ids_b, attn_b)
        h = hidden.data
        vm = valid[:, :, None]
        embs.append((h * vm).sum(axis=1) / np.clip(vm.sum(axis=1), 1, None))
        del hidden
    _release_memory()
    return np.concatenate(embs, axis=0)


def knn_cv(X, y, k=5, n_splits=5, seed=SEED):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    accs = []
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        c = KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr])
        accs.append(c.score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    rng = np.random.RandomState(SEED)
    records = M.build_mature_corpus(seed=SEED)
    y = np.array([FAM2ID[r["family"]] for r in records])

    # stratified split
    tr_idx, va_idx = [], []
    for f in FAMS:
        idx = [i for i, r in enumerate(records) if r["family"] == f]
        rng.shuffle(idx)
        n_val = max(1, int(0.2 * len(idx)))
        va_idx += idx[:n_val]; tr_idx += idx[n_val:]
    print(f"corpus {len(records)}  train {len(tr_idx)}  val {len(va_idx)}  MAX_LEN={M.MAX_LEN}")

    pre = PeptideBERT(n_classes=len(FAMS), seed=SEED, max_len=M.MAX_LEN)
    rnd = PeptideBERT(n_classes=len(FAMS), seed=SEED, max_len=M.MAX_LEN)  # untouched control
    opt = Adam(pre.backbone_params() + [pre.p[k] for k in
               ("mlm.W1", "mlm.b1", "mlm.ln.g", "mlm.ln.b", "mlm.W2", "mlm.b2")], lr=LR)

    curve = []
    for ep in range(1, N_EPOCHS + 1):
        idx = tr_idx.copy(); rng.shuffle(idx)
        tl, ta, n = 0.0, 0.0, 0
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            ids_b, attn_b, lab_b = encode_batch(records, b, rng, mlm=True)
            loss, acc = mlm_loss(pre, ids_b, attn_b, lab_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss.data); ta += acc; n += 1
            del loss
        ids_b, attn_b, lab_b = encode_batch(records, va_idx, rng, mlm=True)
        vloss, vacc = mlm_loss(pre, ids_b, attn_b, lab_b)
        curve.append(dict(epoch=ep, train_loss=tl / n, train_acc=ta / n,
                          val_loss=float(vloss.data), val_acc=vacc))
        del vloss
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train acc {ta/n:.3f}  val acc {vacc:.3f}")
        _release_memory()

    # embedding geometry, pretrained vs random-init
    print("\nEmbedding-space kNN (family, 5-fold, k=5):")
    emb_pre = pooled_embeddings(pre, records)
    emb_rnd = pooled_embeddings(rnd, records)
    knn = {}
    for name, X in [("pretrained", emb_pre), ("random_init", emb_rnd)]:
        m, s = knn_cv(X, y)
        knn[name] = dict(mean=m, std=s)
        print(f"  {name:12s} {m:.3f} +/- {s:.3f}")
    lengths = np.array([[len(r["seq"])] for r in records], dtype=float)
    m, s = knn_cv(lengths, y)
    knn["length_only"] = dict(mean=m, std=s)
    print(f"  {'length_only':12s} {m:.3f} +/- {s:.3f}   (chance = {1/len(FAMS):.3f})")

    np.savez(f"{RESULTS}/mature_pretrained.npz", **{k: v.data for k, v in pre.p.items()})
    np.savez(f"{RESULTS}/mature_randominit.npz", **{k: v.data for k, v in rnd.p.items()})
    with open(f"{RESULTS}/mature_pretrain.json", "w") as f:
        json.dump(dict(curve=curve, knn=knn,
                       records=[{k: v for k, v in r.items()} for r in records],
                       train_idx=tr_idx, val_idx=va_idx), f)
    print("\nsaved mature_pretrained.npz / mature_pretrain.json")


if __name__ == "__main__":
    main()
