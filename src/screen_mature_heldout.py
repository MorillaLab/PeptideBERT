"""Rigorous re-run: retrain with the literature ground-truth peptides
EXCLUDED from training, so the screening test has held-out ACTIVES as well
as held-out inactives.

The first pass (screen_mature_candidates.py) passed tests 1 and 2 but
test 3 was under-powered: every canonical active peptide (CLV3, TDIF,
GrCLE1-1, AtPep1) was a training seed, so there was nothing held out to
compare the known-inactives against. Reporting a "separates correctly"
claim off that would have been meaningless. This script fixes it by
removing those exact sequences (and only those) from the training split
before pretraining, then scoring them blind.
"""
import json
import numpy as np
from model import PeptideBERT, Adam
import mature_peptide_data as M
from mature_pipeline import (encode_batch, mlm_loss, pooled_embeddings, knn_cv,
                             FAMS, FAM2ID, RESULTS, SEED, N_EPOCHS, BATCH, LR)
from screen_mature_candidates import build_test_set, knn_score
from pretrain import _release_memory

HOLDOUT_SEQS = {
    "ATKVKAKQRGKEKVSSGRPGQHN",  # AtPep1 full      ACTIVE
    "RTVPSGPDPLHH",             # CLV3 12-mer      ACTIVE
    "HEVPSGPNPISN",             # TDIF             ACTIVE
    "RVIPGGPDPLHN",             # GrCLE1-1         ACTIVE
}


def main():
    rng = np.random.RandomState(SEED)
    records = M.build_mature_corpus(seed=SEED)
    y = np.array([FAM2ID[r["family"]] for r in records])

    tr_idx, va_idx = [], []
    for f in FAMS:
        idx = [i for i, r in enumerate(records) if r["family"] == f]
        rng.shuffle(idx)
        n_val = max(1, int(0.2 * len(idx)))
        va_idx += idx[:n_val]; tr_idx += idx[n_val:]

    # enforce holdout: no exact test sequence may remain in training
    removed = [i for i in tr_idx if records[i]["seq"] in HOLDOUT_SEQS]
    tr_idx = [i for i in tr_idx if records[i]["seq"] not in HOLDOUT_SEQS]
    print(f"corpus {len(records)}  train {len(tr_idx)} (removed {len(removed)} holdout-matching)  val {len(va_idx)}")

    pre = PeptideBERT(n_classes=len(FAMS), seed=SEED, max_len=M.MAX_LEN)
    opt = Adam(pre.backbone_params() + [pre.p[k] for k in
               ("mlm.W1", "mlm.b1", "mlm.ln.g", "mlm.ln.b", "mlm.W2", "mlm.b2")], lr=LR)
    for ep in range(1, N_EPOCHS + 1):
        idx = tr_idx.copy(); rng.shuffle(idx)
        ta, n = 0.0, 0
        for s in range(0, len(idx), BATCH):
            ids_b, attn_b, lab_b = encode_batch(records, idx[s:s + BATCH], rng, mlm=True)
            loss, acc = mlm_loss(pre, ids_b, attn_b, lab_b)
            opt.zero_grad(); loss.backward(); opt.step()
            ta += acc; n += 1
            del loss
        if ep % 20 == 0:
            print(f"  ep {ep:3d}  train acc {ta/n:.3f}")
        _release_memory()

    train_emb = pooled_embeddings(pre, records, idxs=tr_idx)
    mu, sd = train_emb.mean(0), train_emb.std(0) + 1e-8
    train_std = (train_emb - mu) / sd
    train_lab = [records[i]["family"] for i in tr_idx]
    train_seqs = set(records[i]["seq"] for i in tr_idx)

    val_emb = pooled_embeddings(pre, records, idxs=va_idx)
    val_d = [knn_score((e - mu) / sd, train_std, train_lab)["mean_dist"] for e in val_emb]
    ref = float(np.mean(val_d))
    print(f"\nheld-out in-distribution reference mean_dist = {ref:.2f}")

    rows = []
    for t in build_test_set():
        emb = pooled_embeddings(pre, [dict(seq=t["seq"])])[0]
        sc = knn_score((emb - mu) / sd, train_std, train_lab)
        rows.append(dict(**t, **sc, in_training=t["seq"] in train_seqs, length=len(t["seq"])))

    lit = [r for r in rows if r["group"] == "literature"]
    leaked = [r for r in lit if r["in_training"]]
    print(f"\nliterature peptides: {len(lit)} total, {len(leaked)} still leaked "
          f"(should be 0 for the four holdouts)")

    print("\nTEST 3, properly held out -- ranked by distance (lower = more 'in-family'):")
    for r in sorted(lit, key=lambda x: x["mean_dist"]):
        tag = " [LEAK]" if r["in_training"] else ""
        print(f"   d={r['mean_dist']:5.2f}  {r['name']:20s} expected={r['expected']:10s} "
              f"pred={r['predicted_family']:9s}{tag}")

    clean = [r for r in lit if not r["in_training"]]
    act = [r["mean_dist"] for r in clean if r["expected"] == "ACTIVE"]
    inact = [r["mean_dist"] for r in clean if r["expected"] in ("INACTIVE", "REDUCED", "PARTIAL")]
    print(f"\n   held-out known-ACTIVE   (n={len(act)}): mean d = {np.mean(act):.2f}  {[round(x,2) for x in act]}")
    print(f"   held-out known-REDUCED/INACTIVE (n={len(inact)}): mean d = {np.mean(inact):.2f}  {[round(x,2) for x in inact]}")
    verdict = "SEPARATES CORRECTLY" if np.mean(act) < np.mean(inact) else "DOES NOT SEPARATE"
    print(f"   -> {verdict}")

    # rank-based check: how many active/inactive pairs are ordered correctly?
    pairs = [(a, i) for a in act for i in inact]
    correct = sum(1 for a, i in pairs if a < i)
    print(f"   pairwise ordering: {correct}/{len(pairs)} active-vs-inactive pairs ranked correctly "
          f"({correct/len(pairs)*100:.0f}%)")

    sysres = {r["name"]: r["mean_dist"] for r in rows if r["group"] == "systemin"}
    conservative = [v for k, v in sysres.items() if k.startswith("systemin_var")]
    print(f"\nTEST 2 (clean): disrupted d={sysres['systemin_disrupted']:.2f} vs conservative "
          f"variants mean d={np.mean(conservative):.2f} -> "
          f"{'PASS' if sysres['systemin_disrupted'] > max(conservative) else 'FAIL'}")

    with open(f"{RESULTS}/mature_screening_heldout.json", "w") as f:
        json.dump(dict(candidates=rows, reference_mean_dist=ref,
                       active_mean=float(np.mean(act)), inactive_mean=float(np.mean(inact)),
                       pairwise_correct=correct, pairwise_total=len(pairs)), f, indent=1)
    np.savez(f"{RESULTS}/mature_pretrained_heldout.npz", **{k: v.data for k, v in pre.p.items()})
    print("\nwrote mature_screening_heldout.json")


if __name__ == "__main__":
    main()
