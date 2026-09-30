"""
The re-test: can the MATURE-scale model do what the precursor-scale model
could not -- triage short candidate peptides for tomato treatment?

Same protocol as screen_tomato_candidates.py so the two are directly
comparable. Three classes of test, in increasing order of how much they
count as real evidence:

  1. IN-DISTRIBUTION CHECK. Are 18-aa candidates now actually inside the
     training manifold? (The precursor model put them 3.3x too far out.)

  2. OUR OWN DESIGNED CONTROL. Does the deliberately-disrupted systemin
     variant now rank WORSE than real systemin? The precursor model
     ranked it best -- exactly backwards.

  3. INDEPENDENT LITERATURE GROUND TRUTH -- the part that actually
     matters, because we did not choose the answer. Published
     structure-activity results give known-active and known-inactive
     peptides that we can score blind:
       ACTIVE   AtPep1(1-23)   full activity (Huffaker et al. 2006)
       ACTIVE   CLV3 12-mer    canonical active CLE (Kondo et al. 2006)
       ACTIVE   TDIF           canonical active CLE (Ito et al. 2006)
       REDUCED  GrCLE1-1m      Gly6->Ala, the clv3-1/clv3-5-equivalent
                               knockout substitution (US 8,569,578)
       INACTIVE Ag-16p         published negative-control peptide in the
                               CLE root assay (Fiers et al. 2005)

LEAKAGE DISCIPLINE: some of these sequences are training seeds. Any
candidate whose exact sequence appears in the training corpus is flagged
`in_training=True` and is EXCLUDED from the headline accuracy claim --
scoring a sequence the model memorised proves nothing. Only held-out
candidates count. This is checked programmatically, not by eye.
"""
import json
import numpy as np
from model import PeptideBERT
import mature_peptide_data as M
from mature_pipeline import pooled_embeddings, FAMS, FAM2ID
from pretrain import _release_memory

RESULTS = "/home/morillalab/s2pepanalyst/results"

TOMATO_SYSTEMIN = "AVQSKPPSKRDPPKMQTD"
FREE_POS = [0, 1, 2, 3, 4, 7, 8, 9, 10]


def systemin_variant(rng, n_subs=2, disrupt=False):
    s = list(TOMATO_SYSTEMIN)
    if disrupt:
        s[6] = "G"; s[14], s[15] = "A", "A"   # break a conserved Pro + invariant C-term
        return "".join(s)
    for p in rng.choice(FREE_POS, size=n_subs, replace=False):
        s[p] = rng.choice(M.AMINO_ACIDS)
    return "".join(s)


def build_test_set(seed=0):
    rng = np.random.RandomState(seed)
    t = [
        dict(name="tomato_systemin", seq=TOMATO_SYSTEMIN, expected="ACTIVE",
             basis="Pearce et al. 1991 Science 253:895", group="systemin"),
    ]
    for i in range(3):
        t.append(dict(name=f"systemin_var_{chr(65+i)}", seq=systemin_variant(rng, rng.randint(2, 4)),
                      expected="UNKNOWN", basis="our designed conservative variant", group="systemin"))
    t.append(dict(name="systemin_disrupted", seq=systemin_variant(rng, disrupt=True),
                  expected="INACTIVE (by design)", basis="our negative control: breaks Pro + C-term",
                  group="systemin"))
    # independent literature ground truth
    t += [
        dict(name="AtPep1_full", seq="ATKVKAKQRGKEKVSSGRPGQHN", expected="ACTIVE",
             basis="Huffaker et al. 2006 PNAS", group="literature"),
        dict(name="AtPep1_13_23", seq="KVSSGRPGQHN", expected="PARTIAL",
             basis="Lin et al. 2024 IJMS: ROS yes, root-growth inhibition reduced", group="literature"),
        dict(name="CLV3_12mer", seq="RTVPSGPDPLHH", expected="ACTIVE",
             basis="Kondo et al. 2006", group="literature"),
        dict(name="TDIF", seq="HEVPSGPNPISN", expected="ACTIVE",
             basis="Ito et al. 2006 Science 313:842", group="literature"),
        dict(name="GrCLE1_1", seq="RVIPGGPDPLHN", expected="ACTIVE",
             basis="US 8,569,578 / Fiers et al. 2005", group="literature"),
        dict(name="GrCLE1_1m_G6A", seq="RVIPGAPDPLHN", expected="REDUCED",
             basis="Gly6->Ala, clv3-1/clv3-5-equivalent knockout substitution", group="literature"),
        dict(name="Ag_16p_negctrl", seq="APNNHHYSSAGRQDQT", expected="INACTIVE",
             basis="published negative control peptide, Fiers et al. 2005", group="literature"),
    ]
    return t


def knn_score(q, train_emb, train_lab, k=5):
    d = np.linalg.norm(train_emb - q[None, :], axis=1)
    nn = np.argsort(d)[:k]
    labs = [train_lab[i] for i in nn]
    from collections import Counter
    top, cnt = Counter(labs).most_common(1)[0]
    return dict(predicted_family=top, vote=cnt, mean_dist=float(d[nn].mean()),
                neighbours=labs)


def main():
    meta = json.load(open(f"{RESULTS}/mature_pretrain.json"))
    records = meta["records"]
    train_idx = meta["train_idx"]
    train_seqs = set(records[i]["seq"] for i in train_idx)
    train_lab = [records[i]["family"] for i in train_idx]

    model = PeptideBERT(n_classes=len(FAMS), seed=0, max_len=M.MAX_LEN)
    d = np.load(f"{RESULTS}/mature_pretrained.npz")
    for k in model.p:
        model.p[k].data = d[k].copy()

    train_emb = pooled_embeddings(model, records, idxs=train_idx)
    mu, sd = train_emb.mean(0), train_emb.std(0) + 1e-8
    train_std = (train_emb - mu) / sd

    # in-distribution reference: held-out validation sequences
    val_emb = pooled_embeddings(model, records, idxs=meta["val_idx"])
    val_d = []
    for e in val_emb:
        val_d.append(knn_score((e - mu) / sd, train_std, train_lab)["mean_dist"])
    print(f"In-distribution held-out reference: mean_dist = {np.mean(val_d):.2f} "
          f"(range {min(val_d):.2f}-{max(val_d):.2f}, n={len(val_d)})")

    tests = build_test_set()
    print(f"\n{len(tests)} test peptides:")
    rows = []
    for t in tests:
        emb = pooled_embeddings(model, [dict(seq=t["seq"])])[0]
        sc = knn_score((emb - mu) / sd, train_std, train_lab)
        leaked = t["seq"] in train_seqs
        row = dict(**t, **sc, in_training=leaked, length=len(t["seq"]))
        rows.append(row)
        flag = "  [IN TRAINING - excluded from claims]" if leaked else ""
        print(f"  {t['name']:20s} len={len(t['seq']):2d} exp={t['expected']:18s} "
              f"-> {sc['predicted_family']:9s} d={sc['mean_dist']:5.2f}{flag}")

    held_out = [r for r in rows if not r["in_training"]]
    print(f"\n{len(held_out)}/{len(rows)} candidates are genuinely held out.")

    # --- test 1: in-distribution?
    cand_d = [r["mean_dist"] for r in rows]
    ratio = np.mean(cand_d) / np.mean(val_d)
    print(f"\nTEST 1 (in-distribution): candidates sit {ratio:.2f}x the held-out "
          f"reference distance  [precursor-scale model: 3.3x]")

    # --- test 2: our designed control
    sysres = {r["name"]: r["mean_dist"] for r in rows if r["group"] == "systemin"}
    real = sysres["tomato_systemin"]; dis = sysres["systemin_disrupted"]
    print(f"TEST 2 (designed control): real systemin d={real:.2f}, disrupted d={dis:.2f} "
          f"-> {'PASS (disrupted is worse)' if dis > real else 'FAIL (disrupted looks better)'}"
          f"  [precursor-scale model: FAIL]")

    # --- test 3: literature ground truth, held-out only
    lit = [r for r in rows if r["group"] == "literature" and not r["in_training"]]
    print(f"\nTEST 3 (independent literature ground truth, {len(lit)} held-out peptides):")
    for r in sorted(lit, key=lambda x: x["mean_dist"]):
        print(f"   d={r['mean_dist']:5.2f}  {r['name']:20s} expected={r['expected']:10s} "
              f"pred={r['predicted_family']}")
    actives = [r["mean_dist"] for r in lit if r["expected"] == "ACTIVE"]
    inactives = [r["mean_dist"] for r in lit if r["expected"] in ("INACTIVE", "REDUCED", "PARTIAL")]
    if actives and inactives:
        print(f"   mean d: known-active {np.mean(actives):.2f} vs "
              f"known-reduced/inactive {np.mean(inactives):.2f} -> "
              f"{'separates correctly' if np.mean(actives) < np.mean(inactives) else 'DOES NOT separate'}")

    with open(f"{RESULTS}/mature_screening_test.json", "w") as f:
        json.dump(dict(candidates=rows, in_distribution_ref=val_d,
                       ratio_vs_reference=float(ratio)), f, indent=1)
    _release_memory()
    print("\nwrote mature_screening_test.json")


if __name__ == "__main__":
    main()
