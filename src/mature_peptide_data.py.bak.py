"""
Mature-peptide corpus: the Option (a) rebuild.

The main project's corpus is PRECURSOR-length (33-155 aa). The tomato
screening case study (screen_tomato_candidates.py) showed that a model
trained on that corpus cannot triage 18-aa mature peptides -- candidates
landed 3.3x further from the training manifold than in-distribution
sequences, all collapsed to one predicted family, and a deliberately
disrupted negative control ranked BEST. That is an out-of-distribution
failure, not a subtle one.

This module rebuilds the corpus at the scale screening actually operates
on: MATURE peptides, 10-25 residues, the things you would actually
synthesise and spray.

EVERY seed here is a real, literature- or vendor-verified mature peptide
sequence, verified during this session -- there are no TEMPLATE scaffolds
in this corpus (unlike peptide_data.py, where PSK and PEP were
motif-templated). Provenance per seed is in the `source` field.

A genuine advantage of working at mature-peptide scale: the published
structure-activity literature gives real activity labels for specific
truncations and point mutants, so the screening test has ground truth
that does not come from us. Two are used below:
  - AtPep1(9-23)  : retains full activity (Lin et al. 2024, IJMS)
  - AtPep1(13-23) : retains ROS burst, loses root-growth inhibition (ibid)
  - GrCLE1-1m     : Gly->Ala at position 6, the clv3-1/clv3-5-equivalent
                    knockout substitution (US 8,569,578; Fiers et al. 2005)
  - Ag-16p        : a published NEGATIVE CONTROL peptide (Fiers et al.
                    2005), i.e. a real 16-mer known to be inactive in the
                    CLE root assay -- exactly what a screening tool should
                    rank poorly.
"""
import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"]
VOCAB = SPECIAL + AMINO_ACIDS + ["X"]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
PAD_ID, UNK_ID, MASK_ID, CLS_ID, SEP_ID = (TOK2ID[t] for t in SPECIAL)
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 32  # mature peptides are 10-25 aa; 32 tokens covers +[CLS]/[SEP]


def encode(seq, max_len=MAX_LEN):
    ids = [CLS_ID] + [TOK2ID.get(c, TOK2ID["X"]) for c in seq[: max_len - 2]] + [SEP_ID]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(PAD_ID); attn.append(0)
    return np.array(ids, dtype=np.int64), np.array(attn, dtype=np.float64)


def mask_tokens(ids, rng, mlm_prob=0.15):
    ids = ids.copy()
    labels = np.full_like(ids, -1)
    maskable = np.where(~np.isin(ids, [PAD_ID, CLS_ID, SEP_ID]))[0]
    if len(maskable) == 0:
        return ids, labels
    n_mask = max(1, int(round(mlm_prob * len(maskable))))
    chosen = rng.choice(maskable, size=min(n_mask, len(maskable)), replace=False)
    for pos in chosen:
        labels[pos] = ids[pos]
        r = rng.random()
        if r < 0.8:
            ids[pos] = MASK_ID
        elif r < 0.9:
            ids[pos] = TOK2ID[rng.choice(AMINO_ACIDS)]
    return ids, labels


# ----------------------------------------------------------------------
# Real mature-peptide seeds. All verified this session.
# `conserved` = 0-indexed positions held fixed during augmentation,
# chosen from the published motif definition for each family.
# ----------------------------------------------------------------------
MATURE_FAMILIES = {
    "CLE": dict(
        seeds=[
            ("CLV3",       "RTVPSGPDPLHH", "Arabidopsis CLV3 mature dodecapeptide (Kondo et al. 2006; vendor-confirmed for UniProt Q9XF04)"),
            ("TDIF_CLE41", "HEVPSGPNPISN", "TDIF / CLE41-CLE44 dodecapeptide (Ito et al. 2006 Science 313:842; MeSH C532424)"),
            ("GrCLE1-1",   "RVIPGGPDPLHN", "Globodera rostochiensis CLE1-1 12-mer (US 8,569,578; Fiers et al. 2005)"),
            ("GrCLE4-1",   "RVAGAGPDPIHH", "G. rostochiensis CLE4-1 12-mer (US 8,569,578)"),
            ("GrCLE4-2",   "RAVPAGPDPKHH", "G. rostochiensis CLE4-2 12-mer (US 8,569,578)"),
            ("GrCLE4-3",   "RGAPAGPDPIHH", "G. rostochiensis CLE4-3 12-mer (US 8,569,578)"),
        ],
        # CLE motif: the Pro/Gly-rich core "G..DP" and flanking positions
        # are what define the family; positions 4-10 are the most conserved
        # across all six real seeds above.
        conserved=list(range(4, 11)),
    ),
    "PEP": dict(
        seeds=[
            ("AtPep1",      "ATKVKAKQRGKEKVSSGRPGQHN", "Arabidopsis AtPep1, 23 aa (Huffaker et al. 2006 PNAS 103:10098; multiple vendor datasheets)"),
            ("AtPep1_9_23", "RGKEKVSSGRPGQHN", "AtPep1(9-23) truncation, retains full activity (Lin et al. 2024 IJMS)"),
        ],
        # C-terminal region is the activity-critical part per the
        # truncation series; hold the last 11 fixed.
        conserved=list(range(12, 23)),
    ),
    "SYSTEMIN": dict(
        seeds=[
            ("tomato_systemin", "AVQSKPPSKRDPPKMQTD", "Solanum lycopersicum systemin, UniProt P27058 res. 179-196 (Pearce et al. 1991 Science 253:895)"),
        ],
        # conserved prolines + invariant C-terminal 7 (Constabel et al. 1998)
        conserved=[5, 6] + list(range(11, 18)),
    ),
}

CONSERVATIVE_GROUPS = [set("AVLIM"), set("FWY"), set("STNQ"), set("DE"), set("KRH"), set("GP"), set("C")]
_GROUP_OF = {aa: g for g in CONSERVATIVE_GROUPS for aa in g}


def mutate_mature(seq, conserved, rng, subst_rate=0.18, max_indel=2):
    """Conservative substitution at non-conserved positions + small
    length jitter (+/- up to `max_indel` residues). Length jitter is kept
    much smaller than in the precursor corpus because mature peptides are
    short and real family length variation is genuinely narrow (CLE
    12-13 aa, PEP 15-23 aa) -- the point is to avoid length becoming a
    free class label, not to invent unrealistic length spread."""
    conserved = set(i for i in conserved if i < len(seq))
    s = list(seq)
    free = [i for i in range(len(s)) if i not in conserved]
    for i in free:
        if rng.random() < subst_rate:
            aa = s[i]
            if rng.random() < 0.7 and aa in _GROUP_OF:
                ch = [c for c in _GROUP_OF[aa] if c != aa]
                s[i] = rng.choice(ch) if ch else aa
            else:
                s[i] = rng.choice(AMINO_ACIDS)
    delta = rng.randint(-max_indel, max_indel + 1)
    if delta < 0 and len(free) > abs(delta) + 2:
        drop = set(rng.choice(free, size=min(-delta, len(free) - 2), replace=False))
        s = [c for i, c in enumerate(s) if i not in drop]
    elif delta > 0:
        for _ in range(delta):
            pos = int(rng.choice(free)) if free else 0
            s.insert(pos, rng.choice(AMINO_ACIDS))
    return "".join(s)


def build_mature_corpus(n_per_family=80, n_decoys=80, seed=0):
    rng = np.random.RandomState(seed)
    records = []
    for fam, spec in MATURE_FAMILIES.items():
        seeds = spec["seeds"]
        records.extend(dict(seq=s, family=fam, kind="REAL", name=nm, source=src)
                       for nm, s, src in seeds)
        n_aug = n_per_family - len(seeds)
        for k in range(n_aug):
            nm, s, _ = seeds[k % len(seeds)]
            records.append(dict(seq=mutate_mature(s, spec["conserved"], rng),
                                family=fam, kind="AUGMENTED", name=f"{nm}_aug{k}", source="augmented"))
    real_lengths = [len(r["seq"]) for r in records]
    for i in range(n_decoys):
        L = int(rng.choice(real_lengths))
        records.append(dict(seq="".join(rng.choice(AMINO_ACIDS, size=L)),
                            family="DECOY", kind="SYNTHETIC", name=f"decoy{i}",
                            source="synthetic i.i.d. uniform over 20 aa"))
    rng.shuffle(records)
    return records


if __name__ == "__main__":
    recs = build_mature_corpus()
    from collections import Counter
    print(f"total: {len(recs)}")
    print(Counter(r["family"] for r in recs))
    print(Counter(r["kind"] for r in recs))
    print("\nper-family length (length must NOT be a free class label):")
    for fam in list(MATURE_FAMILIES) + ["DECOY"]:
        L = [len(r["seq"]) for r in recs if r["family"] == fam]
        print(f"  {fam:9s} n={len(L):3d}  {min(L)}-{max(L)} aa  mean={np.mean(L):.1f}")
    n_real = sum(1 for r in recs if r["kind"] == "REAL")
    print(f"\nREAL verified mature-peptide seeds in corpus: {n_real}")
