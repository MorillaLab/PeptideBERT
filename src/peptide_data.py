"""
Data module for S2-PepAnalyst-BERT.

Honesty first: this is a *small* project, and the plant-signalling-peptide
literature simply doesn't offer millions of sequences the way ncRNA
databases do (NucleicBERT's MARS corpus has ~30M sequences; the whole
curated plant-SSP literature is a few thousand at best, per S2-PepAnalyst's
own benchmark sizes of 18 / 779 / 1,177 sequences, and Arabidopsis RALF -
the biggest single family - has only 39 members). So every sequence here
is one of exactly two kinds, and each is tagged as such in FAMILIES below:

  REAL      - fetched from UniProt/NCBI/TAIR, given
              verbatim with its accession. Two of these (RALF, CLE) are
              real, checked precursor sequences.
  TEMPLATE  - the *mature/conserved motif* is real and literature-sourced
              (citation in the comment), but the flanking scaffold
              (signal peptide + variable region) is a synthetic sequence
              built to match that family's known length/composition,
              NOT a specific database accession. Do not cite these as if
              they were a real gene.

Both REAL and TEMPLATE sequences are then used as seeds for controlled
in-silico mutagenesis (mutate_sequence) to build a pretraining corpus big
enough for a masked-language model to see repeated structure - this is the
synthetic-augmentation step, and it is the part of this pipeline you should
replace first once you have real curated data (PlantPepDB, an S2-PepAnalyst
training export, or your own RALF/CLE alignments). augment_family() and
build_corpus() are the two functions to swap out.
"""
import numpy as np

# ----------------------------------------------------------------------
# Tokenizer: character-level over the 20 canonical amino acids, mirroring
# NucleicBERT's "clump size 1" scheme (each residue is its own token) --
# appropriate here too, since SSPs are short and motif positions matter.
# ----------------------------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"]
VOCAB = SPECIAL + AMINO_ACIDS + ["X"]  # X = non-standard / ambiguous residue
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
PAD_ID, UNK_ID, MASK_ID, CLS_ID, SEP_ID = (TOK2ID[t] for t in SPECIAL)
VOCAB_SIZE = len(VOCAB)


def encode(seq, max_len):
    ids = [CLS_ID] + [TOK2ID.get(c, TOK2ID["X"]) for c in seq[: max_len - 2]] + [SEP_ID]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(PAD_ID)
        attn.append(0)
    return np.array(ids, dtype=np.int64), np.array(attn, dtype=np.float64)


def mask_tokens(ids, rng, mlm_prob=0.15):
    """BERT-style 80/10/10 masking. Never masks PAD/CLS/SEP.
    Returns (input_ids_with_masks, labels) where labels[i] = original id
    if position i was selected for the MLM loss, else -1 (ignored)."""
    ids = ids.copy()
    labels = np.full_like(ids, -1)
    maskable = np.where(~np.isin(ids, [PAD_ID, CLS_ID, SEP_ID]))[0]
    n_mask = max(1, int(round(mlm_prob * len(maskable))))
    chosen = rng.choice(maskable, size=min(n_mask, len(maskable)), replace=False)
    for pos in chosen:
        labels[pos] = ids[pos]
        r = rng.random()
        if r < 0.8:
            ids[pos] = MASK_ID
        elif r < 0.9:
            ids[pos] = TOK2ID[rng.choice(AMINO_ACIDS)]
        # else: leave unchanged (10% case)
    return ids, labels


# ----------------------------------------------------------------------
# Families. conserved = 0-indexed residue positions (into `seed`) that
# mutate_sequence() will never touch -- these are the positions the
# literature ties to structure/function (cysteines that disulfide-bond,
# the bioactive motif itself, the dibasic processing site).
# ----------------------------------------------------------------------

def _find(seed, sub):
    i = seed.find(sub)
    if i == -1:
        raise ValueError(f"motif {sub!r} not found in seed")
    return list(range(i, i + len(sub)))


_RALF_SEED = ("MDKSFTLFLTLTILVVFIISSPPVQAGFANDLGGVAWATTGDNGSGCHGSIAECIGAEEE"
              "EMDSEINRRILATTKYISYQSLKRNSVPCSRRGASYYNCQNGAQANPYSRGCSKIARCRS")
_CLE_SEED = ("MDSKSFLLLLLLFCFLFLHDASDLTQAHAHVQGLSNRKMMMMKMESEWVGANGEAEKAKT"
             "KGLGLHEELRTVPSGPDPLHHHVNPPRQPRNNFQLP")
# PSK: literature-consensus template. Real part = the sulfated pentapeptide
# YIYTQ and its immediate dibasic processing site, conserved across all six
# Arabidopsis AtPSK1-6 precursors (Yang et al. 2001, Plant Physiol 127:842;
# Matsubayashi & Sakagami, Srivastava et al. 2008 Plant J 56:219). Precursor
# length (~80 aa) and signal-peptide-like N-terminus match the reported
# architecture but the exact flanking residues are NOT a specific accession.
_PSK_SEED = ("MKHSILVTFLLVVLLASSMVMAVEARILAETTAEIPRSFLPKGLKPPSAKSSNRKGDQD"
             "GRKMKRRDGDDQVYIYTQNKQP")
# PEP1 (AtPep1 / PROPEP1). Mature 23-aa peptide widely reported in the
# literature (Huffaker, Pearce & Ryan 2006, PNAS 103:10098) reproduced here
# from that secondary literature, not re-verified against UniProt in this
# session -- check PROPEP1 (At5g64890) before relying on this beyond a demo.
# Notable because it has NO classical N-terminal signal peptide (released
# on cell damage rather than secreted) -- the "hard case" S2-PepAnalyst's
# own paper singles out.
_PEP1_SEED = ("MASSKVFVILLCVAILMDMAIAKEELIENQKPTKPKIVKKGGKQNKPRPKMEE"
              "ATKVKAKQRGKEKVSSGRPGQHN")

FAMILIES = {
    "RALF": dict(
        seed=_RALF_SEED,
        conserved=_find(_RALF_SEED, "TKYISYQ") + [i for i, c in enumerate(_RALF_SEED) if c == "C"],
        kind="REAL",
        source="UniProt Q9SRY3 / TAIR AT1G02900 (AtRALF1), 120 aa precursor",
        superclass="cysteine-rich",
    ),
    "CLE": dict(
        seed=_CLE_SEED,
        conserved=_find(_CLE_SEED, "RTVPSGPDPLHHH"),
        kind="REAL",
        source="UniProt Q9XF04 (AtCLV3/CLAVATA3), 96 aa precursor",
        superclass="PTM (Hyp-hydroxylation/arabinosylation)",
    ),
    "PSK": dict(
        seed=_PSK_SEED,
        conserved=_find(_PSK_SEED, "YIYTQ") + [_PSK_SEED.find("YIYTQ") - 2, _PSK_SEED.find("YIYTQ") - 1],
        kind="TEMPLATE",
        source="Motif consensus (Yang et al. 2001; Srivastava et al. 2008) around real YIYTQ pentapeptide",
        superclass="PTM (Tyr-sulfation)",
    ),
    "PEP": dict(
        seed=_PEP1_SEED,
        conserved=_find(_PEP1_SEED, "ATKVKAKQRGKEKVSSGRPGQHN"),
        kind="TEMPLATE",
        source="Mature peptide per Huffaker et al. 2006 PNAS (AtPep1/PROPEP1); scaffold illustrative",
        superclass="damage-associated (non-canonical secretion)",
    ),
}

CONSERVATIVE_GROUPS = [
    set("AVLIM"), set("FWY"), set("STNQ"), set("DE"), set("KRH"), set("GP"), set("C")
]
_GROUP_OF = {aa: g for g in CONSERVATIVE_GROUPS for aa in g}


def mutate_sequence(seed, conserved, rng, subst_rate=0.22, length_jitter_sd=16):
    """Point-substitute non-conserved residues (as before: ~70% same-group,
    biochemically conservative), PLUS -- new -- insert/delete residues at
    non-conserved positions so augmented sequences vary in *length*, not
    just identity.

    This fixes a real, documented limitation of the original substitution-
    only version: with every family member exactly the seed's length,
    sequence length alone was a near-perfect classification shortcut,
    undermining any "does the model learn more than trivial statistics"
    comparison. The fix samples a per-sequence length delta from
    Normal(0, length_jitter_sd) and applies exactly that many deletions or
    insertions at randomly chosen non-conserved positions.

    Because indels only ever act at *non-conserved* positions, a conserved
    residue is never deleted and nothing is ever inserted inside a
    contiguous conserved span (there is no non-conserved position between
    two residues of the same span to act on). Returns (new_sequence,
    new_conserved_positions) -- the latter re-indexed to track where each
    originally-conserved residue ends up after indels, since downstream
    code (attention-to-motif analysis) needs to find them per-sequence,
    not at the original seed's fixed indices.
    """
    conserved_set = set(conserved)
    non_conserved_positions = [i for i in range(len(seed)) if i not in conserved_set]
    seq = list(seed)
    tags = ["conserved" if i in conserved_set else "variable" for i in range(len(seed))]

    for i in non_conserved_positions:
        if rng.random() < subst_rate:
            aa = seq[i]
            if rng.random() < 0.7 and aa in _GROUP_OF:
                choices = [c for c in _GROUP_OF[aa] if c != aa]
                seq[i] = rng.choice(choices) if choices else aa
            else:
                seq[i] = rng.choice(AMINO_ACIDS)

    delta = int(round(rng.normal(0, length_jitter_sd)))
    max_del = len(non_conserved_positions) - 5  # always leave a few variable residues
    delta = max(-max_del, delta)

    if delta < 0:
        to_delete = set(rng.choice(non_conserved_positions, size=min(-delta, len(non_conserved_positions)), replace=False))
        seq = [aa for i, aa in enumerate(seq) if i not in to_delete]
        tags = [t for i, t in enumerate(tags) if i not in to_delete]
    elif delta > 0:
        insert_after = rng.choice(non_conserved_positions, size=delta, replace=True)
        counts = {}
        for p in insert_after:
            counts[p] = counts.get(p, 0) + 1
        new_seq, new_tags = [], []
        for i, (aa, t) in enumerate(zip(seq, tags)):
            new_seq.append(aa); new_tags.append(t)
            for _ in range(counts.get(i, 0)):
                new_seq.append(rng.choice(AMINO_ACIDS))
                new_tags.append("variable")
        seq, tags = new_seq, new_tags

    new_conserved = [i for i, t in enumerate(tags) if t == "conserved"]
    return "".join(seq), new_conserved


def make_decoy(length, rng):
    """Synthetic non-signalling background sequence: uniform random draw
    over the 20 amino acids, no signal peptide, no family motif. A
    deliberately naive negative control (documented as such) -- not a
    real protein fragment -- for the optional SSP-vs-decoy contrast."""
    return "".join(rng.choice(AMINO_ACIDS, size=length))


def build_corpus(n_per_family=60, n_decoys=60, seed=0):
    """Returns a list of dicts: {seq, family, kind, is_decoy, conserved}.
    `conserved` is the (per-sequence, indel-corrected) list of conserved
    residue positions -- [] for DECOY records, which have none.

    Decoy lengths are bootstrap-sampled from the *realised* post-indel
    length pool of the real families (not from the four fixed seed
    lengths), so DECOY shares the same empirical length distribution as
    the real classes by construction -- the strongest available length
    control, and the reason a length-matched subset (see
    embeddings_analysis.py) is meaningful at all."""
    rng = np.random.RandomState(seed)
    records = []
    for fam, spec in FAMILIES.items():
        records.append(dict(seq=spec["seed"], family=fam, kind=spec["kind"],
                             is_decoy=False, conserved=list(spec["conserved"])))
        for _ in range(n_per_family - 1):
            new_seq, new_conserved = mutate_sequence(spec["seed"], spec["conserved"], rng)
            records.append(dict(seq=new_seq, family=fam, kind="AUGMENTED",
                                 is_decoy=False, conserved=new_conserved))

    real_lengths = [len(r["seq"]) for r in records]
    for _ in range(n_decoys):
        L = int(rng.choice(real_lengths))
        records.append(dict(seq=make_decoy(L, rng), family="DECOY", kind="SYNTHETIC",
                             is_decoy=True, conserved=[]))
    rng.shuffle(records)
    return records


if __name__ == "__main__":
    recs = build_corpus()
    print(f"Total sequences: {len(recs)}")
    from collections import Counter
    print(Counter(r["family"] for r in recs))
    print(Counter(r["kind"] for r in recs))
    lens = [len(r["seq"]) for r in recs]
    print(f"Overall length range: {min(lens)}-{max(lens)}, mean {np.mean(lens):.1f}")
    print("\nPer-family length stats (this is what the length-matched control needs to see overlap in):")
    import numpy as _np
    for fam in list(FAMILIES) + ["DECOY"]:
        flens = [len(r["seq"]) for r in recs if r["family"] == fam]
        print(f"  {fam:6s} n={len(flens):3d}  min={min(flens):3d}  max={max(flens):3d}  "
              f"mean={_np.mean(flens):6.1f}  sd={_np.std(flens):5.1f}")
