"""
Case study: can PeptideBERT's embedding space be used to triage candidate
peptide variants before synthesising them for a real translational
question -- growth-regulator / biostimulant screening for tomato?

Tomato systemin is the obvious test case: it is the single most
extensively characterised real Solanaceae SSP with an actual published
cross-species structure-activity literature to reason from (Pearce et al.
1991 Science 253:895; McGurl et al. 1992 Science 255:1570; Constabel, Yip
& Ryan 1998 Plant Mol Biol 36:55-62), and -- unlike PSK/PEP in the main
corpus -- its mature sequence is fully known and verified (UniProt P27058;
independently cross-confirmed against five peptide-vendor product sheets
during this session), not a motif-templated scaffold.

READ THIS BEFORE THE RESULTS: mature systemin is 18 residues. Every
sequence PeptideBERT was pretrained and fine-tuned on falls in 33-155
residues (mean ~92; see peptide_data.py). That is a severe, out-of-band
distribution shift -- not a minor caveat. This script is a stress-test of
whether the embedding space degrades gracefully under that shift, not a
demonstration of a working short-peptide screening tool. Results are
reported and figured exactly as computed, including if they look bad.

Candidate provenance (same REAL/TEMPLATE/DESIGNED discipline as
peptide_data.py -- see FAMILIES there):
  REAL      tomato systemin itself (Pearce et al. 1991), used verbatim.
  DESIGNED  variants built under a literature-derived conservation
            constraint (Constabel et al. 1998, and the summary in
            Walker & Constabel-adjacent reviews of the family): natural
            Solanaceae systemin homologues differ from tomato by only
            2-3 substitutions, never in the C-terminal 7 residues
            (...PPKMQTD), and never at any of the four conserved
            prolines. These variants respect that constraint -- they are
            NOT reconstructions of the real potato/pepper/nightshade
            sequences (which were not available to verify verbatim this
            session) and must not be cited as such. A separate
            "disrupted" variant deliberately *violates* the constraint
            (mutates a conserved proline and part of the invariant
            C-terminus), as an internal negative control: literature
            already tells us this would very likely abolish activity in
            planta (a prosystemin lacking the systemin region entirely is
            known to be inactive -- Dombrowski et al. 1999), so a
            screening tool worth trusting should at least not rank it
            *more* promising than the conservative variants.
"""
import json
import numpy as np
from model import PeptideBERT, MAX_LEN
from peptide_data import encode, AMINO_ACIDS, PAD_ID, CLS_ID, SEP_ID
from pretrain import load_model, _release_memory
from embeddings_analysis import model_embeddings, load_records

RESULTS = "/home/morillalab/s2pepanalyst/results"
FAMILY_NAMES = ["RALF", "CLE", "PSK", "PEP", "DECOY"]

TOMATO_SYSTEMIN = "AVQSKPPSKRDPPKMQTD"  # UniProt P27058, residues 179-196 of prosystemin
FREE_POSITIONS = [0, 1, 2, 3, 4, 7, 8, 9, 10]  # literature-conserved elsewhere (Methods docstring)
CONSERVED_PROLINES = [5, 6, 11, 12]
INVARIANT_CTERM = list(range(11, 18))  # "PPKMQTD"


def make_designed_variant(rng, n_subs=2, disrupt=False):
    seq = list(TOMATO_SYSTEMIN)
    if disrupt:
        # deliberately violate the conservation constraint: mutate a
        # conserved proline and scramble part of the invariant C-terminus
        seq[6] = "G"
        seq[14], seq[15] = "A", "A"
        return "".join(seq)
    positions = rng.choice(FREE_POSITIONS, size=min(n_subs, len(FREE_POSITIONS)), replace=False)
    for p in positions:
        seq[p] = rng.choice(AMINO_ACIDS)
    return "".join(seq)


def build_candidates(seed=0):
    rng = np.random.RandomState(seed)
    candidates = [
        dict(name="tomato_systemin", seq=TOMATO_SYSTEMIN, provenance="REAL",
             note="UniProt P27058, residues 179-196 of prosystemin"),
    ]
    for i in range(4):
        candidates.append(dict(
            name=f"designed_variant_{chr(65+i)}", seq=make_designed_variant(rng, n_subs=rng.randint(2, 4)),
            provenance="DESIGNED",
            note="2-3 substitutions at literature-free positions; C-term 7 + prolines held fixed"))
    candidates.append(dict(
        name="designed_disrupted_control", seq=make_designed_variant(rng, disrupt=True),
        provenance="DESIGNED (negative control)",
        note="deliberately mutates a conserved proline + invariant C-terminus"))
    return candidates


def knn_predict(query_embedding, train_embeddings, train_labels, k=5):
    d = np.linalg.norm(train_embeddings - query_embedding[None, :], axis=1)
    nn_idx = np.argsort(d)[:k]
    nn_labels = [train_labels[i] for i in nn_idx]
    nn_dists = d[nn_idx]
    from collections import Counter
    vote = Counter(nn_labels).most_common(1)[0]
    return dict(predicted_family=vote[0], vote_count=vote[1], k=k,
                neighbour_families=nn_labels, neighbour_distances=nn_dists.tolist(),
                mean_dist_to_neighbours=float(nn_dists.mean()))


def main():
    records = load_records()
    train_labels = [r["family"] for r in records]

    model = PeptideBERT(seed=0)
    load_model(model, f"{RESULTS}/pretrained_model.npz")

    print("Computing training-corpus embeddings (for kNN reference)...")
    train_embeddings = model_embeddings(model, records)
    mu, sd = train_embeddings.mean(axis=0), train_embeddings.std(axis=0) + 1e-8

    candidates = build_candidates(seed=0)
    print(f"\n{len(candidates)} candidates:")
    for c in candidates:
        print(f"  {c['name']:28s} [{c['provenance']}]  {c['seq']}")

    print("\nScoring candidates against the pretrained embedding space (kNN, k=5)...")
    out = []
    for c in candidates:
        emb = model_embeddings(model, [dict(seq=c["seq"])])[0]
        emb_std = (emb - mu) / sd
        train_std = (train_embeddings - mu) / sd
        pred = knn_predict(emb_std, train_std, train_labels, k=5)
        row = dict(**c, **pred)
        out.append(row)
        print(f"  {c['name']:28s} -> {pred['predicted_family']:6s} "
              f"({pred['vote_count']}/5 neighbours)  mean_dist={pred['mean_dist_to_neighbours']:.2f}  "
              f"neighbours={pred['neighbour_families']}")

    # sanity control: run the same pipeline on a genuine in-distribution
    # held-out validation sequence per family, so the tomato-systemin
    # results have a same-script baseline to be read against
    with open(f"{RESULTS}/splits.json") as f:
        val_idx = json.load(f)["val_idx"]
    print("\nSanity control: one in-distribution validation sequence per family...")
    controls = []
    seen_fam = set()
    for i in val_idx:
        fam = records[i]["family"]
        if fam in seen_fam:
            continue
        seen_fam.add(fam)
        emb = model_embeddings(model, [records[i]])[0]
        emb_std = (emb - mu) / sd
        train_std = (train_embeddings - mu) / sd
        pred = knn_predict(emb_std, train_std, train_labels, k=5)
        controls.append(dict(name=f"control_{fam}", true_family=fam, seq_length=len(records[i]["seq"]), **pred))
        print(f"  control_{fam:6s} (len={len(records[i]['seq']):3d}) -> {pred['predicted_family']:6s} "
              f"({pred['vote_count']}/5)  mean_dist={pred['mean_dist_to_neighbours']:.2f}")

    with open(f"{RESULTS}/tomato_screening_case_study.json", "w") as f:
        json.dump(dict(candidates=out, in_distribution_controls=controls), f, indent=1)
    _release_memory()
    print("\nWrote tomato_screening_case_study.json")


if __name__ == "__main__":
    main()
