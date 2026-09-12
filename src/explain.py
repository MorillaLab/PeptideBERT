"""
Explainability: saliency mapping and attention analysis, the peptide
analogues of NucleicBERT Fig. 1g-h / Fig. 4a-f.

Saliency: gradient of the masked-token loss w.r.t. each position's input
embedding (L2 norm per position) -- "gradients all the way up to the input
tokens", same recipe as the source paper's Fig. 1g/4a. If pretraining
worked the way it's supposed to, saliency should spike at [MASK] positions.

Attention-to-motif: for each (layer, head), the average attention weight
landing on the family's conserved/functional positions (the same
`conserved` index sets used to protect motifs during data augmentation --
see peptide_data.FAMILIES), averaged over a sample of real sequences.
Analogous to NucleicBERT's per-(layer,head) "average attention to motifs"
heatmap (their Fig. 1h / 4d-f), but here the "motif" is a literal,
known ground truth (the RALF YISY box, the CLE dodecapeptide, etc.)
rather than an inferred structural feature -- arguably a cleaner test bed.
"""
import json
import numpy as np
from tensor import Tensor
from model import PeptideBERT, MAX_LEN
from peptide_data import encode, FAMILIES, MASK_ID, CLS_ID, SEP_ID, PAD_ID
from pretrain import load_model, _release_memory

RESULTS = "/home/morillalab/s2pepanalyst/results"
SEED = 0


def saliency_for_sequence(model, seq, mask_positions):
    """Masks the given 0-indexed *sequence* positions (offset by +1 for the
    leading [CLS] when addressing the token array), runs the MLM loss on
    exactly those positions, and returns per-position saliency = ||d
    loss / d input_embedding||_2, plus the token ids actually fed in."""
    ids, attn = encode(seq, MAX_LEN)
    ids = ids.copy()
    true_tokens = ids.copy()
    tok_positions = [p + 1 for p in mask_positions]  # +1 for [CLS]
    for p in tok_positions:
        ids[p] = MASK_ID

    tok_emb_np = model.p["tok_emb"].data[ids][None, :, :]  # (1,T,D)
    tok_emb_var = Tensor(tok_emb_np, requires_grad=True)
    T = len(ids)
    pos_emb = model.p["pos_emb"][np.arange(T)].reshape(1, T, model.d_model)
    x = tok_emb_var + pos_emb

    add = (1.0 - attn[None, :]) * -1e9
    mask_add = Tensor(add.reshape(1, 1, 1, T), requires_grad=False)
    for l in range(model.n_layers):
        x, _ = model._attn_block(x, mask_add, l)
        x = model._ff_block(x, l)
    x = x.layer_norm(model.p["ln_f.g"], model.p["ln_f.b"])

    gather_idx = np.array(tok_positions)
    logits = model.mlm_logits(x, positions=gather_idx)
    targets = true_tokens[gather_idx]
    loss = logits.cross_entropy(targets)
    loss.backward()

    saliency = np.linalg.norm(tok_emb_var.grad[0], axis=-1)  # (T,)
    return saliency, ids, true_tokens


def attention_to_motifs(model, records_by_family, n_samples=15, seed=SEED):
    """(n_layers, n_heads) matrix: average attention mass landing on each
    family's conserved positions, averaged over queries, positions, and a
    sample of sequences per family.

    Uses each record's OWN `conserved` list (not the fixed seed-based
    FAMILIES[fam]["conserved"]), because indel augmentation shifts where
    a family's conserved motif sits within any given augmented sequence --
    using the seed's fixed indices here would silently look at the wrong
    positions for most sequences."""
    rng = np.random.RandomState(seed)
    acc = np.zeros((model.n_layers, model.n_heads))
    count = 0
    for fam, recs in records_by_family.items():
        if fam == "DECOY" or not recs:
            continue
        sample_idx = rng.choice(len(recs), size=min(n_samples, len(recs)), replace=False)
        for si in sample_idx:
            r = recs[si]
            conserved_tok_pos = np.array([p + 1 for p in r.get("conserved", [])])  # +1 for CLS
            if len(conserved_tok_pos) == 0:
                continue
            ids, attn_mask = encode(r["seq"], MAX_LEN)
            ids_b = ids[None, :]; attn_b = attn_mask[None, :]
            hidden, attns = model.encode(ids_b, attn_b, want_attn=True)
            for l, a in enumerate(attns):
                # a: (1,H,T,T) softmax attention. Average over query positions
                # of the mass each head places on the conserved key positions.
                w = a.data[0]  # (H,T,T)
                mass_to_motif = w[:, :, conserved_tok_pos].sum(axis=-1)  # (H,T)
                acc[l] += mass_to_motif.mean(axis=-1)  # (H,)
            count += 1
            del hidden, attns
    _release_memory()
    return acc / max(count, 1)


def main():
    with open(f"{RESULTS}/splits.json") as f:
        records = json.load(f)["records"]
    records_by_family = {}
    for r in records:
        records_by_family.setdefault(r["family"], []).append(r)

    model = PeptideBERT(seed=SEED)
    load_model(model, f"{RESULTS}/pretrained_model.npz")
    random_model = PeptideBERT(seed=SEED)
    load_model(random_model, f"{RESULTS}/randominit_model.npz")

    # --- saliency, on one representative RALF sequence (has both PTM-free
    # cysteine motif AND the YISY-like functional motif to look for spikes at)
    example = records_by_family["RALF"][0]["seq"]
    rng = np.random.RandomState(1)
    n = len(example)
    mask_positions = sorted(rng.choice(n, size=max(3, n // 10), replace=False).tolist())
    saliency, ids, true_tokens = saliency_for_sequence(model, example, mask_positions)

    print("Computing attention-to-motif heatmap (pretrained)...")
    attn_map_pre = attention_to_motifs(model, records_by_family)
    print("Computing attention-to-motif heatmap (random-init)...")
    attn_map_rnd = attention_to_motifs(random_model, records_by_family)

    out = dict(
        saliency=dict(sequence=example, mask_positions=mask_positions,
                       saliency_scores=saliency.tolist()),
        attention_to_motifs_pretrained=attn_map_pre.tolist(),
        attention_to_motifs_random=attn_map_rnd.tolist(),
        n_layers=model.n_layers, n_heads=model.n_heads,
    )
    with open(f"{RESULTS}/explain_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Done. Wrote explain_results.json")
    print(f"Mean attention-to-motif: pretrained={attn_map_pre.mean():.4f}  random={attn_map_rnd.mean():.4f}")


if __name__ == "__main__":
    main()
