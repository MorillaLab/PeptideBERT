"""
Masked-language-model pretraining for PeptideBERT, plus the pretrained-vs-
random-init comparison that Fig. 2a of NucleicBERT is built around: does
single-sequence MLM pretraining on plant SSPs learn anything beyond what an
untrained transformer already captures via its random-feature statistics?

Outputs (into ../results/):
  pretrain_curve.json   - per-epoch train/val loss & accuracy
  pretrain_ppl_eval.json - per-sequence (accuracy, pseudo-perplexity, length,
                           family) for BOTH the pretrained and the
                           random-init model, computed on the held-out split
  pretrained_model.npz / randominit_model.npz - frozen weight snapshots
  splits.json            - which sequences went to train/val (for reuse by
                           finetune.py and embeddings_analysis.py)
"""
import ctypes
import gc
import json
import os
import numpy as np
from tensor import Tensor
from model import PeptideBERT, Adam, MAX_LEN
from peptide_data import build_corpus, encode, mask_tokens, PAD_ID, CLS_ID, SEP_ID, MASK_ID

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _release_memory():
    """Pure-refcounting should free every Tensor as soon as it's
    dereferenced (backward() explicitly breaks each node's links to its
    parents once used -- see tensor.py), but small numpy allocations
    still fragment glibc's heap arenas under the sheer allocation churn
    of ~100 tensors/forward-pass. gc.collect() plus malloc_trim keeps
    steady-state RSS flat instead of climbing over a long run."""
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(0)


RESULTS = "/home/morillalab/s2pepanalyst/results"
N_EPOCHS = 70
EPOCHS_PER_CALL = 8
BATCH_SIZE = 16
LR = 4e-3
SEED = 0


def save_model(model, path):
    np.savez(path, **{k: v.data for k, v in model.p.items()})


def load_model(model, path):
    d = np.load(path)
    for k in model.p:
        model.p[k].data = d[k].copy()
        model.p[k].zero_grad()


def stratified_split(records, val_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    by_fam = {}
    for i, r in enumerate(records):
        by_fam.setdefault(r["family"], []).append(i)
    train_idx, val_idx = [], []
    for fam, idxs in by_fam.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def encode_batch(records, idxs, rng):
    ids_list, attn_list, label_list = [], [], []
    for i in idxs:
        ids, attn = encode(records[i]["seq"], MAX_LEN)
        ids_m, labels = mask_tokens(ids, rng)
        ids_list.append(ids_m); attn_list.append(attn); label_list.append(labels)
    return np.stack(ids_list), np.stack(attn_list), np.stack(label_list)


def mlm_forward_loss(model, ids_b, attn_b, labels_b):
    hidden, _ = model.encode(ids_b, attn_b)
    B, T, D = hidden.shape
    flat_labels = labels_b.reshape(-1)
    mask = (flat_labels != -1).astype(np.float64)
    safe_targets = np.where(flat_labels == -1, 0, flat_labels)
    logits = model.mlm_logits(hidden)
    loss = logits.cross_entropy(safe_targets, mask=mask)
    acc = loss.aux["correct"].sum() / max(mask.sum(), 1.0)
    return loss, acc


def run_epoch(model, records, idxs, rng, optimizer=None, batch_size=BATCH_SIZE):
    idxs = idxs.copy()
    rng.shuffle(idxs)
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start:start + batch_size]
        ids_b, attn_b, labels_b = encode_batch(records, batch_idxs, rng)
        loss, acc = mlm_forward_loss(model, ids_b, attn_b, labels_b)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.data)
        total_acc += float(acc)
        n_batches += 1
        del loss, acc, ids_b, attn_b, labels_b
        if n_batches % 3 == 0:
            _release_memory()
    _release_memory()
    return total_loss / n_batches, total_acc / n_batches


def pseudo_perplexity_eval(model, records, idxs, max_positions=60):
    """Faithful to NucleicBERT eq. 2: for each sequence, mask each
    (non-special) position one at a time and record log p(true token |
    all others), batching all positions of one sequence together for
    speed. Returns per-sequence accuracy, pseudo-perplexity, length,
    family."""
    out = []
    for count, i in enumerate(idxs):
        seq = records[i]["seq"]
        base_ids, base_attn = encode(seq, MAX_LEN)
        valid_positions = [p for p in range(len(base_ids))
                            if base_ids[p] not in (PAD_ID, CLS_ID, SEP_ID)]
        valid_positions = valid_positions[:max_positions]
        n = len(valid_positions)
        ids_batch = np.tile(base_ids, (n, 1))
        true_tokens = np.zeros(n, dtype=np.int64)
        for row, pos in enumerate(valid_positions):
            true_tokens[row] = base_ids[pos]
            ids_batch[row, pos] = MASK_ID
        attn_batch = np.tile(base_attn, (n, 1))
        hidden, _ = model.encode(ids_batch, attn_batch)
        flat = hidden.reshape(n * MAX_LEN, model.d_model)
        gather_idx = np.array([row * MAX_LEN + pos for row, pos in enumerate(valid_positions)])
        logits = model.mlm_logits(hidden, positions=gather_idx)
        x = logits.data
        z = x - x.max(axis=-1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        logp_true = np.log(probs[np.arange(n), true_tokens] + 1e-8)
        preds = probs.argmax(axis=-1)
        acc = float((preds == true_tokens).mean())
        ppl = float(np.exp(-logp_true.mean()))
        out.append(dict(idx=int(i), family=records[i]["family"], length=len(seq),
                         accuracy=acc, pseudo_perplexity=ppl))
        del hidden, logits, flat
        if count % 5 == 0:
            _release_memory()
    _release_memory()
    return out


def main():
    rng = np.random.RandomState(SEED)
    records = build_corpus(n_per_family=60, n_decoys=60, seed=SEED)
    train_idx, val_idx = stratified_split(records, val_frac=0.2, seed=SEED)

    ckpt_path = f"{RESULTS}/_pretrain_checkpoint.npz"
    curve_path = f"{RESULTS}/pretrain_curve.json"
    start_epoch = 0
    curve = []
    model = PeptideBERT(n_classes=5, seed=SEED)
    opt_params = model.backbone_params() + [model.p["mlm.W1"], model.p["mlm.b1"],
                                             model.p["mlm.ln.g"], model.p["mlm.ln.b"],
                                             model.p["mlm.W2"], model.p["mlm.b2"]]
    opt = Adam(opt_params, lr=LR)

    if os.path.exists(ckpt_path):
        load_model(model, ckpt_path)
        with open(curve_path) as f:
            curve = json.load(f)
        start_epoch = curve[-1]["epoch"]
        opt_state = np.load(ckpt_path.replace(".npz", "_optstate.npz"))
        for p in opt_params:
            i = id(p)
            opt.m[i] = opt_state[f"m{opt_params.index(p)}"]
            opt.v[i] = opt_state[f"v{opt_params.index(p)}"]
        opt.t = int(opt_state["t"])
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    else:
        random_init_model = PeptideBERT(n_classes=5, seed=SEED)  # identical init, kept untouched
        save_model(random_init_model, f"{RESULTS}/randominit_model.npz")
        print(f"corpus: {len(records)} sequences  train={len(train_idx)} val={len(val_idx)}")

    target = min(start_epoch + EPOCHS_PER_CALL, N_EPOCHS)
    for epoch in range(start_epoch + 1, target + 1):
        tr_loss, tr_acc = run_epoch(model, records, train_idx, rng, optimizer=opt)
        va_loss, va_acc = run_epoch(model, records, val_idx, rng, optimizer=None)
        curve.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                           val_loss=va_loss, val_acc=va_acc))
        print(f"epoch {epoch:3d}/{N_EPOCHS}  train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

    save_model(model, ckpt_path)
    np.savez(ckpt_path.replace(".npz", "_optstate.npz"),
             t=opt.t, **{f"m{i}": opt.m[id(p)] for i, p in enumerate(opt_params)},
             **{f"v{i}": opt.v[id(p)] for i, p in enumerate(opt_params)})
    with open(curve_path, "w") as f:
        json.dump(curve, f, indent=1)

    if target < N_EPOCHS:
        print(f"\nCheckpoint saved at epoch {target}/{N_EPOCHS}. Re-run this script to continue.")
        return

    # finished all epochs: promote checkpoint to the final artifact and evaluate
    save_model(model, f"{RESULTS}/pretrained_model.npz")
    with open(f"{RESULTS}/splits.json", "w") as f:
        json.dump(dict(train_idx=train_idx, val_idx=val_idx,
                        records=[{k: v for k, v in r.items()} for r in records]), f, indent=1)

    print("\nComputing faithful pseudo-perplexity (masking each position individually)...")
    random_init_model = PeptideBERT(n_classes=5, seed=SEED)
    load_model(random_init_model, f"{RESULTS}/randominit_model.npz")
    ppl_pretrained = pseudo_perplexity_eval(model, records, val_idx)
    ppl_random = pseudo_perplexity_eval(random_init_model, records, val_idx)
    with open(f"{RESULTS}/pretrain_ppl_eval.json", "w") as f:
        json.dump(dict(pretrained=ppl_pretrained, random_init=ppl_random), f, indent=1)

    mean_acc_pre = np.mean([r["accuracy"] for r in ppl_pretrained])
    mean_acc_rnd = np.mean([r["accuracy"] for r in ppl_random])
    print(f"Held-out per-position accuracy: pretrained={mean_acc_pre:.3f}  random-init={mean_acc_rnd:.3f}")
    for f_ in (ckpt_path, ckpt_path.replace(".npz", "_optstate.npz")):
        if os.path.exists(f_):
            os.remove(f_)
    print("Done. Results written to", RESULTS)


if __name__ == "__main__":
    main()
