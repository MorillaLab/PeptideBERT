"""
Downstream fine-tuning: family classification (RALF / CLE / PSK / PEP /
DECOY) from the pooled [CLS] embedding, under the same four-way regime
NucleicBERT's Table 1 uses -- {pretrained, random-init backbone} crossed
with {full fine-tuning, linear probe (backbone frozen)} -- which is exactly
what disentangles "how much came from pretraining" from "how much came from
task-specific gradient updates."

Run once per combo (keeps each call short/safe):
  python3 finetune.py pretrained finetune
  python3 finetune.py pretrained linear_probe
  python3 finetune.py random_init finetune
  python3 finetune.py random_init linear_probe

Appends into ../results/finetune_results.json (one entry per combo, keyed
by "{backbone}_{regime}"), each holding the per-epoch curve and final
val predictions (for a confusion-matrix figure).
"""
import ctypes
import gc
import json
import sys
import numpy as np
from model import PeptideBERT, Adam, MAX_LEN
from peptide_data import encode
from pretrain import load_model, _release_memory

RESULTS = "/home/morillalab/s2pepanalyst/results"
FAMILY_NAMES = ["RALF", "CLE", "PSK", "PEP", "DECOY"]
FAM2ID = {f: i for i, f in enumerate(FAMILY_NAMES)}
N_EPOCHS = 12
BATCH_SIZE = 16
LR_FINETUNE = 1.5e-3
LR_LINEAR_PROBE = 8e-3  # linear head alone can safely take a bigger step
SEED = 0


def load_split():
    with open(f"{RESULTS}/splits.json") as f:
        d = json.load(f)
    return d["records"], d["train_idx"], d["val_idx"]


def encode_batch_cls(records, idxs):
    ids_list, attn_list, y_list = [], [], []
    for i in idxs:
        ids, attn = encode(records[i]["seq"], MAX_LEN)
        ids_list.append(ids); attn_list.append(attn)
        y_list.append(FAM2ID[records[i]["family"]])
    return np.stack(ids_list), np.stack(attn_list), np.array(y_list, dtype=np.int64)


def run_classification(backbone, regime, records, train_idx, val_idx, seed=SEED):
    model = PeptideBERT(n_classes=len(FAMILY_NAMES), seed=seed)
    if backbone == "pretrained":
        load_model(model, f"{RESULTS}/pretrained_model.npz")
    elif backbone == "random_init":
        load_model(model, f"{RESULTS}/randominit_model.npz")
    else:
        raise ValueError(backbone)

    if regime == "finetune":
        params = model.all_params()
        lr = LR_FINETUNE
    elif regime == "linear_probe":
        params = [model.p["cls.W"], model.p["cls.b"]]
        lr = LR_LINEAR_PROBE
    else:
        raise ValueError(regime)
    opt = Adam(params, lr=lr)

    rng = np.random.RandomState(seed)
    curve = []
    for epoch in range(1, N_EPOCHS + 1):
        idxs = train_idx.copy(); rng.shuffle(idxs)
        tot_loss, tot_correct, n = 0.0, 0, 0
        for start in range(0, len(idxs), BATCH_SIZE):
            b = idxs[start:start + BATCH_SIZE]
            ids_b, attn_b, y_b = encode_batch_cls(records, b)
            hidden, _ = model.encode(ids_b, attn_b)
            logits = model.classify_logits(hidden)
            loss = logits.cross_entropy(y_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss.data) * len(b)
            tot_correct += float(loss.aux["correct"].sum())
            n += len(b)
            del loss, hidden, logits, ids_b, attn_b
        _release_memory()
        train_loss, train_acc = tot_loss / n, tot_correct / n

        ids_b, attn_b, y_b = encode_batch_cls(records, val_idx)
        hidden, _ = model.encode(ids_b, attn_b)
        logits = model.classify_logits(hidden)
        val_loss_t = logits.cross_entropy(y_b)
        val_loss = float(val_loss_t.data)
        val_acc = float(val_loss_t.aux["correct"].sum()) / len(val_idx)
        val_preds = val_loss_t.aux["preds"].tolist()
        del hidden, logits, val_loss_t, ids_b, attn_b
        _release_memory()

        curve.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                           val_loss=val_loss, val_acc=val_acc))
        print(f"[{backbone}/{regime}] epoch {epoch:2d}/{N_EPOCHS}  "
              f"train acc {train_acc:.3f}  val acc {val_acc:.3f}")

    return dict(backbone=backbone, regime=regime, curve=curve,
                final_val_acc=curve[-1]["val_acc"],
                val_true=[FAM2ID[records[i]["family"]] for i in val_idx],
                val_pred=val_preds, family_names=FAMILY_NAMES)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("pretrained", "random_init") \
            or sys.argv[2] not in ("finetune", "linear_probe"):
        print(__doc__)
        sys.exit(1)
    backbone, regime = sys.argv[1], sys.argv[2]
    records, train_idx, val_idx = load_split()

    out_path = f"{RESULTS}/finetune_results.json"
    try:
        with open(out_path) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    key = f"{backbone}_{regime}"
    result = run_classification(backbone, regime, records, train_idx, val_idx)
    all_results[key] = result
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nSaved {key}: final val acc = {result['final_val_acc']:.3f}")


if __name__ == "__main__":
    main()
