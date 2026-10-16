"""Single source of truth for the corpus.

Reads results/splits.json -- built by build_real_corpus.py -- instead of
regenerating sequences. Family names, class count and cluster assignments
all come from the data, so nothing downstream hardcodes the shape of a
particular corpus.
"""
import json
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NEGATIVE_CANDIDATES = ("NONSSP", "DECOY", "NEGATIVE")


def load():
    with open(os.path.join(RESULTS, "splits.json")) as f:
        return json.load(f)


def records():
    return load()["records"]


def family_names(recs=None):
    """Positive families alphabetically, negative class last, so class ids
    are stable across runs and the negative is always the final index."""
    recs = recs if recs is not None else records()
    fams = sorted({r["family"] for r in recs})
    neg = [f for f in fams if f in NEGATIVE_CANDIDATES]
    return [f for f in fams if f not in neg] + neg


def negative_name(recs=None):
    for f in family_names(recs):
        if f in NEGATIVE_CANDIDATES:
            return f
    return None


def fam2id(recs=None):
    return {f: i for i, f in enumerate(family_names(recs))}


def labels(recs=None):
    recs = recs if recs is not None else records()
    m = fam2id(recs)
    return np.array([m[r["family"]] for r in recs])


def clusters(d=None):
    """Per-sequence cluster id, used to group cross-validation folds. Falls
    back to one cluster per sequence if the corpus predates clustering --
    which reproduces the old leaky behaviour, so it warns."""
    d = d if d is not None else load()
    if "clusters" in d:
        return np.array(d["clusters"])
    print("WARNING: splits.json has no 'clusters'; cross-validation folds "
          "will not be homology-aware. Rebuild with build_real_corpus.py.")
    return np.arange(len(d["records"]))


def splits(d=None):
    d = d if d is not None else load()
    return d["train_idx"], d["val_idx"]
