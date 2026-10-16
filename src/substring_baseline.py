"""
Exact-substring baseline: a zero-parameter control for corpus leakage.

For each family, searches for any substring that appears in >= `coverage` of
its members and in NO member of any other family. If such motifs exist, the
family label is recoverable by exact matching and no accuracy measured on
that corpus supports a claim about representation quality.

This is worth REPORTING, not just running privately. On the synthetic corpus
this study originally used, a five-line classifier built from the motifs
below scored 100% -- above every learned method benchmarked, including a
protein language model. On the real UniProt corpus it finds nothing except
genuine biological conservation. A baseline that catches your own earlier
mistake belongs in the paper.

    python3 src/substring_baseline.py
    python3 src/substring_baseline.py --json results/substring_baseline.json
"""
import argparse
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NEGATIVE_CANDIDATES = ("NONSSP", "DECOY", "NEGATIVE")


def discriminating_substrings(records, min_len=5, max_len=30, coverage=0.90):
    """Any motif shared by a whole family must be a substring of its
    shortest member, so only that one needs scanning. Longest first, so the
    most specific motif is reported."""
    fams = sorted({r["family"] for r in records})
    by = {f: [r["seq"] for r in records if r["family"] == f] for f in fams}
    out = {}
    for f in fams:
        own = by[f]
        other = [s for g in fams if g != f for s in by[g]]
        shortest = min(own, key=len)
        found = None
        for L in range(min(max_len, len(shortest)), min_len - 1, -1):
            for i in range(len(shortest) - L + 1):
                sub = shortest[i:i + L]
                hit = sum(sub in s for s in own) / len(own)
                if hit >= coverage and not any(sub in s for s in other):
                    found = dict(motif=sub, length=L, coverage=hit)
                    break
            if found:
                break
        out[f] = found
    return out


def classify_with(records, motifs):
    """The five-line classifier. Zero parameters, no training."""
    neg = next((f for f in motifs if f in NEGATIVE_CANDIDATES), None)
    usable = {f: m["motif"] for f, m in motifs.items() if m and f != neg}
    correct = 0
    for r in records:
        hits = [f for f, m in usable.items() if m in r["seq"]]
        pred = hits[0] if len(hits) == 1 else (neg if not hits else hits[0])
        correct += (pred == r["family"])
    return correct / len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=os.path.join(RESULTS, "splits.json"))
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--json", default=None, help="also write results here")
    a = ap.parse_args()

    records = json.load(open(a.splits))["records"]
    fams = sorted({r["family"] for r in records})
    neg = next((f for f in fams if f in NEGATIVE_CANDIDATES), None)

    print(f"corpus: {len(records)} sequences, {len(fams)} classes, "
          f"chance = {1/len(fams):.3f}")
    print(f"searching for substrings present in >={a.coverage:.0%} of a family "
          f"and 0% of every other\n")

    motifs = discriminating_substrings(records, coverage=a.coverage)
    leaky = []
    for f in fams:
        m = motifs[f]
        if m:
            print(f"  {f:10s} {m['motif']!r}  ({m['length']} aa, "
                  f"{m['coverage']*100:.0f}% of family, 0% elsewhere)")
            if f != neg:
                leaky.append(f)
        else:
            print(f"  {f:10s} none")

    acc = classify_with(records, motifs)
    n_pos = len([f for f in fams if f != neg])
    print(f"\n  families with an exact discriminator: {len(leaky)}/{n_pos}")
    print(f"  zero-parameter substring classifier accuracy: {acc:.3f}")

    verdict = "PASS"
    if len(leaky) >= 2:
        verdict = "FAIL"
        print("\n  FAIL -- the family label is substantially recoverable by exact")
        print("          matching. Any accuracy measured on this corpus is not")
        print("          evidence about representation quality.")
    elif leaky:
        print(f"\n  PASS with note -- {leaky[0]} carries an invariant motif. Check")
        print("          whether this is real biological conservation (report it,")
        print("          it is a positive control) or an artefact of how the")
        print("          corpus was built (fix it).")
    else:
        print("\n  PASS -- no family is recoverable by exact substring matching.")

    if a.json:
        json.dump(dict(n=len(records), classes=fams, chance=1 / len(fams),
                       coverage_threshold=a.coverage, motifs=motifs,
                       substring_classifier_accuracy=acc,
                       leaky_families=leaky, verdict=verdict),
                  open(a.json, "w"), indent=1)
        print(f"\n  wrote {a.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
