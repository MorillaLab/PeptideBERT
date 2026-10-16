"""
Build a REAL plant small-signalling-peptide corpus from UniProtKB, replacing
the synthetic augmentation in peptide_data.py.

WHY: the synthetic corpus holds each family's conserved motif perfectly
invariant across all members, so family identity is a deterministic
function of an exact substring and a zero-parameter `grep` scores 100% --
above every learned method benchmarked. No accuracy measured on that
corpus can support a claim about representation quality. This script
replaces it with real sequences and REFUSES TO WRITE a corpus that fails
the same check.

WHAT IT DOES
  1. Fetches reviewed plant sequences per family from the UniProt REST API.
  2. Reduces redundancy (near-duplicate homologues collapsed to clusters).
  3. Splits BY CLUSTER, not by sequence, so homologues cannot straddle the
     train/test boundary -- the standard requirement in protein ML and the
     first thing a reviewer will check.
  4. Builds a hard negative class from real, length-matched plant proteins
     that are not in any SSP family (not uniform-random strings).
  5. Takes `conserved` positions from UniProt's curated PEPTIDE/PROPEP
     feature annotations rather than from planted motifs, so the
     attention-to-motif analysis is referenced against external evidence.
  6. Runs QC gates and aborts on failure.
  7. Writes results/splits.json in the schema the existing pipeline reads.

RUN IT WHERE THERE IS NETWORK:
    pip install requests numpy
    python3 build_real_corpus.py --out results/splits.json

The fetch step is the only part that needs the internet; everything after
is local. Expect a few minutes.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

BASE = "https://rest.uniprot.org/uniprotkb/stream"
BASE_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
VIRIDIPLANTAE = 33090
FIELDS = ["accession", "id", "protein_name", "gene_names", "organism_name",
          "organism_id", "length", "sequence", "ft_peptide", "ft_propep",
          "ft_signal", "reviewed"]

# Client-side exclusion for the negative class. Belt and braces: we no longer
# push NOT clauses to the server (that is what caused the full table scan), so
# anything SSP-looking is dropped here instead.
SSP_NAME_RE = re.compile(
    r"rapid alkalinization|RALF|phytosulfokine|CLAVATA|\bCLE\b|ESR-related|"
    r"GOLVEN|root meristem growth factor|CLEL|elicitor peptide|PROPEP|"
    r"C-terminally encoded peptide|inflorescence deficient in abscission|"
    r"IDA-like|systemin|\bPIP\b|\bPSY\b|\bCIF\b|stomagen|\bEPF", re.I)

# ---------------------------------------------------------------------------
# FAMILY DEFINITIONS
#
# Pfam accessions marked VERIFIED were checked against the literature.
# The others are name-based because these families have poor or absent Pfam
# coverage (CLE in particular is a ~14-residue C-terminal motif that Pfam
# does not model as a domain). Name-based queries are less precise, so the
# script PRINTS what each query returned -- inspect that output before
# trusting the corpus. Adjust freely; this table is the one place to edit.
# ---------------------------------------------------------------------------
FAMILIES = {
    "RALF": dict(
        query='(xref:pfam-PF05498 OR protein_name:"rapid alkalinization factor" OR gene:RALF*)',
        note="Pfam PF05498 VERIFIED (rapid alkalinization factor)",
    ),
    "PSK": dict(
        query='(xref:pfam-PF06404 OR protein_name:phytosulfokine OR gene:PSK*)',
        note="Pfam PF06404 VERIFIED (phytosulfokine precursor)",
    ),
    "CLE": dict(
        query='((protein_name:"CLAVATA3" OR protein_name:"CLE" OR '
              'protein_name:"ESR-related" OR gene:CLE* OR gene:CLV3) '
              'NOT protein_name:GOLVEN NOT protein_name:"root meristem growth factor" '
              'NOT protein_name:CLEL NOT gene:RGF* NOT gene:GLV*)',
        note="NAME-BASED, with GOLVEN/RGF/CLEL excluded -- UniProt names those "
             "'CLAVATA3/ESR (CLE)-related' for historical reasons but they are a "
             "separate family. Still verify the hits.",
    ),
    "RGF": dict(
        query='(protein_name:GOLVEN OR protein_name:"root meristem growth factor" '
              'OR protein_name:CLEL OR gene:RGF* OR gene:GLV*)',
        note="NAME-BASED: RGF/GLV/CLEL, split out of the CLE query above. A real "
             "SSP family in its own right, so better as a class than as noise.",
    ),
    "PEP": dict(
        query='(protein_name:"elicitor peptide" OR protein_name:PROPEP OR gene:PROPEP*)',
        note="NAME-BASED: verify the hits.",
    ),
    "CEP": dict(
        query='(protein_name:"C-terminally encoded peptide" OR gene:CEP*)',
        note="NAME-BASED. Optional 5th family; drop if yield is low.",
    ),
    "IDA": dict(
        query='(protein_name:"inflorescence deficient in abscission" OR gene:IDA OR gene:IDL*)',
        note="NAME-BASED. Optional 6th family; drop if yield is low.",
    ),
}
NEGATIVE_NAME = "NONSSP"


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def fetch(query, contact, retries=3, limit=None):
    """Fetch a TSV result set from UniProt.

    `limit` switches to the paginated /search endpoint, which is bounded and
    returns quickly. /stream is unbounded: fine for a narrow family query,
    fatal for a broad one over TrEMBL, which is what made the original
    negative-class query time out."""
    url = BASE_SEARCH if limit else BASE
    params = {"query": query, "format": "tsv", "fields": ",".join(FIELDS)}
    if limit:
        params["size"] = min(500, limit)
    headers = {"User-Agent": f"s2pepanalyst-corpus-builder (mailto:{contact})"}
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=180)
        if r.status_code == 200:
            break
        if attempt == retries - 1:
            raise RuntimeError(f"UniProt returned {r.status_code}\nquery: {query}\n{r.text[:400]}")
        time.sleep(3 * (attempt + 1))
    lines = r.text.strip().split("\n")
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


COLS = {
    "seq":      ["Sequence"],
    "acc":      ["Entry", "Accession"],
    "length":   ["Length"],
    "organism": ["Organism", "Organism (name)"],
    "name":     ["Protein names", "Protein name"],
    "peptide":  ["Peptide"],
    "orgid":    ["Organism (ID)", "Organism ID"],
    "reviewed": ["Reviewed"],
    "propep":   ["Propeptide", "Propeptide chain"],
}


def col(row, logical, required=False):
    """UniProt TSV uses human-readable headers whose exact strings vary by
    release. Resolve tolerantly rather than hard-coding one spelling."""
    for c in COLS[logical]:
        if c in row:
            return row[c]
    if required:
        raise KeyError(
            f"Could not find a column for '{logical}' (tried {COLS[logical]}). "
            f"Headers actually returned: {sorted(row)}. "
            f"Add the right name to COLS and re-run.")
    return ""


def family_query(spec, lo, hi, reviewed_only):
    parts = [spec["query"], f"taxonomy_id:{VIRIDIPLANTAE}", f"length:[{lo} TO {hi}]"]
    if reviewed_only:
        parts.append("reviewed:true")
    return " AND ".join(parts)


# ---------------------------------------------------------------------------
# FEATURE PARSING -- curated mature-peptide positions, not planted motifs
# ---------------------------------------------------------------------------
_FT = re.compile(r"(?:PEPTIDE|PROPEP)\s+(\d+)\.\.(\d+)")


def conserved_positions(row, seqlen):
    """0-indexed positions inside UniProt-annotated PEPTIDE / PROPEP features.
    Empty when the entry carries no such annotation -- which is informative,
    and reported as coverage below."""
    pos = []
    for field in ("peptide", "propep"):
        for m in _FT.finditer(col(row, field) or ""):
            a, b = int(m.group(1)) - 1, int(m.group(2))
            pos += [i for i in range(a, min(b, seqlen))]
    return sorted(set(pos))


# ---------------------------------------------------------------------------
# REDUNDANCY REDUCTION + CLUSTER-AWARE SPLIT
# ---------------------------------------------------------------------------
def kmers(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def greedy_cluster(seqs, threshold=0.7, k=4):
    """Greedy single-linkage clustering on k-mer containment.

    Deliberately approximate. It is here so the script has no external
    dependency; for a submission, re-run with CD-HIT or MMseqs2 at 70%
    identity (--cdhit below) and cite that instead. The containment measure
    is |A n B| / min(|A|,|B|), which is more aggressive than Jaccard for
    short-vs-long pairs and so errs toward merging -- the safe direction.
    """
    sets = [kmers(s, k) for s in seqs]
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    assign = [-1] * len(seqs)
    reps = []
    for i in order:
        placed = False
        for cid, rep in enumerate(reps):
            inter = len(sets[i] & sets[rep])
            if inter / max(1, min(len(sets[i]), len(sets[rep]))) >= threshold:
                assign[i] = cid
                placed = True
                break
        if not placed:
            assign[i] = len(reps)
            reps.append(i)
    return assign


def cluster_aware_split(records, clusters, val_frac=0.2, seed=0):
    """Assign whole clusters to train or val, stratified by family, so no
    homologue of a validation sequence appears in training."""
    rng = random.Random(seed)
    by_fam = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        by_fam[r["family"]][clusters[i]].append(i)
    train, val = [], []
    for fam, cl in by_fam.items():
        cids = list(cl)
        rng.shuffle(cids)
        target = sum(len(v) for v in cl.values()) * val_frac
        got, pending = 0, []
        # First pass: take clusters that fit without overshooting the target.
        # A single large cluster must not be allowed to swallow the whole
        # validation budget, which naive greedy filling does.
        for cid in cids:
            if got + len(cl[cid]) <= target * 1.15:
                val += cl[cid]
                got += len(cl[cid])
            else:
                pending.append(cid)
        # Second pass: if still well short (every remaining cluster is large),
        # take the smallest pending cluster so val is never empty.
        if got == 0 and pending:
            pending.sort(key=lambda c: len(cl[c]))
            val += cl[pending.pop(0)]
        for cid in pending:
            train += cl[cid]
    return sorted(train), sorted(val)


# ---------------------------------------------------------------------------
# QC GATES -- the corpus is not written unless these pass
# ---------------------------------------------------------------------------
def substring_baseline(records, min_len=5, coverage=0.90):
    """The check the synthetic corpus failed.

    For each family, search for any substring that appears in >= `coverage`
    of its members and in NO member of any other family. If such motifs
    exist for most families, the label is recoverable by exact matching and
    the task is not a representation benchmark.

    Searches substrings of the shortest member (any motif shared by the whole
    family must be a substring of its shortest sequence).
    """
    fams = sorted({r["family"] for r in records})
    by = {f: [r["seq"] for r in records if r["family"] == f] for f in fams}
    report = {}
    for f in fams:
        own, other = by[f], [s for g in fams if g != f for s in by[g]]
        shortest = min(own, key=len)
        best = None
        for L in range(min(30, len(shortest)), min_len - 1, -1):
            for i in range(len(shortest) - L + 1):
                sub = shortest[i:i + L]
                hit = sum(sub in s for s in own) / len(own)
                if hit >= coverage and not any(sub in s for s in other):
                    best = (sub, hit)
                    break
            if best:
                break
        report[f] = best
    return report


def length_only_accuracy(records, seed=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    fams = sorted({r["family"] for r in records})
    f2i = {f: i for i, f in enumerate(fams)}
    X = np.array([[len(r["seq"])] for r in records], float)
    y = np.array([f2i[r["family"]] for r in records])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        acc.append(KNeighborsClassifier(5).fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(acc))


def run_qc(records, clusters, train, val, strict=True):
    print("\n" + "=" * 72)
    print("QC GATES")
    print("=" * 72)
    ok = True
    fams = Counter(r["family"] for r in records)
    print(f"\n  corpus: {len(records)} sequences, {len(set(clusters))} clusters")
    for f, n in sorted(fams.items()):
        print(f"    {f:10s} {n:5d}")

    dup = len(records) - len({r["seq"] for r in records})
    print(f"\n  [1] exact duplicate sequences: {dup}")
    if dup:
        ok = False
        print("      FAIL -- deduplicate before use")

    straddle = len({clusters[i] for i in train} & {clusters[i] for i in val})
    print(f"  [2] clusters spanning train and val: {straddle}")
    if straddle:
        ok = False
        print("      FAIL -- split is not cluster-aware")

    chance = 1.0 / len(fams)
    lo = length_only_accuracy(records)
    print(f"  [3] length-only kNN accuracy: {lo:.3f}  (chance {chance:.3f})")
    if lo > chance + 0.25:
        ok = False
        print("      FAIL -- length is a strong shortcut; re-balance lengths")

    report = substring_baseline(records)
    leaky = [f for f, v in report.items() if v and f != NEGATIVE_NAME]
    print(f"  [4] exact-substring discriminators: {len(leaky)} of "
          f"{len([f for f in report if f != NEGATIVE_NAME])} families")
    for f, v in sorted(report.items()):
        print(f"      {f:10s} " + (f"{v[0]!r} in {v[1]*100:.0f}% of family, 0% elsewhere"
                                   if v else "none found  OK"))
    if len(leaky) >= 2:
        ok = False
        print("      FAIL -- family label is partly recoverable by exact matching.")
        print("             This is the failure mode of the synthetic corpus.")

    small = {f: n for f, n in fams.items() if n < 25}
    print(f"  [5] classes with <25 members (unstable under 5-fold CV): {len(small)}")
    for f, n in sorted(small.items()):
        print(f"      {f:10s} n={n}  -> {n/5:.1f} per fold")
    if small:
        ok = False
        print("      FAIL -- add --include-unreviewed or widen those queries")

    pos = [r for r in records if r["family"] != NEGATIVE_NAME]
    negs = [r for r in records if r["family"] == NEGATIVE_NAME]
    if negs and "reviewed" in records[0]:
        pr = np.mean([r["reviewed"] for r in pos])
        nr = np.mean([r["reviewed"] for r in negs])
        po = {r.get("organism_id") for r in pos}
        no = {r.get("organism_id") for r in negs}
        jac = len(po & no) / max(1, len(po | no))
        print(f"  [6] positive/negative metadata balance")
        print(f"      reviewed fraction: positives {pr:.2f}, negatives {nr:.2f}")
        print(f"      organism overlap (Jaccard): {jac:.2f}")
        if abs(pr - nr) > 0.20:
            ok = False
            print("      FAIL -- review status is confounded with the label; a model")
            print("             can separate the classes on curation artefacts.")
        if jac < 0.30:
            ok = False
            print("      FAIL -- negatives come from different organisms than positives;")
            print("             taxon composition is confounded with the label.")

    cov = np.mean([len(r["conserved"]) > 0 for r in records if r["family"] != NEGATIVE_NAME])
    print(f"  [7] curated PEPTIDE/PROPEP annotation coverage: {cov*100:.0f}% of non-negative entries")
    if cov < 0.25:
        print("      WARN -- low coverage; the attention-to-motif analysis will rest")
        print("             on few sequences. Consider reporting it on the annotated subset only.")

    print("\n  " + ("ALL GATES PASSED" if ok else "GATES FAILED -- see above"))
    if not ok and strict:
        sys.exit("\nAborting: refusing to write a corpus that fails QC. "
                 "Re-run with --force to write anyway (not recommended).")
    return ok


# ---------------------------------------------------------------------------
def main(argv=None):
    """argv=None reads sys.argv. In a notebook, call
    main(["--contact","you@uma.es","--include-unreviewed", ...]) instead of
    editing this function."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/splits.json")
    ap.add_argument("--contact", required=True, help="your email; UniProt asks clients to identify themselves")
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--per-family-cap", type=int, default=400)
    ap.add_argument("--max-per-organism", type=int, default=0,
                    help="cap sequences per organism per family (0 = no cap). "
                         "Curbs the Arabidopsis skew that Swiss-Prot imposes.")
    ap.add_argument("--min-class", type=int, default=25,
                    help="minimum viable class size for 5-fold CV; QC warns below this")
    ap.add_argument("--negatives", type=float, default=1.0,
                    help="negative-class size as a multiple of the mean positive class")
    ap.add_argument("--identity", type=float, default=0.7, help="redundancy-reduction threshold")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="add TrEMBL. Raises yield a lot, lowers annotation quality.")
    ap.add_argument("--families", default="RALF,PSK,CLE,PEP,RGF",
                    help="comma-separated subset of: " + ",".join(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    wanted = [f.strip() for f in a.families.split(",") if f.strip()]
    reviewed_only = not a.include_unreviewed
    rows_by_fam, seen = {}, set()

    print("=" * 72)
    print(f"FETCHING  ({'Swiss-Prot only' if reviewed_only else 'Swiss-Prot + TrEMBL'})")
    print("=" * 72)
    for fam in wanted:
        spec = FAMILIES[fam]
        q = family_query(spec, a.min_len, a.max_len, reviewed_only)
        rows = fetch(q, a.contact)
        keep = []
        for r in rows:
            s = (col(r, "seq", required=True) or "").strip().upper()
            if not s or s in seen or set(s) - set("ACDEFGHIKLMNPQRSTVWY"):
                continue
            seen.add(s)
            keep.append(r)
        if a.max_per_organism:
            per_org, capped = Counter(), []
            for row in keep:
                org = col(row, "organism")
                if per_org[org] < a.max_per_organism:
                    capped.append(row); per_org[org] += 1
            if len(capped) < len(keep):
                print(f"           organism cap: {len(keep)} -> {len(capped)}")
            keep = capped
        rows_by_fam[fam] = keep[: a.per_family_cap]
        print(f"  {fam:8s} {len(rows):5d} hits -> {len(rows_by_fam[fam]):4d} kept   [{spec['note']}]")
        if keep:
            ex = keep[0]
            print(f"           e.g. {col(ex,'acc') or '?'}  {(col(ex,'name') or '?')[:58]}")
        if len(keep) < 20:
            print(f"           WARN: low yield. Widen the query or add --include-unreviewed.")

    n_pos = sum(len(v) for v in rows_by_fam.values())
    if n_pos == 0:
        sys.exit("No sequences retrieved -- check queries and network.")

    # ---------------------------------------------------------------------
    # NEGATIVES, matched to the positives on organism AND review status.
    #
    # The naive query -- one broad TrEMBL search with a NOT clause per family --
    # makes UniProt scan the whole unreviewed plant set and times out. Worse,
    # "just take Swiss-Prot for the negatives" fixes the timeout by introducing
    # a confound: if positives are mostly TrEMBL and negatives all Swiss-Prot,
    # a model can separate them on curation artefacts instead of biology.
    #
    # Instead: query per organism (small, fast, bounded), draw negatives in the
    # same organism and reviewed/unreviewed proportions as the positives, and
    # exclude SSPs client-side. No NOT clauses reach the server.
    # ---------------------------------------------------------------------
    lens = [len(col(r, "seq")) for v in rows_by_fam.values() for r in v]
    lo, hi = int(np.percentile(lens, 5)), int(np.percentile(lens, 95))
    target = int(round(a.negatives * n_pos / max(1, len(wanted))))

    profile = Counter()
    for fam in wanted:
        for r in rows_by_fam[fam]:
            oid = col(r, "orgid")
            rev = (col(r, "reviewed") or "").strip().lower().startswith("rev")
            if oid:
                profile[(oid, rev)] += 1
    total_prof = sum(profile.values()) or 1

    print(f"\n  {NEGATIVE_NAME:8s} organism- and status-matched negatives "
          f"({lo}-{hi} aa), target {target}")
    print(f"           matching across {len({o for o, _ in profile})} organisms; "
          f"positives are {sum(v for (o, rv), v in profile.items() if rv)/total_prof:.0%} reviewed")

    pos_accs = {col(r, "acc") for v in rows_by_fam.values() for r in v}
    neg, rng = [], random.Random(a.seed)
    for (oid, rev), count in profile.most_common():
        want = int(np.ceil(target * count / total_prof))
        if want <= 0:
            continue
        q = (f"organism_id:{oid} AND length:[{lo} TO {hi}] "
             f"AND reviewed:{'true' if rev else 'false'}")
        try:
            rows = fetch(q, a.contact, limit=500)
        except RuntimeError as e:
            print(f"           skip taxon {oid}: {str(e)[:60]}")
            continue
        pool = []
        for r in rows:
            s_ = (col(r, "seq") or "").strip().upper()
            if (not s_ or s_ in seen or col(r, "acc") in pos_accs
                    or set(s_) - set("ACDEFGHIKLMNPQRSTVWY")
                    or SSP_NAME_RE.search(col(r, "name") or "")):
                continue
            pool.append(r)
        rng.shuffle(pool)
        for r in pool[:want]:
            seen.add(col(r, "seq").strip().upper())
            neg.append(r)
        if len(neg) >= target:
            break
    rows_by_fam[NEGATIVE_NAME] = neg[:target]
    got = rows_by_fam[NEGATIVE_NAME]
    rev_frac = np.mean([(col(r, "reviewed") or "").lower().startswith("rev") for r in got]) if got else 0
    print(f"           kept {len(got)} real proteins, {rev_frac:.0%} reviewed "
          f"(positives {sum(v for (o, rv), v in profile.items() if rv)/total_prof:.0%})")
    if len(got) < target * 0.6:
        print(f"           WARN: only {len(got)}/{target}. Raise --max-per-organism "
              f"or widen the length band.")

    records = []
    for fam, rows in rows_by_fam.items():
        for r in rows:
            s = col(r, "seq").strip().upper()
            cons = conserved_positions(r, len(s))
            # A "mature peptide" annotation covering the whole precursor gives the
            # attention-to-motif analysis no contrast, so treat it as unannotated.
            if len(cons) >= 0.9 * len(s):
                cons = []
            records.append(dict(
                seq=s, family=fam, kind="REAL", is_decoy=(fam == NEGATIVE_NAME),
                accession=col(r, "acc"), organism=col(r, "organism"),
                organism_id=col(r, "orgid"), protein_name=col(r, "name"),
                reviewed=(col(r, "reviewed") or "").strip().lower().startswith("rev"),
                conserved=cons if fam != NEGATIVE_NAME else [],
            ))
    random.Random(a.seed).shuffle(records)

    print(f"\n  clustering at {a.identity:.0%} k-mer containment "
          f"(use CD-HIT/MMseqs2 for the paper)...")
    clusters = greedy_cluster([r["seq"] for r in records], a.identity)
    print(f"  {len(records)} sequences -> {len(set(clusters))} clusters")

    train, val = cluster_aware_split(records, clusters, seed=a.seed)
    print(f"  cluster-aware split: train {len(train)}  val {len(val)}")

    run_qc(records, clusters, train, val, strict=not a.force)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(records=records, train_idx=train, val_idx=val,
                   clusters=[int(c) for c in clusters],
                   provenance=dict(source="UniProtKB REST",
                                   reviewed_only=reviewed_only,
                                   families=wanted, identity=a.identity,
                                   min_len=a.min_len, max_len=a.max_len,
                                   seed=a.seed, built=time.strftime("%Y-%m-%d"))),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  ({len(records)} real sequences)")
    print("\nNEXT: the downstream pipeline reads this file unchanged, but note that")
    print("MAX_LEN in model.py is 180 -- raise it if --max-len exceeds that, and")
    print("re-run pretraining from scratch. Every reported number changes.")


if __name__ == "__main__":
    main()
