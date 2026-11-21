"""Check data/raw/train.csv for the seven kinds of data problem.

Handbook section 7 / PROBLEM.md section 4. The script only READS the CSV files;
it never writes them. It checks this by hashing train.csv before and after.

Run from the repository root:
    python -m mp093.check_data
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

from mp093.config import data_path, load_params

PARAMS = load_params()
TRAIN = data_path(PARAMS, "train")
LATER = data_path(PARAMS, "later")
HOLDOUT = data_path(PARAMS, "holdout")

TARGET = PARAMS["target"]
GROUP = PARAMS["cleaning"]["unit_fix"]["group_column"]
ID_COLS = ["record_id"]
DATE = "date"

SENTINEL_CANDIDATES = [-1, -9, -99, -999, -9999, 0, 99, 999, 9999]
UNIT_FACTORS = [10, 12, 60, 100, 1000]   # exact unit conversions to snap to
STUCK_MIN_DAYS = 7                        # a run must cover this many dates


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_features(df):
    """Numeric columns except the target."""
    return [c for c in df.select_dtypes("number").columns if c != TARGET]


def text_features(df):
    return [c for c in df.select_dtypes(exclude="number").columns
            if c not in ID_COLS + [DATE]]


def header(n, title):
    print("\n" + "=" * 70)
    print(f"({n}) {title}")
    print("=" * 70)


# ---------------------------------------------------------------- (1)
def check_leaking_column(train, holdout):
    header(1, "leaking_column")
    missing = [c for c in train.columns if c not in holdout.columns and c != TARGET]
    print(f"Columns in train.csv but NOT in holdout_features.csv: {missing}")

    corr = train[numeric_features(train)].corrwith(train[TARGET]).sort_values(ascending=False)
    print(f"\nCorrelation of each numeric column with {TARGET}:")
    print(corr.round(4).to_string())

    leaks = [c for c in missing if c in corr.index and abs(corr[c]) > 0.95]
    for c in leaks:
        diff = (train[c] - train[TARGET]).abs()
        print(f"\n{c}: mean |{c} - {TARGET}| = {diff.mean():.1f} kg "
              f"(target mean {train[TARGET].mean():.1f} kg)")
    present = len(leaks) > 0
    print(f"\nVERDICT present={present}" + (f"  column={leaks[0]}" if present else ""))
    return {"present": present, "column": leaks[0] if present else None}


# ---------------------------------------------------------------- (2)
def check_unit_mixup(train):
    header(2, "unit_mixup")
    found = None
    for c in numeric_features(train):
        med = train.groupby(GROUP)[c].median()
        if (med <= 0).any():
            continue
        hi, lo = med.idxmax(), med.idxmin()
        ratio = med[hi] / med[lo]
        flag = "  <-- suspicious" if ratio > 5 else ""
        print(f"{c:30s} per-{GROUP} medians {med.round(1).to_dict()}  max/min={ratio:.1f}{flag}")
        if ratio > 5 and found is None:
            found = (c, hi)

    if found is None:
        print("\nNo column has one block more than 5x away from the others.")
        print("VERDICT present=False")
        return {"present": False, "column": None, "group_value": None, "factor": None}

    c, g = found
    odd = train.loc[train[GROUP] == g, c]
    rest = train.loc[train[GROUP] != g, c]
    ratio = odd.median() / rest.median()
    factor = min(UNIT_FACTORS, key=lambda f: abs(f - ratio))
    print(f"\n{c}: {GROUP}='{g}' median {odd.median():.1f} vs other blocks {rest.median():.1f} "
          f"-> ratio {ratio:.1f}, nearest exact unit factor = {factor}")
    print(f"  '{g}' range:              {odd.min():.1f} .. {odd.max():.1f}")
    print(f"  '{g}' range / {factor}:  {odd.min() / factor:.1f} .. {odd.max() / factor:.1f}")
    print(f"  other blocks range:      {rest.min():.1f} .. {rest.max():.1f}")
    print(f"  rows of '{g}' affected:  {len(odd)} of {len(odd)} (every row of that block)")
    print(f"\nVERDICT present=True  column={c}  group_value={g}  factor={factor}")
    return {"present": True, "column": c, "group_value": g, "factor": factor}


# ---------------------------------------------------------------- (3)
def check_sentinel(train, later):
    header(3, "sentinel_value")
    print("A sentinel must be (a) outside the range seen in the clean later.csv AND")
    print("(b) repeated far more often than a normal value of that column")
    print("(at least 5 rows and 3x the column's typical count per value).")
    print(f"Candidates checked: {SENTINEL_CANDIDATES}\n")
    hits = []
    for c in numeric_features(train):
        lo, hi = later[c].min(), later[c].max()
        typical = train[c].value_counts().median()
        for v in SENTINEL_CANDIDATES:
            n = int((train[c] == v).sum())
            if n == 0:
                continue
            outside = not (lo <= v <= hi)
            repeated = n >= max(5, 3 * typical)
            verdict = "SENTINEL" if outside and repeated else "not a sentinel"
            print(f"{c:30s} value {v:>6}: {n:4d} rows   later.csv range {lo}..{hi}"
                  f"   typical count/value {typical:.0f}   -> {verdict}")
            if outside and repeated:
                hits.append((c, v, n))

    if not hits:
        print("\nNo candidate value is both impossible and over-repeated.")
        print("VERDICT present=False")
        return {"present": False, "column": None, "value": None, "count": None}

    c, v, n = max(hits, key=lambda h: h[2])
    dedup = int((train.drop_duplicates()[c] == v).sum())
    second = train.loc[train[c] != v, c].value_counts().iloc[0]
    print(f"\n{c} = {v}: {n} rows as given ({dedup} after dropping duplicates); "
          f"the next most common value of {c} appears only {second} times.")
    print(f"\nVERDICT present=True  column={c}  value={v}  count={n}")
    return {"present": True, "column": c, "value": int(v), "count": n}


# ---------------------------------------------------------------- (4)
def check_duplicates(train):
    header(4, "duplicate_rows")
    no_id = train.drop(columns=ID_COLS)
    n_no_id = int(no_id.duplicated().sum())
    n_full = int(train.duplicated().sum())
    print(f"Extra copies ignoring record_id:  {n_no_id}")
    print(f"Extra copies including record_id: {n_full}")
    print(f"Rows: {len(train)} as given -> {len(train) - n_no_id} after removing extras")
    if n_no_id:
        print("\nExample duplicated record:")
        dup_ids = train.loc[no_id.duplicated(keep=False), "record_id"].head(2).tolist()
        print(train[train.record_id.isin(dup_ids)].to_string(index=False))
    present = n_no_id > 0
    print(f"\nVERDICT present={present}" + (f"  count={n_no_id}" if present else ""))
    return {"present": present, "count": n_no_id if present else None}


# ---------------------------------------------------------------- (5)
def longest_run(df, col):
    """Longest stretch of consecutive rows (sorted by date) with the same value."""
    run_id = (df[col] != df[col].shift()).cumsum()
    runs = df.groupby(run_id).agg(value=(col, "first"), rows=(col, "size"),
                                  start=(DATE, "first"), end=(DATE, "last"),
                                  days=(DATE, "nunique"))
    return runs.sort_values(["days", "rows"], ascending=False).iloc[0]


def check_stuck_sensor(train):
    header(5, "stuck_sensor")
    data = train.drop_duplicates().sort_values([DATE, "record_id"], kind="stable")
    print(f"Sorted by date ({data[DATE].min()} .. {data[DATE].max()}, "
          f"{data[DATE].nunique()} dates). A stuck run must cover >= {STUCK_MIN_DAYS} dates.\n")
    best = None
    for c in numeric_features(train):
        r = longest_run(data, c)
        per_block = max((longest_run(g, c) for _, g in data.groupby(GROUP)),
                        key=lambda x: (x["days"], x["rows"]))
        print(f"{c:30s} all rows: {r['rows']:3d} rows / {r['days']:2d} dates (value {r['value']})"
              f"   within one block: {per_block['rows']:3d} rows / {per_block['days']:2d} dates")
        for cand in (r, per_block):
            if cand["days"] >= STUCK_MIN_DAYS and (best is None or cand["days"] > best[1]["days"]):
                best = (c, cand)

    if best is None:
        print(f"\nNo column repeats one value over {STUCK_MIN_DAYS}+ consecutive dates.")
        print("VERDICT present=False")
        return {"present": False, "column": None, "start_date": None, "end_date": None}

    c, r = best
    print(f"\n{c} stuck at {r['value']} for {r['rows']} rows, {r['start']} .. {r['end']}")
    print(f"\nVERDICT present=True  column={c}  start_date={r['start']}  end_date={r['end']}")
    return {"present": True, "column": c, "start_date": r["start"], "end_date": r["end"]}


# ---------------------------------------------------------------- (6)
def check_target_typos(train, leak_col):
    header(6, "target_typos")
    y = train[TARGET].sort_values().reset_index(drop=True)
    print(f"{TARGET}: min {y.min()}  median {y.median()}  max {y.max()}")
    print(f"Top 10 values: {y.tail(10).tolist()}")

    jumps = (y / y.shift()).iloc[len(y) // 2:]       # gaps in the upper half
    i = jumps.idxmax()
    print(f"Biggest jump in the upper half: {y[i - 1]} -> {y[i]} (x{jumps[i]:.2f})")
    above_gap = int(len(y) - i) if jumps[i] >= 3 else 0

    above_10x = int((train[TARGET] > 5 * y.median()).sum())
    print(f"Rows above 5 x median ({5 * y.median():.0f} kg): {above_10x}")

    if leak_col:
        ratio = train[TARGET] / train[leak_col]
        n_ratio = int(ratio.between(8, 12).sum())
        print(f"Cross-check: {TARGET}/{leak_col} is between 8 and 12 for {n_ratio} rows "
              f"(normal ratio: median {ratio.median():.3f}, max {ratio.max():.3f})")

    count = max(above_gap, above_10x)
    present = count > 0
    if not present:
        print("\nNo group of values sits far above the rest; no gap of 3x or more.")
    print(f"\nVERDICT present={present}" + (f"  count={count}" if present else ""))
    return {"present": present, "count": count if present else None}


# ---------------------------------------------------------------- (7)
def normalise(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).strip().lower())


def check_spelling(train, later):
    header(7, "spelling_variants")
    found = None
    for c in text_features(train):
        raw = sorted(train[c].unique())
        norm = sorted({normalise(v) for v in raw})
        clean = sorted(later[c].unique())
        not_in_later = [v for v in raw if v not in set(clean)]
        print(f"{c}:")
        print(f"  train raw levels ({len(raw)}): {raw}")
        print(f"  train normalised ({len(norm)}): {norm}")
        print(f"  later.csv levels ({len(clean)}): {clean}")
        print(f"  train values not spelled as in later.csv: {not_in_later or 'none'}")
        if (len(raw) > len(norm) or not_in_later) and found is None:
            found = (c, len(clean))

    if found is None:
        print("\nEvery text value is already spelled exactly as in later.csv.")
        print("VERDICT present=False")
        return {"present": False, "column": None, "true_levels": None}

    c, n = found
    print(f"\nVERDICT present=True  column={c}  true_levels={n}")
    return {"present": True, "column": c, "true_levels": n}


def main():
    before = sha256(TRAIN)
    train = pd.read_csv(TRAIN)
    later = pd.read_csv(LATER)
    holdout = pd.read_csv(HOLDOUT)
    print(f"train.csv sha256: {before}")
    print(f"train {train.shape}, later {later.shape}, holdout {holdout.shape}")

    results = {}
    results["leaking_column"] = check_leaking_column(train, holdout)
    results["unit_mixup"] = check_unit_mixup(train)
    results["sentinel_value"] = check_sentinel(train, later)
    results["duplicate_rows"] = check_duplicates(train)
    results["stuck_sensor"] = check_stuck_sensor(train)
    results["target_typos"] = check_target_typos(train, results["leaking_column"]["column"])
    results["spelling_variants"] = check_spelling(train, later)

    print("\n" + "=" * 70)
    print("SUMMARY (data_issues for reports/findings.yaml)")
    print("=" * 70)
    for kind, r in results.items():
        print(f"{kind}: {r}")

    after = sha256(TRAIN)
    assert before == after, "train.csv changed!"
    print(f"\ntrain.csv unchanged (sha256 identical before and after).")


if __name__ == "__main__":
    main()
