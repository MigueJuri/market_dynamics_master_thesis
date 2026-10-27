#!/usr/bin/env python3
"""Bounded peek at a CSV/Parquet file. Prints ≤40 lines of output."""
import sys
from pathlib import Path
import pandas as pd

SENTINELS = [0, -1, -999, 999999, 9999999999, -9999999999]


def main(path: str) -> None:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    print(f"path: {p}")
    print(f"shape: {df.shape}")
    print("dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print("head(5):")
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        print(df.head(5).to_string(index=False))

    print("na_counts:")
    na = df.isna().sum()
    for col, count in na[na > 0].items():
        print(f"  {col}: {count}")

    print("sentinel_counts (exact values):")
    for col in df.select_dtypes(include="number").columns:
        for v in SENTINELS:
            c = (df[col] == v).sum()
            if c:
                print(f"  {col}=={v}: {c}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path.csv|path.parquet>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
