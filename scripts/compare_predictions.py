#!/usr/bin/env python3
"""Compare a new run with the saved per-domain predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMN_PAIRS = {
    "viterbi_output": "viterbi",
    "fallback_output": "viterbi_fallback",
    "vbe_output": "entropy_vbe",
    "pos_output": "pos",
    "gpt_5_mini_output": "gpt_5_mini",
}


def normalized(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("computed", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    computed = pd.read_csv(args.computed, low_memory=False)
    reference = pd.read_csv(args.reference, low_memory=False)
    if len(computed) > len(reference):
        raise SystemExit("computed output has more rows than its reference")
    reference = reference.iloc[: len(computed)]
    if normalized(computed["domain"]).tolist() != normalized(reference["domain"]).tolist():
        raise SystemExit("domain rows or row order differ")

    compared = 0
    failures: list[str] = []
    for actual_column, expected_column in COLUMN_PAIRS.items():
        if actual_column not in computed:
            continue
        compared += 1
        mismatch = normalized(computed[actual_column]) != normalized(reference[expected_column])
        count = int(mismatch.sum())
        print(f"{actual_column}: {'ok' if count == 0 else f'{count} mismatches'}")
        if count:
            examples = mismatch[mismatch].index[:5].tolist()
            failures.append(f"{actual_column}: rows {examples}")
    if not compared:
        raise SystemExit("no recognized prediction columns were found")
    if failures:
        raise SystemExit("Prediction comparison failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
