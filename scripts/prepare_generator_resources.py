#!/usr/bin/env python3
"""Convert downloaded name, city, brand, and flag data to sampler CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["names", "cities", "brands", "compounds", "acronyms"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--column", help="Input column for compounds/acronyms")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)

    if args.kind == "names":
        output = frame.groupby("Name", as_index=False)["Count"].sum().sort_values("Count", ascending=False)
        output.columns = ["name", "weight"]
    elif args.kind == "cities":
        output = frame[["city", "population"]].drop_duplicates(subset="city")
        output.columns = ["city", "weight"]
    elif args.kind == "brands":
        selected = frame[frame["locale"].eq("UK")]
        selected = selected[selected["brand"].astype(str).str.split().str.len() <= 2]
        output = selected["brand"].value_counts().rename_axis("brand").reset_index(name="weight")
    else:
        column = args.column or ("stim" if args.kind == "compounds" else frame.columns[0])
        values = frame[column].dropna().astype(str).drop_duplicates()
        if args.kind == "acronyms":
            values = values[values.str.fullmatch(re.compile(r"[A-Za-z0-9]+"))]
        output = pd.DataFrame({"word": values, "weight": 1})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
