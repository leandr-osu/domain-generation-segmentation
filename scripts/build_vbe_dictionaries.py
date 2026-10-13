#!/usr/bin/env python3
"""Count character n-grams in a file with one suffix-stripped domain per line."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domains", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-n", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, args.max_n + 1):
        counts: Counter[str] = Counter()
        with args.domains.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                counts.update(text[index : index + n] for index in range(len(text) - n + 1))
        with (args.output_dir / f"{n}gram_dict.txt").open("w", encoding="utf-8") as handle:
            for gram, count in counts.most_common():
                handle.write(f"{gram}: {count}\n")
        print(f"n={n}: {len(counts):,} unique n-grams")


if __name__ == "__main__":
    main()
