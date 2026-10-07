#!/usr/bin/env python3
"""Stream WikiText-103 and build the frequency table used by generation."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "there", "they", "this", "to",
    "was", "were", "will", "with", "you", "your",
}
PUNCTUATION_BEFORE_SPACE = re.compile(r"[^\w\s]+(?=\s)")
LETTERS_ONLY = re.compile(r"^[a-zA-Z]+$")


def tokenize(text: str, minimum_length: int) -> list[str]:
    text = PUNCTUATION_BEFORE_SPACE.sub("", text.lower())
    return [
        token
        for token in text.split()
        if len(token) >= minimum_length and LETTERS_ONLY.fullmatch(token)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--variant", default="wikitext-103-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--minimum-length", type=int, default=1)
    parser.add_argument(
        "--keep-stopwords",
        action="store_true",
        help="The publication run removed the small built-in stopword list.",
    )
    parser.add_argument("--progress-every", type=int, default=200_000)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the 'generation' extra first") from exc

    rows = load_dataset(args.dataset, args.variant, split=args.split, streaming=True)
    frequencies: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        tokens = tokenize(str(row.get("text", "")), args.minimum_length)
        if not args.keep_stopwords:
            tokens = [token for token in tokens if token not in STOPWORDS]
        frequencies.update(tokens)
        if args.progress_every and index % args.progress_every == 0:
            print(f"rows={index:,} unique_words={len(frequencies):,}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", "count"])
        writer.writerows(frequencies.most_common())
    print(args.output)


if __name__ == "__main__":
    main()
