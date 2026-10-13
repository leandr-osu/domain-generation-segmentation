#!/usr/bin/env python3
"""Turn the WikiText word counts into weighted files grouped by POS tag."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TAGS = ("ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON", "PRT", "VERB", "X")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frequencies", type=Path)
    parser.add_argument("pos_model", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SystemExit("Install the 'models' extra first") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(args.pos_model))
    model = AutoModelForTokenClassification.from_pretrained(str(args.pos_model))
    tagger = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    with args.frequencies.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    by_tag: dict[str, list[tuple[str, str]]] = {tag: [] for tag in TAGS}
    # The first version also passed the CSV header to the tagger. It was not
    # written to any POS file, but we keep the same call order for reproducibility.
    for row in rows:
        if len(row) < 2:
            continue
        word, count = row[0], row[1]
        for result in tagger(word):
            tag = str(result["entity_group"])
            if tag in by_tag:
                by_tag[tag].append((word.strip(), count.strip()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tag, values in by_tag.items():
        with (args.output_dir / f"{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["word", "weight"])
            writer.writerows(values)
    print(args.output_dir)


if __name__ == "__main__":
    main()
