#!/usr/bin/env python3
"""Create the expanded synthpop input from the manual annotations.

This is the cleaned equivalent of ``compute_features.py``,
``named_entity_recognition.py``, and ``expand_columns.py`` from the research
workspace. The frozen output is already included for exact reproduction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


FEATURES = ["english", "compound", "abbr", "brand_name", "human_name", "location_name", "misspelled"]


def first_entity(pipe, word: str, default: str | None) -> str | None:
    values = pipe(word)
    return str(values[0]["entity_group"]) if values else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manual_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--pos-model", type=Path, required=True, help="Brown POS checkpoint directory")
    parser.add_argument("--ner-model", default="dslim/bert-base-NER")
    args = parser.parse_args()

    try:
        from publicsuffix2 import PublicSuffixList
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SystemExit("Install the 'models' and 'generation' extras first") from exc

    frame = pd.read_csv(args.manual_csv)
    frame = frame[frame["english"].eq("Yes")].copy().reset_index(drop=True)
    frame = frame.rename(columns={"domain": "Domain_Name", "segmented": "Segmentation"})
    for column in FEATURES:
        frame[column] = frame[column].map({"Yes": 1, "No": 0})

    suffixes = PublicSuffixList()
    frame["Num_Segments"] = frame["Segmentation"].map(lambda value: len(str(value).split("|")) - 1)
    frame["Domain_Length"] = frame["Domain_Name"].map(lambda value: len(str(value).split(".")[0]))
    frame["eTLD"] = frame["Domain_Name"].map(suffixes.get_tld)

    pos_tokenizer = AutoTokenizer.from_pretrained(str(args.pos_model))
    pos_model = AutoModelForTokenClassification.from_pretrained(str(args.pos_model))
    pos_pipe = pipeline(
        "token-classification", model=pos_model, tokenizer=pos_tokenizer, aggregation_strategy="simple"
    )
    ner_tokenizer = AutoTokenizer.from_pretrained(args.ner_model)
    ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_model)
    ner_pipe = pipeline(
        "token-classification", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple"
    )

    pos_rows: list[list[str | None]] = []
    ner_rows: list[list[str | None]] = []
    ner_counts: list[Counter[str]] = []
    for segmentation in frame["Segmentation"].astype(str):
        words = segmentation.split("|")[:-1]
        pos = [first_entity(pos_pipe, word, None) if word else None for word in words]
        ner = [first_entity(ner_pipe, word, "O") if word else None for word in words]
        pos_rows.append(pos)
        ner_rows.append(ner)
        ner_counts.append(Counter(value for value in ner if value not in {None, "O"}))

    for entity in ("ORG", "PER", "LOC", "MISC"):
        frame[f"NER_{entity}"] = [counts.get(entity, 0) for counts in ner_counts]
    for prefix, rows in (("POS_Tags", pos_rows), ("NER_Tags", ner_rows)):
        for index in range(7):
            frame[f"{prefix}_{index}"] = [row[index] if index < len(row) else None for row in rows]

    keep = [
        "Domain_Name", "Segmentation", *FEATURES, "Num_Segments", "Domain_Length", "eTLD",
        "NER_ORG", "NER_PER", "NER_LOC", "NER_MISC",
        *(f"POS_Tags_{index}" for index in range(7)),
        *(f"NER_Tags_{index}" for index in range(7)),
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(args.output_csv, index=True)


if __name__ == "__main__":
    main()
