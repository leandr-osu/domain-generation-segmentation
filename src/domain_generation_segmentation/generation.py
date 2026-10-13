"""Generate domain names from synthpop feature rows."""

from __future__ import annotations

import argparse
import ast
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAX_SEGMENTS = 6
DOMAIN_RE = re.compile(r"[^a-z0-9-]")
NER_RESOURCES = {"PER": "names.csv", "ORG": "brands.csv", "LOC": "cities.csv"}
FLAG_RESOURCES = {"compound": "compounds.csv", "abbr": "acronyms.csv"}

# Resources sampled for X tags, in a fixed order so sampling is portable.
PUBLICATION_X_RESOURCE_ORDER = [
    "ADP.csv",
    "brands.csv",
    "NUM.csv",
    "ADJ.csv",
    "PRON.csv",
    "DET.csv",
    "NOUN.csv",
    "CONJ.csv",
    "PRT.csv",
    "english.csv",
    "ADV.csv",
    "VERB.csv",
]


class WeightedSampler:
    def __init__(self, path: Path, rng: random.Random):
        frame = pd.read_csv(path)
        if frame.shape[1] < 2:
            raise ValueError(f"resource must have word and weight columns: {path}")
        # The original run turned empty cells into the literal string "nan".
        # Do this explicitly so pandas 2 and 3 give the same result.
        self.words = frame.iloc[:, 0].fillna("nan").astype(str).to_numpy()
        weights = pd.to_numeric(frame.iloc[:, 1], errors="raise").to_numpy(dtype=float)
        self.cdf = np.cumsum(weights) / np.sum(weights)
        self.rng = rng

    def sample(self) -> str:
        index = int(np.searchsorted(self.cdf, self.rng.random()))
        return str(self.words[index])


def normalize_label(label: object) -> str:
    value = DOMAIN_RE.sub("", str(label).lower())
    return re.sub(r"-{2,}", "-", value).strip("-")


def transpose_typo(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    index = rng.randint(0, len(word) - 2)
    return word[:index] + word[index + 1] + word[index] + word[index + 2 :]


def _parse_tag_list(value: object) -> list[str | None]:
    if isinstance(value, list):
        raw = value
    elif pd.isna(value):
        raw = []
    else:
        text = str(value).strip()
        try:
            raw = ast.literal_eval(text.replace("nan", "None"))
        except (SyntaxError, ValueError):
            raw = [token.strip().strip("'\"") for token in text.strip("[]").split(",")]
    tags: list[str | None] = []
    for token in raw:
        if token is None or (isinstance(token, float) and np.isnan(token)):
            tags.append(None)
        elif str(token) in {"", "nan", "None"}:
            tags.append(None)
        else:
            tags.append(str(token))
    return [tag for tag in (tags + [None] * MAX_SEGMENTS)[:MAX_SEGMENTS] if tag != "."]


def collapse_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Pack the seven POS/NER columns back into lists for the sampler."""
    output = frame.copy()
    for prefix in ("POS_Tags", "NER_Tags"):
        columns = [f"{prefix}_{index}" for index in range(7)]
        if all(column in output.columns for column in columns):
            output[prefix] = output[columns].values.tolist()
            output = output.drop(columns=columns)
    return output


class DomainGenerator:
    def __init__(self, resource_dir: str | Path, seed: int = 42):
        self.resource_dir = Path(resource_dir)
        self.rng = random.Random(seed)
        self.samplers: dict[str, WeightedSampler] = {}
        missing = [
            name
            for name in sorted(set(NER_RESOURCES.values()) | set(FLAG_RESOURCES.values()) | set(PUBLICATION_X_RESOURCE_ORDER))
            if not (self.resource_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing generator resources: " + ", ".join(missing) +
                "; see \"Generate the synthetic domains\" in README.md"
            )

    def sampler(self, filename: str) -> WeightedSampler:
        if filename not in self.samplers:
            self.samplers[filename] = WeightedSampler(self.resource_dir / filename, self.rng)
        return self.samplers[filename]

    def sample_row(self, row: object) -> tuple[str, str]:
        pos_tags = _parse_tag_list(getattr(row, "POS_Tags"))
        ner_tags = _parse_tag_list(getattr(row, "NER_Tags"))
        slots: list[str | None] = [None] * MAX_SEGMENTS

        for index, ner in enumerate(ner_tags):
            if ner in NER_RESOURCES:
                slots[index] = self.sampler(NER_RESOURCES[ner]).sample()

        for flag, resource in FLAG_RESOURCES.items():
            if int(getattr(row, flag)) == 1:
                for index in range(MAX_SEGMENTS):
                    if slots[index] is None:
                        slots[index] = self.sampler(resource).sample()
                        break

        for index, pos in enumerate(pos_tags):
            if slots[index] is not None or not pos:
                continue
            resource = self.rng.choice(PUBLICATION_X_RESOURCE_ORDER) if pos == "X" else f"{pos}.csv"
            slots[index] = self.sampler(resource).sample()

        words = [normalize_label(word) for word in slots if word]
        if int(getattr(row, "misspelled")) == 1 and words:
            index = self.rng.randrange(len(words))
            words[index] = transpose_typo(words[index], self.rng)
        etld = str(getattr(row, "eTLD"))
        return "".join(words) + "." + etld, "|".join([*words, etld])

    def generate(self, feature_csv: str | Path, output_csv: str | Path, chunk_size: int = 1000) -> Path:
        input_path = Path(feature_csv)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first = True
        for raw_chunk in pd.read_csv(input_path, chunksize=chunk_size):
            chunk = collapse_feature_columns(raw_chunk)
            chunk = chunk.drop(columns=[column for column in chunk.columns if column.startswith("Unnamed:")])
            domains: list[str] = []
            segmentations: list[str] = []
            for row in chunk.itertuples(index=False):
                domain, segmentation = self.sample_row(row)
                domains.append(domain)
                segmentations.append(segmentation)
            chunk.insert(0, "segmented", segmentations)
            chunk.insert(0, "domain", domains)
            chunk.to_csv(output_path, mode="w" if first else "a", index=False, header=first)
            first = False
        return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="synthpop feature CSV")
    parser.add_argument("resources", type=Path, help="prepared generator-resource directory")
    parser.add_argument("output", type=Path, help="output synthetic-domain CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args(argv)
    path = DomainGenerator(args.resources, seed=args.seed).generate(
        args.features, args.output, chunk_size=args.chunk_size
    )
    print(path)


if __name__ == "__main__":
    main()
