"""Variation-of-branching-entropy (VBE) domain segmenter."""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

import numpy as np


SEPARATORS = re.compile(r"[-.]")


def _pieces(text: str) -> list[str]:
    return [part for part in SEPARATORS.split(str(text)) if part]


def _letters(text: str) -> str:
    return "".join(char for char in text.lower() if "a" <= char <= "z")


def _integer_keys(mapping: dict[str | int, object]) -> dict[int, object]:
    return {int(key): value for key, value in mapping.items()}


class VBESegmenter:
    """Entropy segmenter from Magistry (2012), as evaluated in the paper.

    The constructor accepts the complete character n-gram dictionaries. For
    convenient reproduction, :meth:`from_json_gz` loads a safe, corpus-specific
    inference state derived from the original 10-million-domain dictionaries.
    """

    def __init__(self, dictionaries: dict[int, dict[str, int]]):
        if 1 not in dictionaries or len(dictionaries) < 2:
            raise ValueError("VBE requires character n-gram dictionaries starting at n=1")
        self.freq_dicts = dictionaries
        self.max_length = max(dictionaries)
        self.charset = list(dictionaries[1])
        self.vbe_dicts: dict[str, dict[int, dict[str, float]]] = {
            "forwards": {},
            "backwards": {},
        }
        self.norm_constants: dict[str, dict[int, tuple[float, float]]] = {
            "forwards": {},
            "backwards": {},
        }
        for length, dictionary in dictionaries.items():
            if length == self.max_length:
                continue
            for mode in ("forwards", "backwards"):
                self.vbe_dicts[mode][length] = self._bulk_vbe(dictionary, mode)
                values = np.fromiter(dictionary.values(), dtype=np.float64)
                self.norm_constants[mode][length] = (float(values.mean()), float(values.std()))
        # The normalized VBE values are now sufficient for inference.
        self.freq_dicts = {}
        self._branching_entropy.cache_clear()

    @classmethod
    def from_json_gz(
        cls,
        path: str | Path,
        dataset_path: str | Path | None = None,
    ) -> "VBESegmenter":
        """Load the non-executable JSON artifact and optionally verify its CSV."""
        source = Path(path)
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported VBE state format in {source}")
        if dataset_path is not None:
            digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
            if digest != payload.get("dataset_sha256"):
                raise ValueError("VBE state does not match the supplied evaluation CSV")
        return cls.from_precomputed_state(payload["state"])

    @classmethod
    def from_precomputed_state(cls, state: dict[str, object]) -> "VBESegmenter":
        segmenter = cls.__new__(cls)
        segmenter.freq_dicts = {}
        segmenter.max_length = int(state["max_length"])
        segmenter.charset = list(state["charset"])
        raw_vbe = state["vbe_dicts"]
        raw_norm = state["norm_constants"]
        segmenter.vbe_dicts = {
            mode: {
                int(length): {str(context): float(value) for context, value in values.items()}
                for length, values in raw_vbe[mode].items()
            }
            for mode in ("forwards", "backwards")
        }
        segmenter.norm_constants = {
            mode: {
                int(length): (float(values[0]), float(values[1]))
                for length, values in raw_norm[mode].items()
            }
            for mode in ("forwards", "backwards")
        }
        return segmenter

    @lru_cache(maxsize=1024)
    def _branching_entropy(self, context: str, mode: str) -> float:
        length = len(context)
        dictionary = self.freq_dicts.get(length + 1)
        if dictionary is None:
            return 0.0
        counts: list[int] = []
        for char in self.charset:
            key = context + char if mode == "forwards" else char + context
            if key in dictionary:
                counts.append(dictionary[key])
        total = sum(counts)
        if not total:
            return 0.0
        probabilities = np.asarray(counts, dtype=np.float64) / total
        return float(-np.sum(probabilities * np.log(probabilities + 1e-10)))

    def _bulk_vbe(self, dictionary: dict[str, int], mode: str) -> dict[str, float]:
        output: dict[str, float] = {}
        for ngram in dictionary:
            previous = ngram[:-1] if mode == "forwards" else ngram[1:]
            output[ngram] = self._branching_entropy(ngram, mode) - self._branching_entropy(previous, mode)
        return output

    @lru_cache(maxsize=1024)
    def normalized_vbe(self, context: str, mode: str) -> float:
        constants = self.norm_constants[mode].get(len(context))
        if constants is None:
            return 0.0
        mean, std = constants
        value = self.vbe_dicts.get(mode, {}).get(len(context), {}).get(context, 0.0)
        return float((value - mean) / (std + 1e-10))

    def autonomy(self, context: str) -> float:
        return min(self.normalized_vbe(context, "forwards"), self.normalized_vbe(context, "backwards"))

    def segment(self, text: str, limit: int | None = None) -> list[str]:
        output: list[str] = []
        for raw_piece in _pieces(text):
            piece = _letters(raw_piece)
            if not piece:
                continue
            cache: dict[str, tuple[float, list[str]]] = {}

            def search(remaining: str) -> tuple[float, list[str]]:
                if remaining in cache:
                    return cache[remaining]
                if not remaining:
                    return 0.0, []
                best_score = float("-inf")
                best_words: list[str] = []
                max_prefix = min(len(remaining), limit or self.max_length - 1)
                for position in range(1, max_prefix + 1):
                    prefix = remaining[:position]
                    suffix_score, suffix_words = search(remaining[position:])
                    score = self.autonomy(prefix) * len(prefix) + suffix_score
                    if score > best_score:
                        best_score = score
                        best_words = [prefix, *suffix_words]
                cache[remaining] = best_score, best_words
                return cache[remaining]

            output.extend(search(piece)[1])
        return output

    def segment_many(self, domains: Iterable[str]) -> list[str]:
        return ["|".join(self.segment(domain)) for domain in domains]


def read_ngram_dictionaries(directory: str | Path, max_n: int = 10) -> dict[int, dict[str, int]]:
    """Load dictionaries written by ``scripts/build_vbe_dictionaries.py``."""
    base = Path(directory)
    dictionaries: dict[int, dict[str, int]] = {}
    for n in range(1, max_n + 1):
        values: dict[str, int] = {}
        with (base / f"{n}gram_dict.txt").open(encoding="utf-8") as handle:
            for line in handle:
                key, count = line.rstrip("\n").split(": ", 1)
                values[key] = int(count)
        dictionaries[n] = values
    return dictionaries
