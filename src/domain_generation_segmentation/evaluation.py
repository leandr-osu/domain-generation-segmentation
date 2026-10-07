"""Metrics used in the publication's segmentation table."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def normalize_segmentation(value: object) -> str:
    """Return the stored pipe-delimited representation without re-tokenizing it."""
    if _is_missing(value):
        return ""
    return str(value).strip()


def exact_match(reference: object, prediction: object) -> bool:
    return normalize_segmentation(reference) == normalize_segmentation(prediction)


def jaccard_similarity(reference: object, prediction: object) -> float:
    """Jaccard similarity over the *set* of pipe-delimited segments.

    This deliberately matches the evaluator in the research repository. It is
    not a boundary Jaccard and repeated tokens count once.
    """
    left = set(normalize_segmentation(reference).split("|"))
    right = set(normalize_segmentation(prediction).split("|"))
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True)
class Metrics:
    rows: int
    exact_match: float
    mean_jaccard: float


def evaluate_predictions(references: Iterable[object], predictions: Iterable[object]) -> Metrics:
    reference_list = list(references)
    prediction_list = list(predictions)
    if len(reference_list) != len(prediction_list):
        raise ValueError("references and predictions must have equal lengths")
    if not reference_list:
        raise ValueError("cannot evaluate an empty prediction set")
    exact = sum(exact_match(gold, pred) for gold, pred in zip(reference_list, prediction_list, strict=True))
    jaccard = sum(
        jaccard_similarity(gold, pred)
        for gold, pred in zip(reference_list, prediction_list, strict=True)
    )
    rows = len(reference_list)
    return Metrics(rows=rows, exact_match=exact / rows, mean_jaccard=jaccard / rows)
