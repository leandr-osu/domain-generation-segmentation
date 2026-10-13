"""Evaluate the segmentation methods on every dataset in a JSON configuration.

Relative paths in the configuration are resolved against the current working
directory. Each method runs on the first ``rows[method]`` rows of its dataset;
methods missing from ``rows`` are not run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import pandas as pd

from .evaluation import evaluate_predictions
from .runner import (
    METHOD_COLUMNS,
    LlmOptions,
    _artifact_root,
    _default_vbe_state,
    _load_cached_llm,
    fallback_columns,
    fallback_results,
    gpt_predictions,
    hybrid_predictions,
    pos_columns,
    vbe_predictions,
    viterbi_columns,
)
from .segmenters import viterbi


# Method names used in data/reference/publication_metrics.csv.
PUBLICATION_METHOD = {
    "viterbi": "viterbi",
    "fallback": "viterbi_fallback",
    "vbe": "entropy_vbe",
    "pos": "pos",
    "gpt": "gpt_5_mini",
    "hybrid": "hybrid",
}
CONFIG_KEYS = {"output_dir", "pos_model", "llm", "hybrid_threshold", "fallback", "datasets"}
DATASET_KEYS = {"input", "llm_cache", "rows"}
# Keyword arguments of the fallback segmenter that a configuration may set.
FALLBACK_KEYS = {
    name
    for name, parameter in inspect.signature(viterbi.segment_with_metadata).parameters.items()
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name != "compute_confidence"
}
TOLERANCE = 1e-12


@dataclass(frozen=True)
class Dataset:
    name: str
    input: Path
    llm_cache: Path | None
    rows: dict[str, int]


@dataclass(frozen=True)
class Config:
    output_dir: Path
    pos_model: Path | None
    llm: LlmOptions
    hybrid_threshold: float
    fallback: dict[str, Any]
    datasets: list[Dataset]


def _check_keys(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")


def _optional_path(value: str | None) -> Path | None:
    return None if value is None else Path(value)


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _check_keys("config", raw, CONFIG_KEYS)
    llm = raw.get("llm", {})
    _check_keys("llm", llm, {"model", "batch_size", "workers"})
    fallback = dict(raw.get("fallback", {}))
    _check_keys("fallback", fallback, FALLBACK_KEYS)
    if fallback.get("abbr_log_priors") is not None:
        # JSON object keys are strings; the segmenter expects integer lengths.
        fallback["abbr_log_priors"] = {int(key): float(value) for key, value in fallback["abbr_log_priors"].items()}

    datasets = []
    for name, spec in raw["datasets"].items():
        _check_keys(f"datasets.{name}", spec, DATASET_KEYS)
        if name == "metrics":
            raise ValueError("'metrics' is reserved for the metrics output file")
        rows = {method: int(count) for method, count in spec.get("rows", {}).items()}
        _check_keys(f"datasets.{name}.rows", rows, set(METHOD_COLUMNS))
        if any(count < 1 for count in rows.values()):
            raise ValueError(f"datasets.{name}.rows must be positive")
        datasets.append(Dataset(name, Path(spec["input"]), _optional_path(spec.get("llm_cache")), rows))

    return Config(
        output_dir=Path(raw.get("output_dir", "outputs")),
        pos_model=_optional_path(raw.get("pos_model")),
        llm=LlmOptions(**llm),
        hybrid_threshold=float(raw.get("hybrid_threshold", 0.1)),
        fallback=fallback,
        datasets=datasets,
    )


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def skip_reason(method: str, config: Config, dataset: Dataset) -> str | None:
    """Explain why an optional method cannot run here, or return None."""
    if method == "pos":
        if config.pos_model is None:
            return "no pos_model configured"
        if not config.pos_model.is_dir():
            return f"POS checkpoint not found at {config.pos_model}"
        if not _installed("torch", "transformers"):
            return "torch/transformers not installed (install the 'models' extra)"
    if method in {"gpt", "hybrid"} and dataset.llm_cache is None:
        if not _installed("openai"):
            return "no llm_cache and openai not installed (install the 'llm' extra)"
        if not os.environ.get("OPENAI_API_KEY"):
            return "no llm_cache and OPENAI_API_KEY is not set"
    return None


def _status(method: str, dataset: str, rows: int, exact: float, jaccard: float, expected: pd.DataFrame) -> dict[str, Any]:
    match = expected[(expected.method == PUBLICATION_METHOD[method]) & (expected.dataset == dataset)]
    if match.empty:
        return {"expected_exact": None, "expected_jaccard": None, "status": "no_reference"}
    record = match.iloc[0]
    if record.verification == "aggregate_only":
        status = "aggregate_only"
    elif (
        rows == int(record.evaluation_rows)
        and abs(exact - record.artifact_exact) < TOLERANCE
        and abs(jaccard - record.artifact_jaccard) < TOLERANCE
    ):
        status = "ok"
    else:
        status = "MISMATCH"
    return {"expected_exact": record.artifact_exact, "expected_jaccard": record.artifact_jaccard, "status": status}


def _assign(frame: pd.DataFrame, columns: dict[str, list[object]]) -> None:
    # Columns shorter than the frame are left blank below their last row.
    for column, values in columns.items():
        frame[column] = pd.Series(values, index=frame.index[: len(values)])


def evaluate_dataset(config: Config, dataset: Dataset, expected: pd.DataFrame) -> list[dict[str, Any]]:
    full = pd.read_csv(dataset.input, low_memory=False)
    if "domain" not in full.columns or "segmented" not in full.columns:
        raise ValueError(f"{dataset.input} must contain domain and segmented columns")
    rows = {method: min(count, len(full)) for method, count in dataset.rows.items()}
    skipped = {method: reason for method in rows if (reason := skip_reason(method, config, dataset))}
    for method, reason in skipped.items():
        print(f"[{dataset.name}] skipping {method}: {reason}", flush=True)
    active = {method: count for method, count in rows.items() if method not in skipped}

    total_rows = len(full)
    frame = full.head(max(active.values(), default=0)).copy()
    del full
    domains = frame["domain"].astype(str).tolist()

    def log(method: str, start: float) -> None:
        print(f"[{dataset.name}] {method}: {active[method]} rows in {time.perf_counter() - start:.1f}s", flush=True)

    counts_loaded = bool(active.keys() & {"viterbi", "fallback", "hybrid"})
    if counts_loaded:
        viterbi.load_counts()

    if "viterbi" in active:
        start = time.perf_counter()
        _assign(frame, viterbi_columns(domains[: active["viterbi"]]))
        log("viterbi", start)

    results = None
    if active.keys() & {"fallback", "hybrid"}:
        # One fallback pass serves both methods; confidence is only scored on
        # the hybrid rows because it never changes the chosen segmentation.
        start = time.perf_counter()
        results = fallback_results(
            domains[: max(active.get("fallback", 0), active.get("hybrid", 0))],
            confidence_rows=active.get("hybrid", 0),
            **config.fallback,
        )
        if "fallback" in active:
            _assign(frame, fallback_columns(results[: active["fallback"]]))
            log("fallback", start)

    cached = None
    llm_rows = max(active.get("gpt", 0), active.get("hybrid", 0))
    if llm_rows and dataset.llm_cache is not None:
        cached = _load_cached_llm(dataset.llm_cache, domains[:llm_rows])

    if "gpt" in active:
        start = time.perf_counter()
        count = active["gpt"]
        _assign(frame, {"gpt_5_mini_output": gpt_predictions(
            domains[:count], None if cached is None else cached[:count], config.llm
        )})
        log("gpt", start)

    if "hybrid" in active:
        assert results is not None
        start = time.perf_counter()
        count = active["hybrid"]
        predictions, sources = hybrid_predictions(
            domains[:count],
            ["|".join(value.segments) for value in results[:count]],
            [value.confidence_score for value in results[:count]],
            None if cached is None else cached[:count],
            config.hybrid_threshold,
            config.llm,
        )
        _assign(frame, {"hybrid_output": predictions, "hybrid_source": sources})
        log("hybrid", start)

    # Free the word counts before loading the large VBE state.
    results = None
    if counts_loaded:
        viterbi.clear_counts()

    if "pos" in active:
        start = time.perf_counter()
        assert config.pos_model is not None
        _assign(frame, pos_columns(domains[: active["pos"]], config.pos_model))
        log("pos", start)

    if "vbe" in active:
        start = time.perf_counter()
        count = active["vbe"]
        # The state's SHA-256 check only applies when the whole CSV is used.
        dataset_path = dataset.input if count == total_rows else None
        _assign(frame, {"vbe_output": vbe_predictions(
            domains[:count], _default_vbe_state(dataset.input), dataset_path
        )})
        log("vbe", start)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.output_dir / f"{dataset.name}.csv", index=False)

    metrics: list[dict[str, Any]] = []
    for method in METHOD_COLUMNS:
        if method not in rows:
            continue
        if method in skipped:
            metrics.append({
                "method": method, "dataset": dataset.name, "rows": rows[method], "exact": None,
                "jaccard": None, "expected_exact": None, "expected_jaccard": None, "status": "skipped",
            })
            continue
        count = active[method]
        result = evaluate_predictions(frame["segmented"].head(count), frame[METHOD_COLUMNS[method]].head(count))
        metrics.append({
            "method": method, "dataset": dataset.name, "rows": result.rows,
            "exact": result.exact_match, "jaccard": result.mean_jaccard,
            **_status(method, dataset.name, result.rows, result.exact_match, result.mean_jaccard, expected),
        })
    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON evaluation configuration")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    expected = pd.read_csv(_artifact_root() / "data" / "reference" / "publication_metrics.csv")

    rows: list[dict[str, Any]] = []
    for dataset in config.datasets:
        rows.extend(evaluate_dataset(config, dataset, expected))
    table = pd.DataFrame(
        rows,
        columns=["method", "dataset", "rows", "exact", "jaccard", "expected_exact", "expected_jaccard", "status"],
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.output_dir / "metrics.csv", index=False)
    print()
    print(table.to_string(index=False))
    if (table.status == "MISMATCH").any():
        raise SystemExit("one or more metrics differ from data/reference/publication_metrics.csv")


if __name__ == "__main__":
    main()
