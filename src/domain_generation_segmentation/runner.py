"""Run the segmentation methods on domains stored in a CSV file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .evaluation import evaluate_predictions
from .segmenters.llm import hybrid_segment, segment_domains as segment_llm
from .segmenters.pos import PosSegmenter
from .segmenters.vbe import VBESegmenter
from .segmenters import viterbi


METHODS = {"viterbi", "fallback", "vbe", "pos", "gpt", "hybrid"}
METHOD_COLUMNS = {
    "viterbi": "viterbi_output", "fallback": "fallback_output", "vbe": "vbe_output",
    "pos": "pos_output", "gpt": "gpt_5_mini_output", "hybrid": "hybrid_output",
}


@dataclass(frozen=True)
class LlmOptions:
    model: str = "gpt-5-mini"
    batch_size: int = 100
    workers: int = 5


def _artifact_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_vbe_state(input_path: Path) -> Path:
    name = "human_state.json.gz" if "human" in input_path.stem else "synthetic_state.json.gz"
    return _artifact_root() / "artifacts" / "vbe" / name


def _load_cached_llm(path: Path, domains: list[str]) -> list[str | None]:
    cache = pd.read_csv(path, low_memory=False)
    column = next(
        (name for name in ("gpt_5_mini", "gpt_5_mini_output", "prediction") if name in cache.columns),
        None,
    )
    if "domain" not in cache.columns or column not in cache.columns:
        raise ValueError("LLM cache needs domain and gpt_5_mini/gpt_5_mini_output/prediction columns")
    if len(cache) >= len(domains) and cache["domain"].head(len(domains)).astype(str).tolist() == domains:
        # Rows outside the cached run are blank. Treat those cells as empty
        # strings rather than missing cache entries.
        return ["" if pd.isna(value) else str(value) for value in cache[column].head(len(domains))]
    grouped: dict[str, list[str | None]] = {}
    for domain, value in zip(cache["domain"].astype(str), cache[column], strict=True):
        grouped.setdefault(domain, []).append("" if pd.isna(value) else str(value))
    positions: dict[str, int] = {}
    output: list[str | None] = []
    for domain in domains:
        index = positions.get(domain, 0)
        values = grouped.get(domain, [])
        output.append(values[index] if index < len(values) else None)
        positions[domain] = index + 1
    return output


def viterbi_columns(domains: Sequence[str]) -> dict[str, list[object]]:
    values = [viterbi.segment_viterbi_domain(domain) for domain in domains]
    return {
        "viterbi_score": [value[0] for value in values],
        "viterbi_output": ["|".join(value[1]) for value in values],
    }


def fallback_results(
    domains: Sequence[str], confidence_rows: int = 0, **options: object
) -> list[viterbi.SegmentResult]:
    """Run the fallback segmenter, scoring confidence for the first rows only.

    The confidence pass does not change the chosen segmentation, so rows that
    no hybrid run needs can skip it.
    """
    return [
        viterbi.segment_domain_with_metadata(domain, compute_confidence=index < confidence_rows, **options)
        for index, domain in enumerate(domains)
    ]


def fallback_columns(results: Sequence[viterbi.SegmentResult]) -> dict[str, list[object]]:
    return {
        "fallback_score": [value.score for value in results],
        "fallback_confidence": [value.confidence_score for value in results],
        "fallback_output": ["|".join(value.segments) for value in results],
        "fallback_used_abbreviation": [value.used_abbreviation_prior for value in results],
        "fallback_used_fuzzy": [value.used_fuzzy for value in results],
    }


def gpt_predictions(domains: list[str], cached: list[str | None] | None, llm: LlmOptions) -> list[str]:
    if cached is None:
        return segment_llm(domains, model=llm.model, batch_size=llm.batch_size, max_workers=llm.workers)
    if any(value is None for value in cached):
        raise ValueError("the frozen LLM cache does not cover every requested row")
    return [str(value) for value in cached]


def hybrid_predictions(
    domains: list[str],
    baseline: list[str],
    confidence: list[float | None],
    cached: list[str | None] | None,
    threshold: float,
    llm: LlmOptions,
) -> tuple[list[str], list[str]]:
    if cached is None:
        return hybrid_segment(
            domains, baseline, confidence, threshold=threshold,
            model=llm.model, batch_size=llm.batch_size, max_workers=llm.workers,
        )
    predictions = list(baseline)
    sources = ["viterbi_fallback"] * len(domains)
    for index, score in enumerate(confidence):
        if score is None or score < threshold:
            if cached[index] is None:
                raise ValueError(f"LLM cache is missing routed row {index}")
            predictions[index] = str(cached[index])
            sources[index] = "gpt_5_mini"
    return predictions, sources


def pos_columns(domains: Sequence[str], model_path: Path) -> dict[str, list[object]]:
    segmenter = PosSegmenter(model_path)
    values = [segmenter.segment_domain(domain) for domain in domains]
    return {"pos_output": [value[0] for value in values], "pos_score": [value[1] for value in values]}


def vbe_predictions(domains: Sequence[str], state_path: Path, dataset_path: Path | None) -> list[str]:
    """Segment with a saved VBE state; ``dataset_path`` enables its SHA-256 check."""
    return VBESegmenter.from_json_gz(state_path, dataset_path=dataset_path).segment_many(domains)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["viterbi", "fallback", "vbe"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--unigrams", type=Path)
    parser.add_argument("--bigrams", type=Path)
    parser.add_argument("--vbe-state", type=Path)
    parser.add_argument("--pos-model", type=Path)
    parser.add_argument("--llm-cache", type=Path, help="Use frozen predictions instead of making paid API calls")
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--llm-batch-size", type=int, default=100)
    parser.add_argument("--llm-workers", type=int, default=5)
    parser.add_argument("--hybrid-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    llm = LlmOptions(args.llm_model, args.llm_batch_size, args.llm_workers)

    frame = pd.read_csv(args.input_csv, nrows=args.limit)
    if "domain" not in frame.columns:
        raise ValueError("input CSV must contain a domain column")
    domains = frame["domain"].astype(str).tolist()
    methods = set(args.methods)
    if methods & {"viterbi", "fallback", "hybrid"}:
        viterbi.load_counts(args.unigrams, args.bigrams)

    if "viterbi" in methods:
        for column, values in viterbi_columns(domains).items():
            frame[column] = values

    results = None
    if methods & {"fallback", "hybrid"}:
        results = fallback_results(domains, confidence_rows=len(domains) if "hybrid" in methods else 0)
        for column, values in fallback_columns(results).items():
            frame[column] = values

    cached = _load_cached_llm(args.llm_cache, domains) if args.llm_cache else None
    if "gpt" in methods:
        frame["gpt_5_mini_output"] = gpt_predictions(domains, cached, llm)

    if "hybrid" in methods:
        assert results is not None
        predictions, sources = hybrid_predictions(
            domains,
            ["|".join(value.segments) for value in results],
            [value.confidence_score for value in results],
            cached, args.hybrid_threshold, llm,
        )
        frame["hybrid_output"] = predictions
        frame["hybrid_source"] = sources

    # The 32 MB VBE archive becomes a much larger Python mapping in memory.
    # Drop the word counts and fallback details before loading it.
    results = None
    if methods & {"pos", "vbe"}:
        viterbi.clear_counts()

    if "pos" in methods:
        if args.pos_model is None:
            raise ValueError("--pos-model is required for POS segmentation")
        for column, values in pos_columns(domains, args.pos_model).items():
            frame[column] = values

    if "vbe" in methods:
        state_path = args.vbe_state or _default_vbe_state(args.input_csv)
        frame["vbe_output"] = vbe_predictions(
            domains, state_path, args.input_csv if args.limit is None else None
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    if "segmented" in frame.columns:
        rows = []
        for method, column in METHOD_COLUMNS.items():
            if column in frame:
                metric = evaluate_predictions(frame["segmented"], frame[column])
                rows.append({"method": method, **metric.__dict__})
        summary = args.output_csv.with_name(args.output_csv.stem + "_summary.csv")
        pd.DataFrame(rows).to_csv(summary, index=False)
        print(summary)
    print(args.output_csv)


if __name__ == "__main__":
    main()
