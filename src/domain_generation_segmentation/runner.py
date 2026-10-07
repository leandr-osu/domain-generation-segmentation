"""Run any of the six publication segmentation methods on a domain CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .evaluation import evaluate_predictions
from .segmenters.llm import hybrid_segment, segment_domains as segment_llm
from .segmenters.pos import PosSegmenter
from .segmenters.vbe import VBESegmenter
from .segmenters import viterbi


METHODS = {"viterbi", "fallback", "vbe", "pos", "gpt", "hybrid"}


def _artifact_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_vbe_state(input_path: Path) -> Path:
    name = "human_state.json.gz" if "human" in input_path.stem else "synthetic_state.json.gz"
    return _artifact_root() / "artifacts" / "vbe" / name


def _load_cached_llm(path: Path, domains: list[str]) -> list[str | None]:
    cache = pd.read_csv(path, low_memory=False)
    column = "gpt_5_mini" if "gpt_5_mini" in cache.columns else "prediction"
    if "domain" not in cache.columns or column not in cache.columns:
        raise ValueError("LLM cache needs domain and gpt_5_mini/prediction columns")
    if len(cache) >= len(domains) and cache["domain"].head(len(domains)).astype(str).tolist() == domains:
        # A blank retained cell is an observed blank API output, not a missing
        # cache entry. Keeping it as "" is necessary to replay the three blank
        # outputs in the 10,000-row synthetic GPT evaluation exactly.
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

    frame = pd.read_csv(args.input_csv, nrows=args.limit)
    if "domain" not in frame.columns:
        raise ValueError("input CSV must contain a domain column")
    domains = frame["domain"].astype(str).tolist()
    methods = set(args.methods)
    if methods & {"viterbi", "fallback", "hybrid"}:
        viterbi.load_counts(args.unigrams, args.bigrams)

    if "viterbi" in methods:
        values = [viterbi.segment_viterbi_domain(domain) for domain in domains]
        frame["viterbi_score"] = [value[0] for value in values]
        frame["viterbi_output"] = ["|".join(value[1]) for value in values]
        del values

    fallback_results = None
    if methods & {"fallback", "hybrid"}:
        fallback_results = [
            viterbi.segment_domain_with_metadata(domain, compute_confidence="hybrid" in methods)
            for domain in domains
        ]
        frame["fallback_score"] = [value.score for value in fallback_results]
        frame["fallback_confidence"] = [value.confidence_score for value in fallback_results]
        frame["fallback_output"] = ["|".join(value.segments) for value in fallback_results]
        frame["fallback_used_abbreviation"] = [value.used_abbreviation_prior for value in fallback_results]
        frame["fallback_used_fuzzy"] = [value.used_fuzzy for value in fallback_results]

    cached = _load_cached_llm(args.llm_cache, domains) if args.llm_cache else None
    if "gpt" in methods:
        if cached is None:
            predictions = segment_llm(
                domains, model=args.llm_model, batch_size=args.llm_batch_size, max_workers=args.llm_workers
            )
        else:
            if any(value is None for value in cached):
                raise ValueError("the frozen LLM cache does not cover every requested row")
            predictions = [str(value) for value in cached]
        frame["gpt_5_mini_output"] = predictions

    if "hybrid" in methods:
        assert fallback_results is not None
        baseline = ["|".join(value.segments) for value in fallback_results]
        confidence = [value.confidence_score for value in fallback_results]
        if cached is None:
            predictions, sources = hybrid_segment(
                domains, baseline, confidence, threshold=args.hybrid_threshold,
                model=args.llm_model, batch_size=args.llm_batch_size, max_workers=args.llm_workers,
            )
        else:
            predictions = baseline.copy()
            sources = ["viterbi_fallback"] * len(domains)
            for index, score in enumerate(confidence):
                if score is None or score < args.hybrid_threshold:
                    if cached[index] is None:
                        raise ValueError(f"LLM cache is missing routed row {index}")
                    predictions[index] = str(cached[index])
                    sources[index] = "gpt_5_mini"
        frame["hybrid_output"] = predictions
        frame["hybrid_source"] = sources

    # Loading the synthetic VBE state expands a 32 MB gzip file into a large
    # Python mapping. Release the word-count dictionaries and rich fallback
    # records first so a combined 100,000-row run remains within ordinary RAM.
    fallback_results = None
    if methods & {"pos", "vbe"}:
        viterbi.clear_counts()

    if "pos" in methods:
        if args.pos_model is None:
            raise ValueError("--pos-model is required for POS segmentation")
        segmenter = PosSegmenter(args.pos_model)
        values = [segmenter.segment_domain(domain) for domain in domains]
        frame["pos_output"] = [value[0] for value in values]
        frame["pos_score"] = [value[1] for value in values]
        del values, segmenter

    if "vbe" in methods:
        state_path = args.vbe_state or _default_vbe_state(args.input_csv)
        segmenter = VBESegmenter.from_json_gz(state_path, dataset_path=args.input_csv if args.limit is None else None)
        frame["vbe_output"] = segmenter.segment_many(domains)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    if "segmented" in frame.columns:
        rows = []
        for method, column in {
            "viterbi": "viterbi_output", "fallback": "fallback_output", "vbe": "vbe_output",
            "pos": "pos_output", "gpt": "gpt_5_mini_output", "hybrid": "hybrid_output",
        }.items():
            if column in frame:
                metric = evaluate_predictions(frame["segmented"], frame[column])
                rows.append({"method": method, **metric.__dict__})
        summary = args.output_csv.with_name(args.output_csv.stem + "_summary.csv")
        pd.DataFrame(rows).to_csv(summary, index=False)
        print(summary)
    print(args.output_csv)


if __name__ == "__main__":
    main()
