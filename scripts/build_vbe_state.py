#!/usr/bin/env python3
"""Build the saved VBE inference state for one evaluation CSV.

The state holds VBE values only for the character n-grams that segmenting
the CSV's domains can look up. Each n-gram count file is streamed once, so the
full dictionaries never have to fit in memory.

This follows the streaming builder that produced the published states.
Branching entropy is accumulated in file order as
``log(total) - sum(count * log(count)) / total``. The normalization
constants are the population mean and standard deviation of each n-gram
file's raw counts, computed as ``sqrt(E[x^2] - mean^2)``. Results agree with
``VBESegmenter(read_ngram_dictionaries(...))`` to within about 1e-8, but only
this procedure reproduces ``artifacts/vbe/*.json.gz`` exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd

from domain_generation_segmentation.segmenters.vbe import _letters, _pieces


FORMAT_VERSION = 1


@dataclass
class FileSummary:
    """What one ``{n}gram_dict.txt`` file contributes to the state."""

    n: int
    mean: float = 0.0
    std: float = 0.0
    present: set[str] = field(default_factory=set)
    forward_entropy: dict[str, float] = field(default_factory=dict)
    backward_entropy: dict[str, float] = field(default_factory=dict)


def queried_contexts(domains: list[str], max_context_length: int) -> dict[int, set[str]]:
    """Return every substring that ``VBESegmenter.segment`` can look up, by length.

    Length 0 holds the empty context, whose entropy the 1-gram VBE values need.
    """
    contexts: dict[int, set[str]] = {length: set() for length in range(max_context_length + 1)}
    contexts[0].add("")
    for domain in domains:
        for raw_piece in _pieces(domain):
            piece = _letters(raw_piece)
            for start in range(len(piece)):
                limit = min(max_context_length, len(piece) - start)
                for length in range(1, limit + 1):
                    contexts[length].add(piece[start : start + length])
    return contexts


def _entropies(totals: dict[str, int], weighted_logs: dict[str, float]) -> dict[str, float]:
    return {
        context: math.log(total) - weighted_logs[context] / total
        for context, total in totals.items()
        if total
    }


def summarize_file(
    path: Path,
    n: int,
    targets: set[str],
    contexts: set[str],
    charset: frozenset[str],
) -> FileSummary:
    """Stream one n-gram file.

    Args:
        path: The ``{n}gram_dict.txt`` file.
        n: The n-gram length of the file.
        targets: Queried n-grams whose presence in the file should be recorded.
            Leave this empty for the longest file, which has no VBE values.
        contexts: Queried ``(n - 1)``-character contexts whose forward and
            backward branching entropy this file determines.
        charset: Characters allowed to extend a context (the 1-gram keys).

    Returns:
        The file's count statistics, the present targets, and context entropies.
    """
    summary = FileSummary(n=n)
    rows = 0
    total = 0.0
    total_sq = 0.0
    f_totals: dict[str, int] = {}
    f_logs: dict[str, float] = {}
    b_totals: dict[str, int] = {}
    b_logs: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            ngram, raw_count = line.strip().split(": ", 1)
            count = int(raw_count)
            rows += 1
            total += count
            total_sq += count * count
            if ngram in targets:
                summary.present.add(ngram)
            prefix = ngram[:-1]
            if prefix in contexts and ngram[-1] in charset:
                f_totals[prefix] = f_totals.get(prefix, 0) + count
                f_logs[prefix] = f_logs.get(prefix, 0.0) + count * math.log(count)
            suffix = ngram[1:]
            if suffix in contexts and ngram[0] in charset:
                b_totals[suffix] = b_totals.get(suffix, 0) + count
                b_logs[suffix] = b_logs.get(suffix, 0.0) + count * math.log(count)
    summary.mean = total / rows if rows else 0.0
    variance = total_sq / rows - summary.mean * summary.mean if rows else 0.0
    summary.std = math.sqrt(max(variance, 0.0))
    summary.forward_entropy = _entropies(f_totals, f_logs)
    summary.backward_entropy = _entropies(b_totals, b_logs)
    return summary


def read_charset(path: Path) -> list[str]:
    """Return the 1-gram keys in file order."""
    with path.open(encoding="utf-8") as handle:
        return [line.strip().split(": ", 1)[0] for line in handle]


def build_state(
    counts_dir: Path,
    domains: list[str],
    max_n: int,
    workers: int,
) -> dict[str, object]:
    """Compute the filtered state for ``domains`` from the n-gram files."""
    contexts = queried_contexts(domains, max_n - 1)
    for length in range(1, max_n):
        print(f"  {len(contexts[length]):,} queried {length}-grams", flush=True)
    charset = read_charset(counts_dir / "1gram_dict.txt")
    allowed = frozenset(charset)

    summaries: dict[int, FileSummary] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(
                summarize_file,
                counts_dir / f"{n}gram_dict.txt",
                n,
                contexts[n] if n < max_n else set(),
                contexts[n - 1],
                allowed,
            )
            for n in range(max_n, 0, -1)
        ]
        for job in jobs:
            summary = job.result()
            summaries[summary.n] = summary
            print(
                f"  read {summary.n}-grams: {len(summary.present):,} queried present, "
                f"{len(summary.forward_entropy):,}/{len(summary.backward_entropy):,} "
                "forward/backward context entropies",
                flush=True,
            )

    vbe_dicts: dict[str, dict[str, dict[str, float]]] = {"forwards": {}, "backwards": {}}
    norm_constants: dict[str, dict[str, list[float]]] = {"forwards": {}, "backwards": {}}
    for length in range(1, max_n):
        # Entropy of a length-L context comes from the (L + 1)-gram file.
        current = summaries[length + 1]
        previous = summaries[length]
        present = summaries[length].present
        vbe_dicts["forwards"][str(length)] = {
            ngram: current.forward_entropy.get(ngram, 0.0)
            - previous.forward_entropy.get(ngram[:-1], 0.0)
            for ngram in present
        }
        vbe_dicts["backwards"][str(length)] = {
            ngram: current.backward_entropy.get(ngram, 0.0)
            - previous.backward_entropy.get(ngram[1:], 0.0)
            for ngram in present
        }
        constants = [summaries[length].mean, summaries[length].std]
        norm_constants["forwards"][str(length)] = list(constants)
        norm_constants["backwards"][str(length)] = list(constants)
    return {
        "charset": charset,
        "max_length": max_n,
        "norm_constants": norm_constants,
        "vbe_dicts": vbe_dicts,
    }


def write_payload(path: Path, payload: dict[str, object]) -> None:
    """Write sorted, compact JSON with a fixed gzip timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
    ) as handle:
        handle.write(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts_dir", type=Path, help="directory with {n}gram_dict.txt files")
    parser.add_argument("dataset_csv", type=Path, help="evaluation CSV with a domain column")
    parser.add_argument("output", type=Path, help="destination .json.gz state file")
    parser.add_argument("--name", required=True, help="dataset name stored in the state")
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    frame = pd.read_csv(args.dataset_csv)
    if "domain" not in frame.columns:
        raise ValueError("dataset CSV must contain a domain column")
    domains = frame["domain"].astype(str).tolist()
    print(f"{len(domains):,} domains from {args.dataset_csv}", flush=True)
    state = build_state(args.counts_dir, domains, args.max_n, args.workers)
    payload = {
        "dataset": args.name,
        "dataset_sha256": hashlib.sha256(args.dataset_csv.read_bytes()).hexdigest(),
        "format_version": FORMAT_VERSION,
        "max_n": args.max_n,
        "state": state,
    }
    write_payload(args.output, payload)
    print(f"wrote {args.output} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
