"""Verify frozen predictions and print the publication segmentation table."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from .evaluation import evaluate_predictions


COLUMN_BY_METHOD = {
    "viterbi": "viterbi",
    "viterbi_fallback": "viterbi_fallback",
    "entropy_vbe": "entropy_vbe",
    "pos": "pos",
    "gpt_5_mini": "gpt_5_mini",
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(base: Path) -> list[str]:
    failures: list[str] = []
    manifest = base / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = base / relative.strip().lstrip("*")
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, got {actual}")
    return failures


def compute_prediction_metrics(base: Path, expected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("synthetic", "human"):
        predictions = pd.read_csv(base / "data" / "reference" / f"{dataset}_predictions.csv.gz", low_memory=False)
        selected = expected[(expected["dataset"] == dataset) & (expected["verification"] == "frozen_predictions")]
        for record in selected.itertuples(index=False):
            column = COLUMN_BY_METHOD[record.method]
            subset = predictions.iloc[: int(record.evaluation_rows)]
            metrics = evaluate_predictions(subset["reference"], subset[column])
            rows.append(
                {
                    "method": record.method,
                    "dataset": dataset,
                    "rows": metrics.rows,
                    "computed_exact": metrics.exact_match,
                    "computed_jaccard": metrics.mean_jaccard,
                    "expected_exact": record.artifact_exact,
                    "expected_jaccard": record.artifact_jaccard,
                    "status": (
                        "ok"
                        if abs(metrics.exact_match - record.artifact_exact) < 1e-12
                        and abs(metrics.mean_jaccard - record.artifact_jaccard) < 1e-12
                        else "MISMATCH"
                    ),
                }
            )
    return pd.DataFrame(rows)


def publication_table(expected: pd.DataFrame) -> pd.DataFrame:
    order = ["viterbi", "viterbi_fallback", "entropy_vbe", "pos", "gpt_5_mini", "hybrid"]
    output = []
    for method in order:
        synthetic = expected[(expected.method == method) & (expected.dataset == "synthetic")].iloc[0]
        human = expected[(expected.method == method) & (expected.dataset == "human")].iloc[0]
        output.append(
            {
                "method": method,
                "synthetic_EM": f"{synthetic.paper_exact:.3f}",
                "synthetic_Jaccard": f"{synthetic.paper_jaccard:.3f}",
                "human_EM": f"{human.paper_exact:.3f}",
                "human_Jaccard": f"{human.paper_jaccard:.3f}",
            }
        )
    return pd.DataFrame(output)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--output", type=Path, help="optional CSV for recomputed metrics")
    args = parser.parse_args(argv)
    base = root()
    if not args.skip_checksums:
        failures = verify_checksums(base)
        if failures:
            raise SystemExit("Checksum verification failed:\n" + "\n".join(failures))
        print("Checksums: ok")

    expected = pd.read_csv(base / "data" / "reference" / "publication_metrics.csv")
    computed = compute_prediction_metrics(base, expected)
    print("\nFrozen prediction verification")
    print(computed.to_string(index=False))
    if (computed.status != "ok").any():
        raise SystemExit("one or more frozen prediction metrics changed")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        computed.to_csv(args.output, index=False)

    print("\nPublication table (recorded display precision)")
    print(publication_table(expected).to_string(index=False))
    print("\nHybrid rows are aggregate-only; see docs/reproducibility_notes.md.")


if __name__ == "__main__":
    main()
