#!/usr/bin/env python3
"""Verify locally acquired generator files against the publication manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/resources/generator_resource_manifest.csv"),
    )
    args = parser.parse_args()
    failures: list[str] = []
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = args.resource_dir / row["filename"]
            if not path.is_file():
                failures.append(f"MISSING  {path}")
                continue
            size = path.stat().st_size
            digest = sha256_file(path)
            if size != int(row["bytes"]) or digest != row["sha256"]:
                failures.append(
                    f"MISMATCH {path} (bytes={size}, sha256={digest})"
                )
    if failures:
        raise SystemExit("Generator resource verification failed:\n" + "\n".join(failures))
    print("Generator resources: ok")


if __name__ == "__main__":
    main()
