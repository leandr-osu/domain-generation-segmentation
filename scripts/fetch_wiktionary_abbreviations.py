#!/usr/bin/env python3
"""Fetch the three Wiktionary categories used for the abbreviation sampler."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import requests


API = "https://en.wiktionary.org/w/api.php"
CATEGORIES = (
    "Category:English_acronyms",
    "Category:English_initialisms",
    "Category:English_abbreviations",
)


def category_members(category: str, user_agent: str) -> set[str]:
    output: set[str] = set()
    continuation: str | None = None
    while True:
        parameters = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "format": "json",
        }
        if continuation:
            parameters["cmcontinue"] = continuation
        response = requests.get(
            API,
            params=parameters,
            headers={"User-Agent": user_agent},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        output.update(item["title"] for item in payload["query"]["categorymembers"])
        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            return output
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--user-agent",
        required=True,
        help="Descriptive user agent with contact information, as requested by Wikimedia.",
    )
    args = parser.parse_args()
    values: set[str] = set()
    for category in CATEGORIES:
        values.update(category_members(category, args.user_agent))
    values = {value for value in values if value.isalnum()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word", "weight"])
        writer.writerows((value, 1) for value in sorted(values))
    print(args.output)


if __name__ == "__main__":
    main()
