"""Publication Viterbi segmenters.

This module contains both the baseline Viterbi implementation and the tuned
abbreviation/fuzzy fallback variant.  It only needs the WordSegment-style
``unigrams.txt`` and ``bigrams.txt`` count files distributed with this artifact.

The implementation originated from the standalone script used for the final
experiments. Keeping the scoring and tie-breaking code together prevents the
baseline and fallback paths from silently loading different dictionaries.
"""

from __future__ import annotations

import argparse
import csv
import re
from functools import lru_cache, wraps
from math import log10
from pathlib import Path
from string import ascii_lowercase
from typing import NamedTuple


TOTAL_WORD_COUNT = 1024908267229.0

BASE_ABBR_LOG_PRIORS = {
    2: -9.5,
    3: -10.0,
    4: -10.75,
    5: -11.5,
}

# Best balanced-exact settings from wordsegment_joint_hyperparameter_tuning.csv.
DEFAULT_BOUNDARY_PENALTY = 0.35
DEFAULT_ABBR_PRIOR_OFFSET = 0.25
DEFAULT_EDIT_PENALTY = 2.5
DEFAULT_ACCEPT_MARGIN = 1.0

DEFAULT_SEGMENT_REDUCTION_MARGIN = 2.0
DEFAULT_LOW_VOWEL_RATIO = 0.34
DEFAULT_LOW_VOWEL_BONUS = 0.35
DEFAULT_NO_VOWEL_BONUS = 0.75
DEFAULT_MIN_FUZZY_LENGTH = 5
DEFAULT_MAX_FUZZY_CANDIDATES = 12
DEFAULT_MIN_CANDIDATE_COUNT = 10
DEFAULT_MAX_KNOWN_COUNT_FOR_FUZZY = 250_000.0

VOWELS = set("aeiou")
DOMAIN_SEPARATOR_PATTERN = re.compile(r"[-.]")


class CandidateScore(NamedTuple):
    log_score: float
    used_abbreviation_prior: bool


class Candidate(NamedTuple):
    observed: str
    matched: str
    edit_distance: int
    used_fuzzy: bool


class SegmentResult(NamedTuple):
    score: float | None
    confidence_score: float | None
    segments: list[str]
    used_abbreviation_prior: bool
    used_fuzzy: bool
    matched_words: list[str]


def _candidate_count_paths(filename: str) -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[2]
    workspace_dir = project_dir.parent
    cwd = Path.cwd()
    return [
        project_dir / "data" / "resources" / "word_counts" / filename,
        script_dir / filename,
        project_dir / "data" / "dictionaries" / filename,
        project_dir / "data" / filename,
        workspace_dir / "data" / filename,
        workspace_dir / "scripts" / filename,
        cwd / "data" / filename,
        cwd / "scripts" / filename,
        cwd / filename,
    ]


def resolve_counts_file(filename: str, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Could not locate {filename}: {path}")
        return path

    for path in _candidate_count_paths(filename):
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in _candidate_count_paths(filename))
    raise FileNotFoundError(f"Could not locate {filename}. Searched: {searched}")


def parse_count_file(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        return {word: float(number) for word, number in (line.split("\t") for line in handle)}


UNIGRAM_COUNTS: dict[str, float] = {}
BIGRAM_COUNTS: dict[str, float] = {}


def load_counts(unigrams_path: Path | None = None, bigrams_path: Path | None = None) -> None:
    """Load unigram and bigram counts.

    Calling this explicitly is optional. The segmenter lazily loads the default
    files the first time it is used.
    """
    global UNIGRAM_COUNTS, BIGRAM_COUNTS
    UNIGRAM_COUNTS = parse_count_file(resolve_counts_file("unigrams.txt", unigrams_path))
    BIGRAM_COUNTS = parse_count_file(resolve_counts_file("bigrams.txt", bigrams_path))
    _edit_distance_one_candidates.cache_clear()


def clear_counts() -> None:
    """Release count dictionaries and fuzzy caches between large method runs."""
    global UNIGRAM_COUNTS, BIGRAM_COUNTS
    UNIGRAM_COUNTS = {}
    BIGRAM_COUNTS = {}
    _edit_distance_one_candidates.cache_clear()


def ensure_counts_loaded() -> None:
    if not UNIGRAM_COUNTS or not BIGRAM_COUNTS:
        load_counts()


def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


def divide(text: str, limit: int = 24):
    for position in range(1, min(len(text), limit) + 1):
        yield text[:position], text[position:]


def default_abbr_log_priors(abbr_prior_offset: float = DEFAULT_ABBR_PRIOR_OFFSET) -> dict[int, float]:
    return {length: score + abbr_prior_offset for length, score in BASE_ABBR_LOG_PRIORS.items()}


def parse_abbr_log_priors(value: str | None) -> dict[int, float] | None:
    """Parse values like '2:-9.5,3:-10,4:-10.75,5:-11.5'."""
    if value is None or value.strip() == "":
        return None

    priors: dict[int, float] = {}
    for part in value.split(","):
        length_text, score_text = part.split(":", maxsplit=1)
        priors[int(length_text.strip())] = float(score_text.strip())
    return priors


def current_oov_log_score(word: str) -> float:
    return log10(10.0 / (TOTAL_WORD_COUNT * 10 ** len(word)))


def log10_sum_exp(values: list[float]) -> float:
    """Return log10(sum(10**value)) without numerical underflow."""
    maximum = max(values)
    return maximum + log10(sum(10 ** (value - maximum) for value in values))


def abbreviation_shape_bonus(
    word: str,
    *,
    low_vowel_ratio: float,
    low_vowel_bonus: float,
    no_vowel_bonus: float,
) -> float:
    if not word.isalpha():
        return 0.0

    vowel_count = sum(1 for char in word if char in VOWELS)
    if vowel_count == 0:
        return no_vowel_bonus
    if vowel_count / len(word) <= low_vowel_ratio:
        return low_vowel_bonus
    return 0.0


def abbreviation_prior_log_score(
    word: str,
    *,
    abbr_log_priors: dict[int, float],
    abbr_min_length: int,
    abbr_max_length: int,
    low_vowel_ratio: float,
    low_vowel_bonus: float,
    no_vowel_bonus: float,
) -> CandidateScore | None:
    if not word.isalpha() or word in UNIGRAM_COUNTS:
        return None
    if not abbr_min_length <= len(word) <= abbr_max_length:
        return None

    base = abbr_log_priors.get(len(word))
    if base is None:
        return None

    return CandidateScore(
        base
        + abbreviation_shape_bonus(
            word,
            low_vowel_ratio=low_vowel_ratio,
            low_vowel_bonus=low_vowel_bonus,
            no_vowel_bonus=no_vowel_bonus,
        ),
        True,
    )


def unigram_log_score(
    word: str,
    *,
    abbr_log_priors: dict[int, float],
    abbr_min_length: int,
    abbr_max_length: int,
    low_vowel_ratio: float,
    low_vowel_bonus: float,
    no_vowel_bonus: float,
) -> CandidateScore:
    if word in UNIGRAM_COUNTS:
        return CandidateScore(log10(UNIGRAM_COUNTS[word] / TOTAL_WORD_COUNT), False)

    abbreviation_score = abbreviation_prior_log_score(
        word,
        abbr_log_priors=abbr_log_priors,
        abbr_min_length=abbr_min_length,
        abbr_max_length=abbr_max_length,
        low_vowel_ratio=low_vowel_ratio,
        low_vowel_bonus=low_vowel_bonus,
        no_vowel_bonus=no_vowel_bonus,
    )
    if abbreviation_score is not None:
        return abbreviation_score

    return CandidateScore(current_oov_log_score(word), False)


def token_log_score(
    word: str,
    prev: str,
    *,
    alpha: float,
    abbr_log_priors: dict[int, float],
    abbr_min_length: int,
    abbr_max_length: int,
    low_vowel_ratio: float,
    low_vowel_bonus: float,
    no_vowel_bonus: float,
) -> CandidateScore:
    bigram = f"{prev} {word}"
    if bigram in BIGRAM_COUNTS and prev in UNIGRAM_COUNTS:
        prev_probability = UNIGRAM_COUNTS[prev] / TOTAL_WORD_COUNT
        probability = BIGRAM_COUNTS[bigram] / TOTAL_WORD_COUNT / prev_probability
        return CandidateScore(log10(probability), False)

    score = unigram_log_score(
        word,
        abbr_log_priors=abbr_log_priors,
        abbr_min_length=abbr_min_length,
        abbr_max_length=abbr_max_length,
        low_vowel_ratio=low_vowel_ratio,
        low_vowel_bonus=low_vowel_bonus,
        no_vowel_bonus=no_vowel_bonus,
    )
    return CandidateScore(log10(alpha) + score.log_score, score.used_abbreviation_prior)


def word_score(word: str, prev: str | None = None, alpha: float = 1.0) -> float:
    if prev is None:
        if word in UNIGRAM_COUNTS:
            return UNIGRAM_COUNTS[word] / TOTAL_WORD_COUNT
        return 10.0 / (TOTAL_WORD_COUNT * 10 ** len(word))

    bigram = f"{prev} {word}"
    if bigram in BIGRAM_COUNTS and prev in UNIGRAM_COUNTS:
        return BIGRAM_COUNTS[bigram] / TOTAL_WORD_COUNT / word_score(prev)
    return alpha * word_score(word)


def segment_viterbi_piece(
    text: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
    limit: int = 24,
) -> tuple[float | None, list[str]]:
    """Segment one separator-free string with the publication baseline."""
    ensure_counts_loaded()

    @memoize
    def search(remaining: str, prev: str = "<S>") -> tuple[float, list[str]]:
        if remaining == "":
            return 0.0, []

        def candidates():
            for prefix, suffix in divide(remaining, limit=limit):
                suffix_score, suffix_words = search(suffix, prefix)
                yield log10(word_score(prefix, prev, alpha=alpha)) + suffix_score, [prefix] + suffix_words

        # Tuple comparison is intentional: it preserves the original tie break.
        return max(candidates())

    score, words = search(str(text).lower())
    return (None, [str(text)]) if score < threshold else (score, words)


def segment_viterbi_domain(
    domain: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
) -> tuple[float | None, list[str]]:
    """Split dots/hyphens first, then apply the baseline to each piece."""
    pieces = split_domain_separators(str(domain))
    if not pieces:
        return None, []
    total = 0.0
    words: list[str] = []
    has_score = False
    for piece in pieces:
        score, piece_words = segment_viterbi_piece(piece, threshold=threshold, alpha=alpha)
        words.extend(piece_words)
        if score is not None:
            total += score
            has_score = True
    return total if has_score else None, words


def _replace_one(token: str) -> set[str]:
    candidates: set[str] = set()
    for index, original in enumerate(token):
        for char in ascii_lowercase:
            if char != original:
                candidates.add(token[:index] + char + token[index + 1 :])
    return candidates


def _delete_one(token: str) -> set[str]:
    return {token[:index] + token[index + 1 :] for index in range(len(token))}


def _insert_one(token: str) -> set[str]:
    candidates: set[str] = set()
    for index in range(len(token) + 1):
        for char in ascii_lowercase:
            candidates.add(token[:index] + char + token[index:])
    return candidates


def _transpose_one(token: str) -> set[str]:
    return {
        token[:index] + token[index + 1] + token[index] + token[index + 2 :]
        for index in range(len(token) - 1)
        if token[index] != token[index + 1]
    }


@lru_cache(maxsize=100_000)
def _edit_distance_one_candidates(token: str, max_candidates: int, min_count: int) -> tuple[str, ...]:
    """Return known words within one Damerau-Levenshtein edit of token."""
    if not token.isalpha():
        return ()

    generated = set()
    generated.update(_replace_one(token))
    generated.update(_delete_one(token))
    generated.update(_insert_one(token))
    generated.update(_transpose_one(token))

    candidates = [
        candidate
        for candidate in generated
        if candidate in UNIGRAM_COUNTS and UNIGRAM_COUNTS[candidate] >= min_count
    ]
    candidates.sort(key=lambda word: UNIGRAM_COUNTS[word], reverse=True)
    return tuple(candidates[:max_candidates])


def candidate_prefixes(
    prefix: str,
    *,
    min_fuzzy_length: int,
    max_fuzzy_candidates: int,
    min_candidate_count: int,
    fuzzy_known_words: bool,
    max_known_count_for_fuzzy: float | None,
) -> list[Candidate]:
    candidates = [Candidate(prefix, prefix, 0, False)]
    if len(prefix) < min_fuzzy_length:
        return candidates

    prefix_count = UNIGRAM_COUNTS.get(prefix)
    known_word_is_protected = (
        prefix_count is not None
        and not fuzzy_known_words
        and (
            max_known_count_for_fuzzy is None
            or prefix_count > max_known_count_for_fuzzy
        )
    )
    if known_word_is_protected:
        return candidates

    candidates.extend(
        Candidate(prefix, matched, 1, True)
        for matched in _edit_distance_one_candidates(
            prefix,
            max_candidates=max_fuzzy_candidates,
            min_count=min_candidate_count,
        )
    )
    return candidates


def abbreviation_segment_with_metadata(
    text: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
    *,
    boundary_penalty: float = DEFAULT_BOUNDARY_PENALTY,
    abbr_log_priors: dict[int, float] | None = None,
    abbr_prior_offset: float = DEFAULT_ABBR_PRIOR_OFFSET,
    abbr_min_length: int = 2,
    abbr_max_length: int = 5,
    low_vowel_ratio: float = DEFAULT_LOW_VOWEL_RATIO,
    low_vowel_bonus: float = DEFAULT_LOW_VOWEL_BONUS,
    no_vowel_bonus: float = DEFAULT_NO_VOWEL_BONUS,
    limit: int = 24,
) -> tuple[float | None, list[str], bool]:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    ensure_counts_loaded()
    if abbr_log_priors is None:
        abbr_log_priors = default_abbr_log_priors(abbr_prior_offset)

    @memoize
    def search(remaining: str, prev: str = "<S>") -> tuple[float, list[str], bool]:
        if remaining == "":
            return 0.0, [], False

        def candidates():
            for prefix, suffix in divide(remaining, limit=limit):
                prefix_score = token_log_score(
                    prefix,
                    prev,
                    alpha=alpha,
                    abbr_log_priors=abbr_log_priors,
                    abbr_min_length=abbr_min_length,
                    abbr_max_length=abbr_max_length,
                    low_vowel_ratio=low_vowel_ratio,
                    low_vowel_bonus=low_vowel_bonus,
                    no_vowel_bonus=no_vowel_bonus,
                )
                segment_penalty = 0.0 if prev == "<S>" else boundary_penalty
                suffix_score, suffix_words, suffix_used_abbreviation = search(suffix, prefix)
                yield (
                    prefix_score.log_score - segment_penalty + suffix_score,
                    [prefix] + suffix_words,
                    prefix_score.used_abbreviation_prior or suffix_used_abbreviation,
                )

        return max(candidates(), key=lambda item: item[0])

    result_score, result_words, used_abbreviation_prior = search(text)
    if result_score < threshold:
        return None, [text], used_abbreviation_prior
    return result_score, result_words, used_abbreviation_prior


def segment_with_metadata(
    text: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
    *,
    boundary_penalty: float = DEFAULT_BOUNDARY_PENALTY,
    abbr_log_priors: dict[int, float] | None = None,
    abbr_prior_offset: float = DEFAULT_ABBR_PRIOR_OFFSET,
    abbr_min_length: int = 2,
    abbr_max_length: int = 5,
    low_vowel_ratio: float = DEFAULT_LOW_VOWEL_RATIO,
    low_vowel_bonus: float = DEFAULT_LOW_VOWEL_BONUS,
    no_vowel_bonus: float = DEFAULT_NO_VOWEL_BONUS,
    edit_penalty: float = DEFAULT_EDIT_PENALTY,
    accept_margin: float = DEFAULT_ACCEPT_MARGIN,
    segment_reduction_margin: float = DEFAULT_SEGMENT_REDUCTION_MARGIN,
    min_fuzzy_length: int = DEFAULT_MIN_FUZZY_LENGTH,
    max_fuzzy_candidates: int = DEFAULT_MAX_FUZZY_CANDIDATES,
    min_candidate_count: int = DEFAULT_MIN_CANDIDATE_COUNT,
    fuzzy_known_words: bool = False,
    max_known_count_for_fuzzy: float | None = DEFAULT_MAX_KNOWN_COUNT_FOR_FUZZY,
    compute_confidence: bool = True,
    limit: int = 24,
) -> SegmentResult:
    """Segment with abbreviation-aware exact scoring plus conservative fuzzy fallback."""
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    ensure_counts_loaded()
    if abbr_log_priors is None:
        abbr_log_priors = default_abbr_log_priors(abbr_prior_offset)

    exact_score, exact_words, exact_used_abbreviation = abbreviation_segment_with_metadata(
        text,
        threshold=float("-inf"),
        alpha=alpha,
        boundary_penalty=boundary_penalty,
        abbr_log_priors=abbr_log_priors,
        abbr_min_length=abbr_min_length,
        abbr_max_length=abbr_max_length,
        low_vowel_ratio=low_vowel_ratio,
        low_vowel_bonus=low_vowel_bonus,
        no_vowel_bonus=no_vowel_bonus,
        limit=limit,
    )
    exact_score_value = exact_score if exact_score is not None else float("-inf")

    @memoize
    def search(
        remaining: str,
        prev_matched: str = "<S>",
    ) -> tuple[float, float | None, list[str], list[str], bool, bool]:
        if remaining == "":
            return 0.0, 0.0 if compute_confidence else None, [], [], False, False

        best_candidates: list[tuple[float, list[str], list[str], bool, bool]] = []
        partition_terms: list[float] = []
        for prefix, suffix in divide(remaining, limit=limit):
            for candidate in candidate_prefixes(
                prefix,
                min_fuzzy_length=min_fuzzy_length,
                max_fuzzy_candidates=max_fuzzy_candidates,
                min_candidate_count=min_candidate_count,
                fuzzy_known_words=fuzzy_known_words,
                max_known_count_for_fuzzy=max_known_count_for_fuzzy,
            ):
                if candidate.used_fuzzy:
                    prefix_score = log10(word_score(candidate.matched, prev_matched, alpha=alpha))
                    prefix_score -= edit_penalty * candidate.edit_distance
                    used_abbreviation = False
                else:
                    scored = token_log_score(
                        candidate.observed,
                        prev_matched,
                        alpha=alpha,
                        abbr_log_priors=abbr_log_priors,
                        abbr_min_length=abbr_min_length,
                        abbr_max_length=abbr_max_length,
                        low_vowel_ratio=low_vowel_ratio,
                        low_vowel_bonus=low_vowel_bonus,
                        no_vowel_bonus=no_vowel_bonus,
                    )
                    prefix_score = scored.log_score
                    used_abbreviation = scored.used_abbreviation_prior

                segment_penalty = 0.0 if prev_matched == "<S>" else boundary_penalty
                (
                    suffix_score,
                    suffix_partition,
                    suffix_words,
                    suffix_matched,
                    suffix_used_abbr,
                    suffix_used_fuzzy,
                ) = search(suffix, candidate.matched)
                path_prefix_score = prefix_score - segment_penalty
                best_candidates.append(
                    (
                        path_prefix_score + suffix_score,
                        [candidate.observed] + suffix_words,
                        [candidate.matched] + suffix_matched,
                        used_abbreviation or suffix_used_abbr,
                        candidate.used_fuzzy or suffix_used_fuzzy,
                    )
                )
                if compute_confidence:
                    if suffix_partition is None:
                        raise RuntimeError("Missing suffix partition while computing confidence.")
                    partition_terms.append(path_prefix_score + suffix_partition)

        best = max(best_candidates, key=lambda item: item[0])
        partition = log10_sum_exp(partition_terms) if compute_confidence else None
        return best[0], partition, *best[1:]

    (
        fuzzy_score,
        fuzzy_partition,
        fuzzy_words,
        matched_words,
        used_abbreviation_prior,
        used_fuzzy,
    ) = search(text)
    required_margin = accept_margin
    if len(fuzzy_words) < len(exact_words):
        required_margin = max(required_margin, segment_reduction_margin)

    if used_fuzzy and fuzzy_score >= exact_score_value + required_margin:
        if fuzzy_score < threshold:
            return SegmentResult(
                None,
                None,
                [text],
                used_abbreviation_prior=used_abbreviation_prior,
                used_fuzzy=True,
                matched_words=matched_words,
            )
        return SegmentResult(
            fuzzy_score,
            (
                min(1.0, max(0.0, 10 ** (fuzzy_score - fuzzy_partition)))
                if fuzzy_partition is not None
                else None
            ),
            fuzzy_words,
            used_abbreviation_prior=used_abbreviation_prior,
            used_fuzzy=True,
            matched_words=matched_words,
        )

    if exact_score is None or exact_score < threshold:
        return SegmentResult(
            None,
            None,
            [text],
            used_abbreviation_prior=exact_used_abbreviation,
            used_fuzzy=False,
            matched_words=exact_words,
        )
    return SegmentResult(
        exact_score,
        (
            min(1.0, max(0.0, 10 ** (exact_score - fuzzy_partition)))
            if fuzzy_partition is not None
            else None
        ),
        exact_words,
        used_abbreviation_prior=exact_used_abbreviation,
        used_fuzzy=False,
        matched_words=exact_words,
    )


def segment(
    text: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
    **kwargs,
) -> tuple[float | None, list[str]]:
    result = segment_with_metadata(text, threshold=threshold, alpha=alpha, **kwargs)
    return result.score, result.segments


def split_domain_separators(text: str) -> list[str]:
    return [piece for piece in DOMAIN_SEPARATOR_PATTERN.split(text) if piece]


def segment_domain_with_metadata(
    domain: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
    **kwargs,
) -> SegmentResult:
    pieces = split_domain_separators(str(domain))
    if not pieces:
        return SegmentResult(None, None, [], False, False, [])

    score_sum = 0.0
    has_score = False
    confidence_product = 1.0
    has_confidence = False
    all_segments: list[str] = []
    matched_words: list[str] = []
    used_abbreviation_prior = False
    used_fuzzy = False

    for piece in pieces:
        result = segment_with_metadata(piece, threshold=threshold, alpha=alpha, **kwargs)
        all_segments.extend(result.segments)
        matched_words.extend(result.matched_words)
        used_abbreviation_prior = used_abbreviation_prior or result.used_abbreviation_prior
        used_fuzzy = used_fuzzy or result.used_fuzzy
        if result.score is not None:
            score_sum += result.score
            has_score = True
        if result.confidence_score is not None:
            confidence_product *= result.confidence_score
            has_confidence = True

    return SegmentResult(
        score_sum if has_score else None,
        confidence_product if has_confidence else None,
        all_segments,
        used_abbreviation_prior,
        used_fuzzy,
        matched_words,
    )


def segment_domain(
    domain: str,
    threshold: float = float("-inf"),
    alpha: float = 1.0,
    **kwargs,
) -> tuple[float | None, list[str]]:
    result = segment_domain_with_metadata(domain, threshold=threshold, alpha=alpha, **kwargs)
    return result.score, result.segments


def segment_many(words: list[str], **kwargs) -> list[tuple[float | None, float | None, str, bool, bool]]:
    results = []
    for word in words:
        result = segment_domain_with_metadata(word, **kwargs)
        results.append(
            (
                result.score,
                result.confidence_score,
                "|".join(result.segments),
                result.used_abbreviation_prior,
                result.used_fuzzy,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unigrams", type=Path, default=None, help="Path to unigrams.txt.")
    parser.add_argument("--bigrams", type=Path, default=None, help="Path to bigrams.txt.")
    parser.add_argument("--domain", help="Single domain/string to segment.")
    parser.add_argument("--input-csv", type=Path, help="Input CSV containing a column to segment.")
    parser.add_argument("--output-csv", type=Path, help="Output CSV for CSV mode.")
    parser.add_argument("--column", default="domain", help="Input CSV column to segment.")
    parser.add_argument("--limit", type=int, default=None, help="Optional CSV row limit.")
    parser.add_argument("--threshold", type=float, default=float("-inf"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--boundary-penalty", type=float, default=DEFAULT_BOUNDARY_PENALTY)
    parser.add_argument("--abbr-prior-offset", type=float, default=DEFAULT_ABBR_PRIOR_OFFSET)
    parser.add_argument(
        "--abbr-log-priors",
        default=None,
        help=(
            "Comma-separated log10 priors by abbreviation length. If set, this "
            "overrides --abbr-prior-offset."
        ),
    )
    parser.add_argument("--abbr-min-length", type=int, default=2)
    parser.add_argument("--abbr-max-length", type=int, default=5)
    parser.add_argument("--low-vowel-ratio", type=float, default=DEFAULT_LOW_VOWEL_RATIO)
    parser.add_argument("--low-vowel-bonus", type=float, default=DEFAULT_LOW_VOWEL_BONUS)
    parser.add_argument("--no-vowel-bonus", type=float, default=DEFAULT_NO_VOWEL_BONUS)
    parser.add_argument("--edit-penalty", type=float, default=DEFAULT_EDIT_PENALTY)
    parser.add_argument("--accept-margin", type=float, default=DEFAULT_ACCEPT_MARGIN)
    parser.add_argument("--segment-reduction-margin", type=float, default=DEFAULT_SEGMENT_REDUCTION_MARGIN)
    parser.add_argument("--min-fuzzy-length", type=int, default=DEFAULT_MIN_FUZZY_LENGTH)
    parser.add_argument("--max-fuzzy-candidates", type=int, default=DEFAULT_MAX_FUZZY_CANDIDATES)
    parser.add_argument("--min-candidate-count", type=int, default=DEFAULT_MIN_CANDIDATE_COUNT)
    parser.add_argument(
        "--max-known-count-for-fuzzy",
        type=float,
        default=DEFAULT_MAX_KNOWN_COUNT_FOR_FUZZY,
        help="Use -1 to disable fuzzy alternatives for all known substrings.",
    )
    parser.add_argument(
        "--fuzzy-known-words",
        action="store_true",
        help="Also consider fuzzy alternatives for substrings already in the unigram dictionary.",
    )
    parser.add_argument(
        "--no-confidence-score",
        action="store_true",
        help="Use the prior best-path-only search without computing a posterior confidence score.",
    )
    return parser


def kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    abbr_log_priors = parse_abbr_log_priors(args.abbr_log_priors)
    return {
        "threshold": args.threshold,
        "alpha": args.alpha,
        "boundary_penalty": args.boundary_penalty,
        "abbr_log_priors": abbr_log_priors,
        "abbr_prior_offset": args.abbr_prior_offset,
        "abbr_min_length": args.abbr_min_length,
        "abbr_max_length": args.abbr_max_length,
        "low_vowel_ratio": args.low_vowel_ratio,
        "low_vowel_bonus": args.low_vowel_bonus,
        "no_vowel_bonus": args.no_vowel_bonus,
        "edit_penalty": args.edit_penalty,
        "accept_margin": args.accept_margin,
        "segment_reduction_margin": args.segment_reduction_margin,
        "min_fuzzy_length": args.min_fuzzy_length,
        "max_fuzzy_candidates": args.max_fuzzy_candidates,
        "min_candidate_count": args.min_candidate_count,
        "fuzzy_known_words": args.fuzzy_known_words,
        "compute_confidence": not args.no_confidence_score,
        "max_known_count_for_fuzzy": (
            None if args.max_known_count_for_fuzzy < 0 else args.max_known_count_for_fuzzy
        ),
    }


def run_single_domain(args: argparse.Namespace) -> None:
    result = segment_domain_with_metadata(args.domain, **kwargs_from_args(args))
    print(
        "\t".join(
            [
                str(result.score),
                str(result.confidence_score),
                "|".join(result.segments),
                str(result.used_abbreviation_prior),
                str(result.used_fuzzy),
                "|".join(result.matched_words),
            ]
        )
    )


def run_csv(args: argparse.Namespace) -> None:
    if args.output_csv is None:
        raise SystemExit("--output-csv is required with --input-csv")

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{args.input_csv} has no header row.")
        if args.column not in reader.fieldnames:
            raise SystemExit(f"{args.input_csv} is missing column {args.column!r}.")
        rows = []
        for index, row in enumerate(reader):
            if args.limit is not None and index >= args.limit:
                break
            result = segment_domain_with_metadata(str(row[args.column]), **kwargs_from_args(args))
            row["result_score"] = "" if result.score is None else f"{result.score:.12g}"
            row["confidence_score"] = (
                "" if result.confidence_score is None else f"{result.confidence_score:.12g}"
            )
            row["segmentation"] = "|".join(result.segments)
            row["used_abbreviation_prior"] = str(result.used_abbreviation_prior)
            row["used_fuzzy"] = str(result.used_fuzzy)
            row["matched_words"] = "|".join(result.matched_words)
            rows.append(row)
        fieldnames = list(reader.fieldnames)

    for column in [
        "result_score",
        "confidence_score",
        "segmentation",
        "used_abbreviation_prior",
        "used_fuzzy",
        "matched_words",
    ]:
        if column not in fieldnames:
            fieldnames.append(column)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if bool(args.domain) == bool(args.input_csv):
        parser.error("Provide exactly one of --domain or --input-csv.")

    try:
        load_counts(args.unigrams, args.bigrams)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if args.domain:
        run_single_domain(args)
    else:
        run_csv(args)


if __name__ == "__main__":
    main()
