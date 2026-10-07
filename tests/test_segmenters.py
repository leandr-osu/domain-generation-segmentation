from pathlib import Path

import pandas as pd

from domain_generation_segmentation.segmenters import viterbi
from domain_generation_segmentation.segmenters.vbe import VBESegmenter
from domain_generation_segmentation.runner import _load_cached_llm


ROOT = Path(__file__).resolve().parents[1]


def test_viterbi_and_fallback_fixture_rows():
    inputs = pd.read_csv(ROOT / "data/manual/human_domains.csv", nrows=5)
    expected = pd.read_csv(ROOT / "data/reference/human_predictions.csv.gz", nrows=5)
    viterbi.load_counts()
    baseline = ["|".join(viterbi.segment_viterbi_domain(value)[1]) for value in inputs.domain]
    fallback = [
        "|".join(viterbi.segment_domain_with_metadata(value, compute_confidence=False).segments)
        for value in inputs.domain
    ]
    assert baseline == expected.viterbi.tolist()
    assert fallback == expected.viterbi_fallback.tolist()


def test_safe_vbe_state_fixture_rows():
    inputs = pd.read_csv(ROOT / "data/manual/human_domains.csv", nrows=5)
    expected = pd.read_csv(ROOT / "data/reference/human_predictions.csv.gz", nrows=5)
    segmenter = VBESegmenter.from_json_gz(
        ROOT / "artifacts/vbe/human_state.json.gz",
        ROOT / "data/manual/human_domains.csv",
    )
    assert segmenter.segment_many(inputs.domain) == expected.entropy_vbe.tolist()


def test_aligned_llm_cache_preserves_observed_blank_outputs():
    inputs = pd.read_csv(ROOT / "data/synthetic/synthetic_domains.csv", nrows=10_000)
    values = _load_cached_llm(
        ROOT / "data/reference/synthetic_predictions.csv.gz",
        inputs.domain.astype(str).tolist(),
    )
    assert values.count("") == 3
    assert None not in values
