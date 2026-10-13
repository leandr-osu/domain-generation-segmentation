from pathlib import Path

import pandas as pd

from domain_generation_segmentation.segmenters import viterbi
from domain_generation_segmentation.segmenters.vbe import VBESegmenter
from domain_generation_segmentation.runner import _default_vbe_state, _load_cached_llm


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


def test_aligned_llm_cache_replays_stored_outputs():
    inputs = pd.read_csv(ROOT / "data/synthetic/domains.csv", nrows=10_000)
    values = _load_cached_llm(
        ROOT / "data/reference/synthetic_predictions.csv.gz",
        inputs.domain.astype(str).tolist(),
    )
    expected = pd.read_csv(ROOT / "data/reference/synthetic_predictions.csv.gz", nrows=10_000)
    assert values == expected.gpt_5_mini.tolist()


def test_runner_output_can_be_reused_as_llm_cache(tmp_path):
    path = tmp_path / "runner_output.csv"
    pd.DataFrame(
        {"domain": ["example.com", "test.org"], "gpt_5_mini_output": ["example|com", "test|org"]}
    ).to_csv(path, index=False)
    assert _load_cached_llm(path, ["example.com", "test.org"]) == ["example|com", "test|org"]


def test_default_vbe_states_match_their_evaluation_csvs():
    for csv in ("data/manual/human_domains.csv", "data/synthetic/domains.csv"):
        VBESegmenter.from_json_gz(_default_vbe_state(ROOT / csv), dataset_path=ROOT / csv)
