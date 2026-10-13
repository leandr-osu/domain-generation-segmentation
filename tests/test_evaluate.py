import json
from pathlib import Path

import pandas as pd

from domain_generation_segmentation.evaluate import main


ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_replays_reference_predictions(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "output_dir": str(tmp_path / "outputs"),
        "pos_model": str(tmp_path / "missing_checkpoint"),
        "datasets": {
            # A dataset without publication reference rows is reported as
            # no_reference, so the 20-row scope does not count as a mismatch.
            "sample": {
                "input": str(ROOT / "data/manual/human_domains.csv"),
                "llm_cache": str(ROOT / "data/reference/human_predictions.csv.gz"),
                "rows": {"viterbi": 20, "fallback": 20, "vbe": 20, "pos": 20, "gpt": 20, "hybrid": 10},
            }
        },
    }))
    main([str(config)])

    predictions = pd.read_csv(tmp_path / "outputs/sample.csv")
    expected = pd.read_csv(ROOT / "data/reference/human_predictions.csv.gz", nrows=20)
    for column, reference in {
        "viterbi_output": "viterbi",
        "fallback_output": "viterbi_fallback",
        "vbe_output": "entropy_vbe",
        "gpt_5_mini_output": "gpt_5_mini",
    }.items():
        assert predictions[column].tolist() == expected[reference].tolist()
    assert predictions.hybrid_output.notna().sum() == 10

    metrics = pd.read_csv(tmp_path / "outputs/metrics.csv").set_index("method")
    assert metrics.loc["pos", "status"] == "skipped"
    assert (metrics.drop(index="pos").status == "no_reference").all()
    assert metrics.loc["viterbi", "rows"] == 20 and metrics.loc["hybrid", "rows"] == 10
