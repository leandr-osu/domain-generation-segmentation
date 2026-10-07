from domain_generation_segmentation.evaluation import evaluate_predictions, jaccard_similarity


def test_set_jaccard_matches_publication_definition():
    assert jaccard_similarity("a|b|b", "a|c") == 1 / 3


def test_evaluate_predictions():
    metrics = evaluate_predictions(["a|b", "c"], ["a|b", "c|d"])
    assert metrics.rows == 2
    assert metrics.exact_match == 0.5
    assert metrics.mean_jaccard == 0.75
