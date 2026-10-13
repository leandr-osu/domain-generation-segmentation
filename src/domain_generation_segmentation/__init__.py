"""Tools for generating and segmenting synthetic domain names."""

from .evaluation import evaluate_predictions, exact_match, jaccard_similarity

__all__ = ["evaluate_predictions", "exact_match", "jaccard_similarity"]
__version__ = "1.0.0"
