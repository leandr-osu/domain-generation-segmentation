"""Part-of-speech confidence segmenter evaluated in the publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SEPARATORS = re.compile(r"[-.]")


def generate_candidates(text: str, max_segments: int = 4) -> list[list[str]]:
    """Enumerate 2--4 segmentations with a maximum segment length of 14."""
    length = len(text)
    stack: list[tuple[int, list[str]]] = [(0, [])]
    candidates: list[list[str]] = []
    while stack:
        start, path = stack.pop()
        if start == length:
            if 2 <= len(path) <= max_segments:
                candidates.append(path)
            continue
        if len(path) >= max_segments:
            continue
        for end in reversed(range(start + 1, min(length + 1, start + 15))):
            stack.append((end, [*path, text[start:end]]))
    return candidates


@dataclass
class PosSegmenter:
    """Choose the candidate with highest mean POS-token confidence."""

    model_path: str | Path

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
            raise RuntimeError("install the 'models' extra to run POS segmentation") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path))
        self.model.eval()
        has_mps = bool(getattr(getattr(torch, "backends", None), "mps", None)) and torch.backends.mps.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu")
        self.model.to(self.device)

    def candidate_confidences(self, candidates: list[list[str]], batch_size: int = 256) -> list[float]:
        if not candidates:
            return []
        torch = self._torch
        confidences: list[float] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            encodings = self.tokenizer(
                [" ".join(words) for words in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                is_split_into_words=False,
            ).to(self.device)
            with torch.no_grad():
                probabilities = torch.nn.functional.softmax(self.model(**encodings).logits, dim=-1)
            for index in range(len(batch)):
                maxima = probabilities[index].max(-1).values.tolist()
                seen: set[int] = set()
                per_word: list[float] = []
                for token_index, word_id in enumerate(encodings.word_ids(index)):
                    if word_id is None or word_id in seen:
                        continue
                    seen.add(word_id)
                    per_word.append(maxima[token_index])
                confidences.append(sum(per_word) / len(per_word) if per_word else 0.0)
        return confidences

    def segment_piece(self, text: str) -> tuple[list[str], float]:
        candidates = generate_candidates(text)
        if not candidates:
            return [text], 0.0
        confidences = self.candidate_confidences(candidates)
        best = max(range(len(candidates)), key=lambda index: confidences[index])
        return candidates[best], float(confidences[best])

    def segment_domain(self, domain: str) -> tuple[str, float]:
        pieces = [piece for piece in SEPARATORS.split(str(domain)) if piece]
        if not pieces:
            return "", 0.0
        segments: list[str] = []
        confidences: list[float] = []
        # The publication run scored every separator-delimited piece,
        # including the eTLD. This explains outputs such as ``co|m`` and is
        # intentionally preserved even though holding the eTLD intact would
        # generally be a better production choice.
        for piece in pieces:
            words, score = self.segment_piece(piece)
            segments.extend(words)
            confidences.append(score)
        score = sum(confidences) / len(confidences) if confidences else 0.0
        return "|".join(segments), score
