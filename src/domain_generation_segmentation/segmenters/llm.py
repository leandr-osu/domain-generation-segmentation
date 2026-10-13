"""GPT-5-mini segmentation and the Viterbi/GPT hybrid."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from typing import Sequence


MODEL = "gpt-5-mini"

PROMPT_PREFIX = (
    "Below is a list of Internet domain names. "
    "Segment each domain name into its component words. "
    "Domain names can consist of any number of words of any language. "
    "They may also contain misspelled words, numbers, and other types of non-standard text. "
    "Any punctuation should be treated as a separator between two segments and be replaced by a | character. "
    "Aside from punctuation, no other characters should be removed from the domain. "
    "Output the segmentations in the form segment1|segment2|..."
    "You must return exactly one segmentation for each input string. "
    "Do not omit or skip any input. If unsure, still output the input with | separators where appropriate."
    "Here are some examples:"
    "Ex 1. fresacoin.com -> fresa|coin|com"
    "Ex 2. armchairexperts.com -> armchair|experts|com"
    "Ex 3. forum-stone.com -> forum|stone|com"
)

SEGMENT_SCHEMA = {
    "name": "segments_schema",
    "schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Segmented strings in input order, with words separated by '|'.",
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    },
}


def create_prompt(domains: Sequence[str]) -> str:
    return "\nReturn the segmentation for each input string in order.\nInput strings:\n" + "\n".join(domains) + "\n"


def _one_batch(domains: list[str], model: str, max_retries: int) -> list[str]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'llm' extra to call the OpenAI API") from exc

    client = OpenAI()
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": PROMPT_PREFIX},
                    {"role": "user", "content": create_prompt(domains)},
                ],
                response_format={"type": "json_schema", "json_schema": SEGMENT_SCHEMA},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content or "{}")
            values = parsed.get("segments", [])
            if isinstance(values, dict):
                values = list(values.values())
            if len(values) != len(domains):
                raise ValueError(f"expected {len(domains)} outputs, received {len(values)}")
            return [str(value) for value in values]
        except Exception:
            if attempt + 1 == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def segment_domains(
    domains: Sequence[str],
    *,
    model: str = MODEL,
    batch_size: int = 100,
    max_workers: int = 5,
    max_retries: int = 3,
) -> list[str]:
    """Send batches to GPT and return the results in input order."""
    domain_list = [str(domain) for domain in domains]
    batches = [domain_list[start : start + batch_size] for start in range(0, len(domain_list), batch_size)]
    output: list[list[str] | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_one_batch, batch, model, max_retries): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            output[futures[future]] = future.result()
    return [value for batch in output for value in (batch or [])]


def hybrid_segment(
    domains: Sequence[str],
    fallback_outputs: Sequence[str],
    confidence_scores: Sequence[float | None],
    *,
    threshold: float = 0.1,
    model: str = MODEL,
    batch_size: int = 100,
    max_workers: int = 5,
) -> tuple[list[str], list[str]]:
    """Use GPT when the fallback segmenter's confidence is below ``threshold``."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not (len(domains) == len(fallback_outputs) == len(confidence_scores)):
        raise ValueError("domains, outputs, and confidence scores must have equal lengths")
    low = [index for index, value in enumerate(confidence_scores) if value is None or value < threshold]
    outputs = list(fallback_outputs)
    sources = ["viterbi_fallback"] * len(outputs)
    if low:
        llm_outputs = segment_domains(
            [str(domains[index]) for index in low],
            model=model,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        for index, prediction in zip(low, llm_outputs, strict=True):
            outputs[index] = prediction
            sources[index] = "gpt_5_mini"
    return outputs, sources
