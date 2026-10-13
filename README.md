# Domain generation and segmentation

Code and data for synthetic domain generation and domain-name segmentation from:

> Andrew Le and Zane Ma. **Semantic Analysis of DNS Names Through Automated
> Domain Segmentation.**

## Repository structure

```text
.
├── artifacts/
│   ├── vbe/                          # VBE inference states (human, synthetic)
│   └── pos_*_model_manifest.csv      # hashes of the external POS checkpoints
├── configs/publication.json          # publication evaluation configuration
├── data/
│   ├── generation/                   # feature table and synthpop feature vectors
│   ├── manual/human_domains.csv      # 2,000 manually segmented domains
│   ├── reference/                    # frozen predictions and publication metrics
│   ├── resources/                    # Viterbi word counts and generator word lists
│   └── synthetic/domains.csv         # 100,000 synthetic domains
├── scripts/                          # resource preparation and model training
├── src/domain_generation_segmentation/
│   ├── generation.py                 # synthetic domain sampler
│   ├── evaluate.py                   # configuration-driven evaluation
│   └── segmenters/                   # Viterbi, VBE, POS, GPT, and hybrid
└── tests/
```

## Setup

Python 3.12 is recommended; dependency versions are pinned in `pyproject.toml`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Install `.[models]` as well to run the POS segmenter, and `.[llm]` to make new
GPT API calls.

## Generate the synthetic domains

```bash
domain-generate data/generation/synthetic_features.csv data/resources/generator outputs/domains.csv --seed 42
```

This reproduces `data/synthetic/domains.csv` byte for byte. It needs four word
lists that cannot be redistributed (`names.csv`, `cities.csv`,
`compounds.csv`, `acronyms.csv`); place them in `data/resources/generator/`.
Their hashes are listed in `data/resources/generator_manifest.csv`.

## Evaluate the segmenters

```bash
domain-evaluate configs/publication.json
```

This runs every segmenter on the synthetic and human domains, writes the
predictions and `outputs/metrics.csv`, and checks each score against
`data/reference/publication_metrics.csv`. Run it from the repository root. GPT
and the hybrid replay the frozen API responses, so the run is offline and
deterministic. POS runs only when its checkpoint is at
`artifacts/pos_segmentation_model/`, and is skipped otherwise. The human hybrid
score is aggregate-only, so it is reported as `aggregate_only` and not checked.

## Change the evaluation configuration

Copy `configs/publication.json`, edit the copy, and pass it to
`domain-evaluate`.

- `datasets.<name>.rows` sets how many leading rows each method is scored on.
  Remove a method to skip it. Add a dataset with its own `input` CSV, which
  needs `domain` and `segmented` columns.
- Set `llm_cache` to `null` to call the API instead of replaying responses.
  This needs `OPENAI_API_KEY`. `llm` sets the model, batch size, and workers.
- `hybrid_threshold` is the fallback confidence below which the hybrid routes a
  domain to GPT.
- `fallback` holds the Viterbi fallback's priors and penalties.
- `pos_model` is the POS checkpoint directory, and `output_dir` is where
  results are written.

Unknown keys are rejected. Scores that differ from the publication are
reported as `MISMATCH`; datasets without publication scores are reported as
`no_reference`.
