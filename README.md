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
│   ├── llm/                          # stored GPT-5 mini responses for replay
│   ├── manual/human_domains.csv      # 2,000 manually segmented domains
│   ├── resources/                    # Viterbi word counts and generator word lists
│   └── synthetic/domains.csv         # 100,000 synthetic domains
├── scripts/                          # resource preparation and model training
└── src/domain_generation_segmentation/
    ├── generation.py                 # synthetic domain sampler
    ├── evaluate.py                   # configuration-driven evaluation
    └── segmenters/                   # Viterbi, VBE, POS, GPT, and hybrid
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

It needs `data/resources/generator/cities.csv`, which is not included because
its source license does not allow public redistribution. You can build it from
the free [SimpleMaps US Cities database](https://simplemaps.com/data/us-cities):
save the download's `uscities.csv` as `data/resources/uscities.csv`, then run

```bash
python scripts/prepare_generator_resources.py cities data/resources/uscities.csv data/resources/generator/cities.csv
```

SimpleMaps updates this database, so a newer version gives a slightly different
corpus. To reproduce `data/synthetic/domains.csv` exactly, contact
<leandr@oregonstate.edu> for the `cities.csv` used in the publication; its
expected hash is in `data/resources/generator_manifest.csv`.

## Evaluate the segmenters

```bash
domain-evaluate configs/publication.json
```

This runs every segmenter on the synthetic and human domains, prints the exact
match and Jaccard scores, and writes them to `outputs/metrics.csv` alongside the
predictions. Run it from the repository root; the full run takes about 40
minutes. GPT and the hybrid replay the stored responses in `data/llm/`, so the
run is offline and deterministic. POS runs only when its checkpoint is at
`artifacts/pos_segmentation_model/`, and is skipped otherwise. The paper's
human hybrid score came from live API responses that were not retained, so the
replayed score differs slightly.

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

Unknown keys are rejected.

## Data sources

The generator word lists in `data/resources/generator/` include third-party
data:

- `cities.csv` (available on request): city names and populations from the
  [SimpleMaps US Cities database](https://simplemaps.com/data/us-cities),
  Basic v1.92.
- `names.csv`: first names with their counts summed over 1880–2014, from the
  US Social Security Administration's baby-name data, which is in the public
  domain ([CC0](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data)).
- `acronyms.csv`: alphanumeric entries from Wiktionary's English acronym,
  initialism, and abbreviation categories, licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This file
  is shared under the same license.
- `compounds.csv`: the unique compound words (`stim`) from LADEC, licensed
  under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for
  noncommercial use only. The first entry, `gadabout`, is omitted, as in the
  published corpus. Gagné, C. L., Spalding, T. L., & Schmidtke, D. (2019).
  LADEC: The Large Database of English Compounds. *Behavior Research Methods*,
  51, 2152–2179. <https://doi.org/10.3758/s13428-019-01282-6>; data:
  <https://doi.org/10.7939/r3-dyqx-9b36>.
