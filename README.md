# Domain generation and segmentation artifact

This repository is the publication artifact for **“Semantic Analysis of DNS
Names Through Automated Domain Segmentation.”** It contains the cleaned code,
manual annotations, synthetic data, model inputs, and frozen predictions needed
to reproduce the domain-generation and segmentation experiments. Exploratory
programs and outputs that were not used in the final paper are intentionally
excluded.

The six evaluated segmentation configurations are included:

1. frequency-based Viterbi;
2. Viterbi with abbreviation and edit-distance fallbacks;
3. variation of branching entropy (VBE);
4. part-of-speech (POS) confidence;
5. GPT-5 mini; and
6. the confidence-routed Viterbi/GPT hybrid.

The repository also includes the 2,000 manually segmented domains, both raw
annotator files, the 100,000 generated domains, the frozen synthpop
intermediates, safe VBE inference states, and row-level predictions wherever
they survived the original experiment.

## Reproducibility at a glance

There are three kinds of result in this artifact:

- **Immediately verifiable:** all committed-file checksums and the scores
  computed from frozen row-level predictions. This needs only Python, NumPy,
  and pandas and does not make network calls.
- **Locally rerunnable:** synthetic-feature simulation, the final domain
  sampler, Viterbi, fallback Viterbi, and VBE. Exact generator resources are
  required only when rebuilding the already committed synthetic corpus.
- **Externally dependent:** POS needs the 431 MB checkpoint identified in
  `artifacts/pos_segmentation_model_manifest.csv`. Rebuilding the generation
  features needs the separate Brown checkpoint manifest. GPT and hybrid need an OpenAI API model
  and are stochastic. Frozen GPT predictions are provided; the original
  row-level hybrid snapshot was not retained, so only its aggregate scores can
  be preserved exactly.

Start with the offline verification before attempting a new model run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

domain-reproduce
pytest -q
```

`domain-reproduce` verifies `checksums.sha256`, recomputes exact match and
Jaccard from the frozen prediction files, and prints the paper table. A passing
run ends with ten `ok` rows and this displayed-precision result:

| method | synthetic EM | synthetic Jaccard | human EM | human Jaccard |
|---|---:|---:|---:|---:|
| Viterbi | 0.803 | 0.881 | 0.531 | 0.672 |
| Viterbi + fallbacks | 0.846 | 0.906 | 0.596 | 0.717 |
| VBE/entropy | 0.389 | 0.549 | 0.445 | 0.597 |
| POS | 0.000 | 0.114 | 0.000 | 0.077 |
| GPT-5 mini | 0.865 | 0.911 | 0.763 | 0.838 |
| Hybrid | 0.884 | 0.926 | 0.655 | 0.760 |

Full-precision retained values and evaluation row counts are in
`data/reference/publication_metrics.csv`. GPT used all 2,000 human rows but
only the first 10,000 synthetic rows. The other methods used all 100,000
synthetic rows.

### Important result note

The final tuned fallback implementation gives synthetic EM `0.846900`, which
normally rounds to `0.847`; the manuscript table contains `0.846`. Its Jaccard
is `0.9062766609`. This release preserves both the actual final output and the
reported table entry instead of altering either. See
`docs/reproducibility_notes.md` for this and the hybrid-artifact boundary.

## Repository map

```text
.
├── artifacts/                    # VBE states and POS checkpoint manifests
├── configs/publication.json      # seeds and final hyperparameters
├── data/
│   ├── generation/               # manual feature table + synthpop output
│   ├── manual/                   # canonical data + both annotator sheets
│   ├── reference/                # frozen predictions and expected metrics
│   ├── resources/                # Viterbi counts + generator manifest
│   └── synthetic/                # final 100,000-domain evaluation set
├── scripts/                      # preprocessing, training, and checks
├── src/domain_generation_segmentation/
│   ├── generation.py             # deterministic domain sampler
│   ├── evaluation.py             # publication metrics
│   ├── runner.py                 # common segmentation CLI
│   └── segmenters/               # Viterbi, VBE, POS, GPT, and hybrid
└── tests/
```

All paths in the rest of this document assume the repository root as the
working directory.

## Data

### Manual segmentation

`data/manual/human_domains.csv` is the canonical 2,000-row human evaluation
set. A segmentation uses `|` as its boundary marker and includes the effective
top-level domain (eTLD), for example:

```text
fittedkitchensupplies.com -> fitted|kitchen|supplies|com
```

The CSV also records annotator judgments for English language, compound words,
abbreviations, brands, human names, locations, and misspellings. The original
annotation sheets are:

- `annotator_1_overlap.csv`: the second independent annotation for the
  250-domain overlap;
- `annotator_2_all.csv`: the full primary annotation, whose first 250 rows are
  the other side of the overlap.

The annotators agreed exactly on 196/250 segmentations (78.4%). The derived
agreement files are in `data/reference/`. Details and column definitions are
in `data/manual/README.md`.

### Synthetic segmentation

`data/synthetic/synthetic_domains.csv` has 100,000 domains and their known
component boundaries. Its SHA-256 is:

```text
e37eece01c9aaf9bd92138b803a9433dc7f8b297b08374a1e3b2023eb6c32821
```

The generator first uses synthpop to model the joint distribution of manual
binary features, eTLD, POS tags, and named-entity tags. It then fills entity
slots, abbreviation/compound slots, and remaining POS slots from weighted word
dictionaries; when requested, it creates one adjacent-character transposition
typo. The words are concatenated into the domain while the unjoined sequence is
stored as the reference segmentation.

These are research labels and generated examples, not authoritative statements
about registrants or intent. Users should apply appropriate ethics and data
handling review when combining the domains with live DNS or security data.

## Rerun the deterministic segmenters

The base installation is sufficient for Viterbi, fallback Viterbi, and VBE.
The bundled unigram/bigram tables and VBE states are loaded automatically.

```bash
mkdir -p outputs

domain-segment \
  data/manual/human_domains.csv \
  outputs/human_deterministic.csv \
  --methods viterbi fallback vbe

python scripts/compare_predictions.py \
  outputs/human_deterministic.csv \
  data/reference/human_predictions.csv.gz
```

For the complete synthetic run:

```bash
domain-segment \
  data/synthetic/synthetic_domains.csv \
  outputs/synthetic_deterministic.csv \
  --methods viterbi fallback vbe

python scripts/compare_predictions.py \
  outputs/synthetic_deterministic.csv \
  data/reference/synthetic_predictions.csv.gz
```

The runner writes a sibling `*_summary.csv` whenever the input has a
`segmented` reference column. The synthetic fallback is the slowest of these
three methods; on a recent workstation the complete validation took several
minutes. To smoke-test the pipeline, append `--limit 100` to both commands.
Do not compare a limited run's summary with the full publication score.

### Metric definitions

Exact match is the fraction of predicted pipe-delimited strings exactly equal
to their references. Jaccard is computed for each row over the **sets of
segments**, then averaged:

```text
J(reference, prediction) = |set(reference) ∩ set(prediction)|
                           / |set(reference) ∪ set(prediction)|
```

It is not boundary Jaccard; repeated segments count once. This behavior is
implemented in `evaluation.py` and covered by tests.

## Segmentation methods

### Viterbi

The baseline maximizes a unigram/bigram log-probability over every split of
each dot- or hyphen-separated domain component. It uses the bundled Norvig
counts in `data/resources/word_counts/`, a maximum candidate length of 24, and
the publication total count `1,024,908,267,229`. Its implementation is in
`segmenters/viterbi.py`.

### Viterbi with fallbacks

The tuned variant adds explicit log priors for unknown abbreviations and a
conservative Damerau-Levenshtein-distance-one search. The final boundary,
prior, edit, and acceptance parameters are in `configs/publication.json` and
are the defaults in the code. It emits the observed spelling even when a fuzzy
dictionary match supplies its probability.

### Variation of branching entropy

VBE computes left and right changes in character branching entropy and uses
the minimum normalized change as an autonomy score. Dynamic programming finds
the maximum-autonomy segmentation. The original character dictionaries were
learned from 10 million eTLD-stripped passive-DNS domains. That corpus cannot
be redistributed; the two committed JSON/gzip states contain the normalized
values needed for inference on the evaluation sets and no executable pickle.

`scripts/build_vbe_dictionaries.py` documents the n-gram counting step for
researchers with access to an equivalent authorized corpus. The supplied
states are the canonical artifacts for reproducing the paper.

### POS confidence

Install the model dependencies and place the canonical evaluation checkpoint at
`artifacts/pos_segmentation_model/` (or pass another location):

```bash
python -m pip install -e '.[models]'
python scripts/verify_generator_resources.py artifacts/pos_segmentation_model \
  --manifest artifacts/pos_segmentation_model_manifest.csv

domain-segment \
  data/manual/human_domains.csv \
  outputs/human_pos.csv \
  --methods pos \
  --pos-model artifacts/pos_segmentation_model
```

The generic verifier accepts the POS manifest because it uses the same
`filename,bytes,sha256` columns and ignores extra manifest columns.

The method enumerates every 2–4-token split with token length at most 14,
scores the candidates with a BERT token-classification model trained on
Universal Dependencies English EWT, and chooses the largest mean token
confidence. It repeats this for every dot- or hyphen-delimited piece, including
the eTLD; the latter is unusual but preserves the evaluated behavior exactly.
The original checkpoint is 431 MB and exceeds normal Git hosting
limits; its exact files and hashes are in
`artifacts/pos_segmentation_model_manifest.csv`.
The original training run did not record its random seed. Consequently,
`scripts/train_segmentation_pos_tagger.py` provides a deterministic retraining procedure,
but newly trained weights are a replication and are not expected to be
byte-identical to the canonical checkpoint. Frozen POS predictions remain
fully score-verifiable with `domain-reproduce`.

The manuscript describes this evaluator as reusing the Brown model. The
retained predictions instead match the Universal Dependencies checkpoint and
do not match the Brown checkpoint. Both model manifests are included so this
implementation-level discrepancy is explicit.

### GPT-5 mini

The exact publication prompt and structured-output schema are in
`segmenters/llm.py`. To replay the retained responses without network access or
cost:

```bash
domain-segment \
  data/manual/human_domains.csv \
  outputs/human_gpt_frozen.csv \
  --methods gpt \
  --llm-cache data/reference/human_predictions.csv.gz

domain-segment \
  data/synthetic/synthetic_domains.csv \
  outputs/synthetic_gpt_frozen.csv \
  --methods gpt \
  --limit 10000 \
  --llm-cache data/reference/synthetic_predictions.csv.gz
```

To make a new paid API run instead, install the optional client, set the key in
the environment, and omit `--llm-cache`:

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY='...'

domain-segment \
  data/manual/human_domains.csv \
  outputs/human_gpt_new.csv \
  --methods gpt \
  --llm-model gpt-5-mini \
  --llm-batch-size 100 \
  --llm-workers 5
```

This sends domain names to an external service and can incur cost. Review your
institutional policy before sending any dataset. The publication used the
`gpt-5-mini` alias through Chat Completions with JSON-schema structured output.
The current [official model page](https://developers.openai.com/api/docs/models/gpt-5-mini)
marks that model as deprecated, so future availability and behavior are not
guaranteed. The [Chat Completions API reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)
documents the request interface retained here. A new response is a replication
rather than a byte-stable reproduction.

### Hybrid

The hybrid first runs fallback Viterbi, computes its posterior confidence, and
routes confidence below `0.1` to GPT-5 mini. Run it with `--methods hybrid` and
the same LLM options described above. A no-cost diagnostic can use the frozen
GPT cache:

```bash
domain-segment \
  data/manual/human_domains.csv \
  outputs/human_hybrid_cached_diagnostic.csv \
  --methods hybrid \
  --hybrid-threshold 0.1 \
  --llm-cache data/reference/human_predictions.csv.gz
```

This diagnostic recombines the retained standalone GPT responses; it is not a
replacement for the lost original hybrid response file. The original
workspace preserved only aggregate hybrid metrics (`0.884/0.926` synthetic and
`0.655/0.7601924603` human). Those values are explicitly marked
`aggregate_only` in `publication_metrics.csv`.

## Rebuild the synthetic domains

The committed synthetic CSV is sufficient for every segmentation experiment.
This section reconstructs how it was generated.

### 1. Optional: recreate the manual feature table

The exact table is already frozen as
`data/generation/combined_domains_expanded.csv`. With the canonical Brown POS
generation checkpoint and network access for `dslim/bert-base-NER`, its cleaned
preprocessing pipeline is:

```bash
python -m pip install -e '.[models,generation]'
python scripts/prepare_generation_features.py \
  data/manual/human_domains.csv \
  outputs/combined_domains_expanded.csv \
  --pos-model artifacts/pos_generation_model \
  --ner-model dslim/bert-base-NER
```

The publication table contains the 1,414 annotations marked English. Its
canonical SHA-256 is
`688dace7ca6223db9eae59fdfa901f4fbb3aab50368f4a07a32258d1541af549`.

### 2. Recreate 100,000 feature vectors with synthpop

The byte-matched environment was R 4.3.3 with synthpop 1.9.2. Confirm the
installed version, then run the recorded seed:

```bash
Rscript -e 'stopifnot(as.character(packageVersion("synthpop")) == "1.9.2")'
mkdir -p outputs
Rscript scripts/simulate_features.R \
  data/generation/combined_domains_expanded.csv \
  outputs/synthetic_features \
  141727361

sha256sum outputs/synthetic_features.csv
```

Expected digest:

```text
d529f433909246ad9874ed6fea5b98afa9e84d011a49cb41e46dab89dc2eb53b
```

The identical file is committed as `data/generation/synthetic_features.csv`,
so lack of the exact R version does not block subsequent steps.

### 3. Prepare the weighted sampler resources

Several source datasets have their own terms, and the SimpleMaps license in
the research workspace specifically prohibited public redistribution. They are
therefore not silently republished here. `data/resources/README.md` lists the
upstream sources, transformations, required filenames, byte sizes, and SHA-256
hashes.

Place authorized prepared copies in `data/resources/generator/`, then verify
them:

```bash
python scripts/verify_generator_resources.py data/resources/generator
```

The supporting scripts cover all transformations used in the pipeline:

- `build_wikitext_frequencies.py` streams WikiText-103 and builds word counts;
- `tag_wikitext_pos.py` creates weighted POS files with the Brown checkpoint;
- `fetch_wiktionary_abbreviations.py` queries the three abbreviation
  categories;
- `prepare_generator_resources.py` converts the raw names, cities, brands,
  compounds, and acronym data to the two-column sampler format.

Exact reconstruction requires files matching the manifest, not merely newer
versions of the upstream datasets. The odd-looking `DET copy.csv` and
`brands copy.csv` names are intentional: those files were members of the
original `X`-tag fallback pool.

### 4. Run the deterministic sampler

Python 3.12.3, NumPy 1.26.4, and pandas 2.2.3 were used for the byte-level
validation. The versions are pinned in `pyproject.toml`.

```bash
domain-generate \
  data/generation/synthetic_features.csv \
  data/resources/generator \
  outputs/synthetic_domains.csv \
  --seed 42

sha256sum outputs/synthetic_domains.csv
```

The expected digest is the synthetic-corpus hash shown above. The generator
uses an explicit resource order for `X` tags, because relying on filesystem
glob order would make sampling nonportable.

## Rebuild supporting model resources

To train a new deterministic Brown POS checkpoint for generation:

```bash
python -m pip install -e '.[models]'
python scripts/train_pos_tagger.py artifacts/pos_generation_model_retrained \
  --base-model bert-base-cased \
  --seed 42 \
  --epochs 3
```

To train a new Universal Dependencies English EWT checkpoint for the POS
segmentation experiment, first obtain an authorized checkout containing
`en_ewt-ud-train.conllu` and `en_ewt-ud-dev.conllu`, then run:

```bash
python scripts/train_segmentation_pos_tagger.py \
  /path/to/UD_English-EWT \
  artifacts/pos_segmentation_model_retrained \
  --base-model bert-base-cased \
  --seed 42 \
  --epochs 3
```

To count VBE dictionaries from an authorized file containing one
eTLD-stripped domain per line:

```bash
python scripts/build_vbe_dictionaries.py \
  /path/to/ten_million_domains.txt \
  data/resources/vbe_ngrams \
  --max-n 10
```

These commands require downloads and substantial compute/storage. They are
provided to document the full model-building path; the retained checkpoint
hash and VBE states define the exact publication models.

## Integrity, timing, and troubleshooting

- Run `sha256sum -c checksums.sha256` for a standard checksum-only check, or
  `domain-reproduce` for checksums plus score verification.
- A VBE “state does not match” error means the CSV bytes differ from the
  canonical evaluation file. Pass the corresponding state explicitly with
  `--vbe-state` only when you understand that provenance check.
- A generator mismatch almost always means a resource differs, a different
  resource ordering was used, or the pandas/NumPy versions changed.
- API rate limits and model updates can change GPT output and throughput.
- Throughput is hardware-, concurrency-, and network-dependent. The paper's
  recorded values are in `data/reference/publication_throughput.csv`; accuracy
  reproduction does not assert wall-clock equality.

## Citation and licensing

Citation metadata is provided in `CITATION.cff`. Before public release, the
authors or their institution should add the software license they intend to
grant. No license is inferred here because code licensing is a rights-holder
decision. Third-party datasets, word counts, model weights, and APIs remain
subject to their own terms.
