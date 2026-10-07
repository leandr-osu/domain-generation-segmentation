# Synthetic evaluation data

`synthetic_domains.csv` contains 100,000 generated domains and their known
component boundaries. The columns are:

- `domain`, `segmented`: generated domain and pipe-delimited reference;
- `compound`, `abbr`, `brand_name`, `human_name`, `location_name`,
  `misspelled`: generated binary features;
- `eTLD`: sampled effective top-level domain;
- `POS_Tags`, `NER_Tags`: synthpop-generated tag sequences used to select the
  component dictionaries.

Canonical SHA-256:

```text
e37eece01c9aaf9bd92138b803a9433dc7f8b297b08374a1e3b2023eb6c32821  synthetic_domains.csv
```

The final dataset is the canonical object for evaluating segmentation. To
reconstruct it, first reproduce `data/generation/synthetic_features.csv` with
`scripts/simulate_features.R`, acquire the exact sampler resources described in
`data/resources/`, and then run `domain-generate` with seed 42.
