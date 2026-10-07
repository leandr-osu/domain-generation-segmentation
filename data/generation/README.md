# Frozen generation intermediates

- `combined_domains_expanded.csv` contains the 1,414 manual domains labeled as
  English, their binary features, eTLDs, and expanded Brown-POS/bert-base-NER
  tag columns. SHA-256: `688dace7ca6223db9eae59fdfa901f4fbb3aab50368f4a07a32258d1541af549`.
- `synthetic_features.csv` is the direct 100,000-row synthpop output before
  words were sampled. SHA-256:
  `d529f433909246ad9874ed6fea5b98afa9e84d011a49cb41e46dab89dc2eb53b`.

The synthpop seed recovered from its saved run metadata is `141727361`. The R
script sets it explicitly, fixing an omission in the exploratory script while
retaining byte-identical output under R 4.3.3 and synthpop 1.9.2.
