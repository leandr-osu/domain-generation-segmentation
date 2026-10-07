# Inter-Annotator Agreement Analysis

## Binary Annotation Agreement

| Category | Cohen's Kappa | Agreement % | Agreement 95% CI |
|----------|---------------|-------------|-----------------|
| Acronyms/Abbreviations? | 0.269 | 91.2% | [87.6%, 94.4%] |
| Brand names? | 0.056 | 79.6% | [74.4%, 84.4%] |
| Compound words? | 0.211 | 94.8% | [92.0%, 97.2%] |
| English? (Yes/No) | 0.480 | 84.8% | [80.4%, 89.2%] |
| Human names? | 0.614 | 92.8% | [89.6%, 95.6%] |
| Misspelling? | 0.089 | 94.0% | [90.8%, 96.8%] |

## Segmentation Agreement

| Metric | Mean | 95% CI |
|--------|------|--------|
| Boundary F1 | 0.926 | [0.907, 0.945] |
| Token Jaccard | 0.846 | [0.808, 0.882] |

Note: 1 domains had concatenation mismatches (different underlying strings).

## Summary

- Domains analyzed: 250
- Resampling unit: Domain
- Bootstrap iterations: 10,000
- Confidence level: 95%
- Random seed: 42