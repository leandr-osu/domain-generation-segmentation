# Manual annotations

`human_domains.csv` is the canonical 2,000-domain evaluation set. It contains:

| column | meaning |
|---|---|
| `Unnamed: 0` | retained original row identifier |
| `domain` | observed domain name |
| `segmented` | manual pipe-delimited segmentation, including the eTLD |
| `english` | whether the annotator judged the domain English |
| `compound` | contains a compound word |
| `abbr` | contains an acronym or abbreviation |
| `brand_name` | contains a brand name |
| `human_name` | contains a person's name |
| `location_name` | contains a location name |
| `misspelled` | contains a misspelling |

`annotator_1_overlap.csv` contains the second independent annotation of the
250-domain overlap. `annotator_2_all.csv` is the original full annotation sheet
whose first 250 rows form the other side of that overlap. Original headers and
row identifiers are retained to avoid silently changing the source data.

The overlap has 196/250 (78.4%) exact segmentation agreement. The committed
agreement summaries are in `../reference/annotation_agreement.csv` and
`../reference/annotation_agreement_report.md`.

Labels are judgments, not ground truth about the registrants' intent. Domain
names can also be sensitive observational data; downstream users should follow
their institution's data-handling and ethics requirements.
