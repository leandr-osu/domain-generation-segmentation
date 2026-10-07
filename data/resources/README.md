# External generator resources

The frozen feature vectors and final 100,000-domain dataset are committed, so
none of these files is needed to reproduce the segmentation table. They are
needed only to reconstruct every sampled domain from the feature vectors.

The original generator directory cannot be redistributed as-is. In particular,
the SimpleMaps license supplied with the US Cities database says that customers
may use and back up the database but may not publicly redistribute it without
permission. Obtain the sources yourself and place the prepared CSVs in
`data/resources/generator/`. The generator refuses to start when a file is
missing. Verify every prepared file against `generator_resource_manifest.csv`;
an exact final SHA-256 match is possible only with those exact derived files.

Upstream sources used by the publication:

- WikiText-103: <https://huggingface.co/datasets/Salesforce/wikitext>
- Amazon-M2: <https://www.kaggle.com/datasets/marquis03/amazon-m2>
- US baby names: <https://www.kaggle.com/datasets/kaggle/us-baby-names/data>
- SimpleMaps US Cities: <https://simplemaps.com/data/us-cities>
- Wiktionary categories for English acronyms, initialisms, and abbreviations

`scripts/prepare_generator_resources.py` captures the transformations used for
names, cities, brands, compounds, and abbreviation lists.
`scripts/build_wikitext_frequencies.py` and `scripts/tag_wikitext_pos.py`
capture the WikiText-103 counting and Brown POS tagging steps, while
`scripts/fetch_wiktionary_abbreviations.py` captures the category query.
`scripts/verify_generator_resources.py` checks a prepared directory against
the manifest before sampling. The POS dictionaries were produced by counting
WikiText-103 words and applying the Brown POS model.
The two files with ` copy` in their names look odd, but they were present in the
original X-tag fallback pool and therefore remain part of the exact manifest.

The bundled Norvig unigram/bigram counts under `word_counts/` are separate:
they drive Viterbi segmentation, not synthetic generation. Their source is
<https://norvig.com/ngrams/>.
