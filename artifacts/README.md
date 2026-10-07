# Model artifacts

## VBE

`vbe/human_state.json.gz` and `vbe/synthetic_state.json.gz` are safe JSON/gzip
conversions of the corpus-specific inference caches used for the final results.
They contain only required normalized entropy values, not executable pickle
objects. They let the VBE predictions be rerun without redistributing the
roughly 2 GB of character n-gram dictionaries derived from 10 million passive
DNS domains.

## POS checkpoints

Two different 431 MB checkpoints survived the final workflow:

- `pos_segmentation_model_manifest.csv` identifies the Universal Dependencies
  English EWT checkpoint that produced the POS row in the publication table.
  Put it at `artifacts/pos_segmentation_model/` or pass another path to
  `domain-segment --pos-model`.
- `pos_generation_model_manifest.csv` identifies the Brown-corpus checkpoint
  used to create the POS features and weighted dictionaries for synthetic
  generation. Put it at `artifacts/pos_generation_model/` when reconstructing
  those inputs.

Frozen POS predictions are committed under `data/reference/`, so the
publication scores can be verified without either checkpoint.

The original training programs did not persist a random seed. Consequently,
the cleaned training scripts make future training deterministic but should not
be expected to reproduce the canonical weights byte for byte. The weight hashes
identify the models actually used in the experiment.

The training entry points are `scripts/train_pos_tagger.py` for the Brown
generation model and `scripts/train_segmentation_pos_tagger.py` for the
Universal Dependencies evaluation model.
