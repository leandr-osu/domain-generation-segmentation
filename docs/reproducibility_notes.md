# Reproducibility notes

The release distinguishes three levels of reproducibility.

1. The manual annotations, synthpop input/output, final synthetic dataset,
   Viterbi count tables, VBE inference state, and frozen method predictions are
   byte-stable artifacts.
2. Viterbi, fallback Viterbi, and VBE can be rerun locally against the canonical
   inputs. POS can be rerun when the checkpoint identified by
   `artifacts/pos_segmentation_model_manifest.csv` is supplied.
3. GPT and hybrid calls are external and stochastic. The GPT predictions used
   for the table are frozen. The original workspace retained only aggregate
   final hybrid metrics, so those two cells can be reported and checked against
   the retained aggregate, but cannot be reconstructed row by row without a
   new API run. A new run is scientifically a replication, not a byte-for-byte
   reproduction.

## Known manuscript discrepancy

The final tuned fallback implementation produces synthetic EM `0.846900` and
Jaccard `0.906277` on all 100,000 rows. Conventional three-decimal rounding is
therefore `0.847 / 0.906`. The manuscript table says `0.846 / 0.906`. Human
results (`0.595500 / 0.717243`) and every other deterministic table cell agree
with the retained artifacts at the displayed precision. This repository keeps
both values visible rather than silently modifying either the code or paper.

The methodology text says the POS segmenter reused the Brown-corpus checkpoint.
The retained row-level predictions match the Universal Dependencies English
EWT checkpoint (`model.safetensors` SHA-256 `48075b55...`) and do not match the
Brown generation checkpoint (`f682f38f...`). This release therefore uses the
former for segmentation and identifies both complete checkpoint manifests.

## Timing

Wall-clock throughput depends on CPU/GPU, process count, API limits, and network
conditions. `publication_throughput.csv` records the numbers printed in the
manuscript; reruns are expected to differ. Accuracy checks do not compare time.

## External data

The 10-million-domain VBE training corpus and some generator dictionaries are
not redistributable. Safe VBE inference states and the frozen synthetic corpus
are committed so this restriction does not prevent reproduction of the table.
See `data/resources/README.md` for the exact generator-resource hashes.
