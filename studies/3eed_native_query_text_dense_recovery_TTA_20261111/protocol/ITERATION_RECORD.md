# R75 bounded native dense alignment recovery

Hypothesis: fixed query-to-text soft relations under mild pre-fusion evidence loss align adaptation with the original grounding interface. Change versus R74: no region pooling or region negatives; freeze complete-observation projected text and distill row+column relations on two genuinely recomputed masked suffixes.

Risks: complete-input teacher wrong, native head not trained for this masking, all-token special/background effects, sparse task decision changes, output-only/extra-forward benefit. Controls: same-budget probability/token/representation fusion, output-only update, shuffled target rows. C-like M3 not generated until all fits sealed. No experts/source retraining/box changes/RGB/EMA/layer search/FINAL. Fixed DEV336, no automatic expansion.
