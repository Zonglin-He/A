# TA-STVG query-swap specificity audit v1

Authorized scope: the user's review of fd57375d asks for query-swap only. This
is a representation intervention, not a new selector, training run, or TTA
method. No MLP, layerwise, appearance/motion, new expert, or legacy queue.

## Inputs and intervention

Reuse the 288 sealed expert cells of the latent-quality/atlas panel: VidSTG
144 cells / 24 sources (16 search, 8 confirm), HC-STVG-v2 144 / 21 (14, 7).
There are 240 corrupted and 48 clean cells. Each original panel had 32 search
and 16 confirmation sources; sparse schedule coverage explains the smaller
counts here. All sources have historical exposure, not fresh-test status.

Keep original pixels, Paper48 frame grid, official same-domain checkpoint,
the arrival's pre-update A state, all 32 candidate endpoint pairs including
Old8, anchor, and every frozen atlas probe. The original query latent and
readout are reused. For (V,q') recompute the full text-conditioned encoder;
never substitute the old query's multimodal H. Reuse a swapped encoder output
only across identical video/query/pixels/frame-grid inputs. Replay each arrival
with its own saved A state. No SGD or new candidate decoding is performed.

Within each dataset, sort all 48 original source rows by SHA256 of
`query-swap-v1|dataset|source`. Choose the smallest cyclic offset giving a
one-to-one derangement with different source, different video hash, and
different normalized caption for EVERY row. Freeze this mapping before model
execution, reuse it across orders/conditions, and use the donor's cached query
subject parse. Do not use GT, model scores, or caption similarity to choose it.
Different-source captions are mismatched by provenance, not certified false
descriptions of the recipient video. This audit tests this fixed intervention;
it cannot separate precise semantics from generic caption/subject changes or
guarantee that every swapped caption is incompatible.

## No-GT parity and frozen readout

Two clean cells per dataset, selected by cell SHA, undergo a true-query full
recompute and a swapped-query smoke. Require original hidden, candidate
features, native output, and A state to reproduce the old cache bitwise. The
temporal hidden must exactly reproduce its native start/end head. Root reviews
these four smokes before the full serial Vid -> HC capture. Annotation and
labelled-result opens are denied inside model/readout workers. A failed smoke
is preserved and is not a scored partial experiment.

Use existing source-trained ridge/logistic probes WITHOUT refitting,
renormalizing, choosing alpha, or model selection. Retain all 136 frozen
models unchanged, evaluate the focused subset per dataset: frame event with
Hidden and position/Geometry, candidate precision/recall/tIoU with
Endpoint/Inside/Context/Contrast/Full/Geometry; real and shuffled-fit controls.
Also retain the source-fit intercept-only Null. Position and candidate geometry
inputs are identical for true and swap and must give exact no-change controls.
No swap-native candidates, swap GT, or new top-1 output are used.

Seal every hidden/features and every true/swap probe prediction for BOTH
datasets before joining the already-exposed cached ORIGINAL query's GT span.
Do not read raw annotations, boxes, or source-fit labels in this round.

## Endpoints and aggregation

Primary corrupted expert endpoints: Hidden event AUROC and Full candidate
precision, recall, tIoU R2. Report true, swap, and paired true-minus-swap 95%
intervals. Prespecified block readouts and shuffle/geometry/null controls are
supplementary, not a search for a winning representation. Include clean,
search/confirm, each fixed order, event AP/logloss, MSE/MAE and cell R2.

As in the atlas, first average points within cell; then condition -> order ->
source. Pooled R2 uses source-balanced label moments and MSE, not the mean of
cell R2. Event AUROC/AP are cell metrics, source-macro aggregated; a cell with
only one label class is undefined and counted, not filled with 0.5. Paired
10,000 source-bootstrap draws, seed 20261003. Query donors remain fixed in
bootstrap: uncertainty is conditional on this mapping, not over all mismatches.
All original labels and denominators are identical between true and swap.

Within-cell R2 is absolute prediction accuracy, NOT pairwise ranking accuracy.
Its large negative values can be magnified by low label variance. Pooled R2
and a query gap do not establish safe top-1 selection or deployment vIoU gain.
A positive true-minus-swap gap supports query dependence under this
intervention; it does not establish exclusively semantic coding. A null or
negative gap does not prove absence of query-conditioned information, that
all signal is prior, or that all nonlinear readouts would fail.

## Completion

Independent root readback checks all pixels/state/candidate bindings, frozen
probe hashes, direct readout arithmetic, labels, moments, paired aggregation,
seal ordering and resources. Publish anonymous metrics, controls, failures,
code, protocol, report and plots to Zonglin-He/A; verify remote bytes and
commit. Update RESEARCH_HISTORY with check/snapshot/check. A and CURRENT stay
unchanged. No follow-up experiment is automatically authorized by the outcome.
