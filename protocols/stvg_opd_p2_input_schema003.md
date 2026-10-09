# Strict legacy input-schema bridge for fixed P2

The first formal P2 clean input check stopped before writing any new formal
prediction. Its sealed P0 input stores `parent` and the SHA of the original
baseline NPZ but omits `source_model_state_sha256`; later inputs explicitly
store that source digest, query ordinal and observation budget. The source
digest exists in the exact old NPZ referenced by the old input. The old worker
and controller exited; there is no failed fit or dead-memory comparison.

Original code, metadata, logs and the real 96-arrival/192-fit qualification
evidence are retained. This additive bridge reads the referenced old NPZ only
after exact byte SHA and receipt/runtime checks. It checks pixel SHA, native
boxes, event interval, frame IDs, selected native indices and the full packed
expert evidence, then uses that cache's recorded source-model digest and the
original Uniform4 budget. The original legacy schema, source direction and
zero-new-DINO clean-cache provenance must match exactly. Current/current pairs
retain exact equality too. Neither old nor new input bytes are rewritten.

CPU root qualification reads all 256 original cross-domain inputs and their
hash-bound caches, checks the four actual old/current qualification pairs, and
rejects 48 deliberately wrong metadata/native/expert pairs. No GT or model runs
are involved. Numerical P2 qualification remains the separately saved real
192 old/new complete fits; this input bridge changes no fitter or audit.

The first missing Direct fit in each source direction is qualified in the real
formal capture/input-alias path before acceptance. It must equal the previously
qualified complete fit and a new actual repeat, with unchanged source/expert
state. Qualification stops at the save boundary without accepting predictions.
CPU root reads its actual complete fit, exact original math dictionary and
1792 reset/commit chain. The first formal acceptance again requires equality
with that actual serialized qualified fit. These comparisons never refer to
memory of an exited process. Qualification is not spliced into formal streams.

Original fixed P2 source/configuration/arms/rosters/state reset/readout and all
scientific runtime bytes remain unchanged. All phase deployment arms and both
directions must seal before GT, and real phase root/view/public/archive closure
precedes P3. EATA and other paused historical queues remain paused. The bridge
closes only the bounded compatibility repair, not P2 efficacy or the paper.
