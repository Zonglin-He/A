> Historical v3 source-preparation protocol. The user cancelled the full-source fit at35 committed steps; do not automatically resume. See docs/desta3d_v3/EXTERNAL_PRIVILEGED_OPD.md.

# Structured source loss real interface, v1

Pre-GPU contract (2026-09-28). This is engineering for full-source training,
not a post-full-source actuation experiment and not a new target evaluation.
One fixed already exposed Vid training query 28199 / parent5624461612 / 32
frames. Two fixed states: valid B1 and fresh dual128 seed20260928. No optimizer,
no new native predictions, no target input/GT. Frozen original PTD4B BF16 body
and head. Four source-GT-conditioned branch forward/backward calls.

Exact official GT-prefix, positions/context, physical pixels and inherited
source frame times are retained. Select only MTP endpoint positions (2 x T,
CE sum of two endpoints) or coordinate positions (4 x K x1001, CE mean),
excluding NTP copies, semantic/null/format tokens from structured terms.
Each branch adds .05 original full-vocabulary branch CE (together .1 times
the two-branch mean) and .1 own evidence BCE. No geometry extra term, no
precision change. Auxiliary reference/event support follows the audited v2
source record; unavailable GT is never fabricated.

Only full-source adapter parameters trainable for backward; no parameter
updates. Capture all named raw gradients, component scalars, selected logits,
targets, original CE counts and input/support hashes. The B1 regularizer CE
must reproduce old original joint CE within 1e-6; exact equality is reported
when observed. Independent CPU float64 selected-support CE versus GPU
float32 tolerance2e-5 (both branches), grad finite/nonzero, PTD grads allNone,
state unchanged. Frozen nonparticipating branches may have None gradients.
CPU tests do not establish actual GPU success or training/native utility.

600 seconds engineering stage, <=80MB new evidence, 8GiB disk minimum,
GPU serial lease, cumulative cap=null. All worker/failed/import/check times
counted; prior total38820.55116519004s. No timer/push/promotion. On failure
keep immutable evidence/receipt and fix in a new registered version.
