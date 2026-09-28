# Joint mixer: first seed training complete, utility pending

> Update after both seeds completed: this is the preserved first-seed milestone.
> Seed2 revealed that early format failure can retain valid time logits; use
> the isolated v3 auditor for that case. See [both-seed audit](JOINT_LEARNABILITY_BOTH_TRAINING.md)
> and [completed confirmation result](JOINT_LEARNABILITY_CONFIRMATION.md).

Seed 20260928 completed the locked one-epoch source training: 618 queries from
95 Vid parents, 155 actual Adam calls, no empty windows; final window has two
queries. Seed 20260929 is running independently. The 447-query / 31-parent
confirmation and its 1,341 predictions have not been scored. This is training
completion, not privileged teacher advantage or target-TTA success.

PTD4B, B1 and the union256 basis remain frozen. Only the 103,424-parameter
mixer is optimized. CPU restoration of the final checkpoint verifies eight
integer-key Adam states bound to live mixer Parameters, all counters155,
finite moments, exact union basis, deterministic complete query order and
final/history consistency. All eight mixer parameter tensors changed. Frozen
PTD/B1 scope is supported by the locked worker assertions and per-window B1
hashes; this audit does not inspect the old worker's process memory or save a
new copy of every backbone tensor.

## Full training-support accounting

| Item | Count |
|---|---:|
| Training queries | 618 |
| Source parents | 95 |
| Committed windows / Adam calls | 155 / 155 |
| Clipped windows | 58 |
| Zero-learning-rate windows in the locked cosine schedule | 1 (final window) |
| Event support missing: no observed event frame | 6 |
| Event native support missing after generation failure | 1 |
| Spatial support missing: no annotated box at native anchors | 7 |
| Spatial native support missing after generation failure | 1 |
| Native format failures during training | 1 |

The missing-event and missing-spatial counts can overlap. Every query remains
in the epoch; missing support supplies no invented target and the accumulation
divisor stays the registered window size. One retained event generation
failure stops before the spatial pass, as the original decoder specifies.
The field is shared wherever both passes execute; a configuration flag is not
proof that a failed sample executed both passes. Actual correction/F norm
ranges from0 to0.1354551763, within the fixed0.1354558043 bound. These are
training diagnostics, not independent utility metrics.

The first CPU audit attempt incorrectly required event+spatial calls for every
query and stopped before publishing a report. Its original script and failure
record remain preserved locally. The isolated v2 auditor accepts the original
early event-failure contract only when both native supports are explicitly
missing and format is false. Predictions, samples, loss, metrics, scientific
pins and GPU execution were not changed or repeated.

The final checkpoint SHA256 is
`0126916f03ed7405d82a12846a577453373195d90ff5cf35d1de6948bf7ab01c`.
Three sequential engineering allocations and their separately measured wrapper
overheads total 8322.032339851 seconds. Settled historical cumulative time is
51455.210822529 seconds, cap=null; the running second seed is not yet charged using an
estimate. Raw checkpoints, labels, media and per-query histories stay local.

Audit entry: `scripts/audit_desta3d_v3_joint_training_v2.py --seed 20260928`.
This CPU audit was executed and passed. Reuse it on seed20260929 only after that
seed completes. Source GT is explicit privileged evidence and action supervision;
confirmation annotations were prepared earlier and there is no globally
untouched or pretraining-unseen claim. No expert, OPD, target, scalar-mask or
cancelled full-source training was launched.
