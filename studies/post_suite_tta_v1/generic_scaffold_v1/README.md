# Generic TTA preparation: Tent, SAR, EATA, MEMO, CoTTA

This is the primary preparation track following the user's latest request.
Domain-specific sources remain preserved as secondary references.

The scaffold performs source integrity, dependency and interface checks only.
It imports no upstream algorithm or framework and does not load weights, videos,
captions, project labels or a GPU. Actual model adaptation is still deferred
until the entire preceding experiment scope has completed and been published.

```bash
outputs/post_suite_tta_v1/scaffold_env/bin/python -B studies/post_suite_tta_v1/generic_scaffold_v1/cli.py doctor
outputs/post_suite_tta_v1/scaffold_env/bin/python -B studies/post_suite_tta_v1/generic_scaffold_v1/cli.py configs
outputs/post_suite_tta_v1/scaffold_env/bin/python -B studies/post_suite_tta_v1/generic_scaffold_v1/cli.py gate
```

## What the adapters must preserve

Tent/SAR/EATA need an explicit differentiable model interface and declared
normalization/update scope. Official Tent and EATA primarily configure BN2d;
SAR supports BN/LN/GN. Applying them to a LayerNorm temporal decoder is a
declared STVG adapter, not an unchanged full-network reproduction. No wrapper
may silently return zero trainable parameters or use stale pre-update logits.

EATA additionally needs the original Fisher/anti-forgetting setup. An absent
Fisher term cannot be silently called complete EATA. ETA, if used, is labeled
as the separately declared ablation.

MEMO requires real augmented inputs and marginal-output entropy; cached single
head inputs do not become augmentation views. Spatially referring words and
temporal coordinate alignment constrain the augmentations. Include the same
augmentations without gradient updates to isolate ensembling gains.

CoTTA requires teacher/EMA and restoration state, plus compatible transformed
inputs. A continual stream and per-query reset are different protocols; label
an episodic variant explicitly. Keep stream ordering and before/after controls
fixed if a continual track is used.

## Readiness is not a result

Source commits, per-file SHA inventories, dependency snapshots, separate runtime
slots and method config templates are prepared. Upstream installation scripts
and shell commands are never executed automatically. GPU dependencies and
gradient bridges are not validated by these static checks. The scaffold has no
experiment launch command; a later independent frozen execution plan is needed.

All positive/negative outcomes and costs remain required. Do not tune until
outputs differ, select by full final scores or turn a joint-readout gain into
an optimizer gain. Source-native, same-readout no-update, adaptation-after-update
and augmentation-only controls are distinct.
