# DESTA-3D v2 optimizer persistence correction

This is an engineering correction to the authorized auxiliary-backflow contrast,
not a new architecture, hyperparameter search, or target experiment.

The original recursive CPU copy converted every dictionary key to a string.
AdamW state dictionaries require integer parameter IDs. Native load accepted
the string-key states without binding them to Parameters; the next update
created fresh moments. The prior CPU test compared serialized structures only,
which did not expose this error. A real post-resume check at B cursor352 found
logical step88 but actual Adam step2 and stopped contrast002 after97.878507206s.

## Evidence and recovery boundary

- Original directories, pins, predictions, logs and failed receipt remain intact.
- Common A in contrast001 completed continuously:618 source-train queries,
  95 Vid parents,155 updates. Its immutable weights are valid for this contrast.
- Original B344 safe checkpoint was replaced by the invalid resumed states;
  exact recovery of that checkpoint is unavailable. Do not claim otherwise.
- A new isolated `fit_recovery_v2` reuses the identical COMMON_A weights,
  creates fresh B0/B1/B2 optimizers and repeats the same one-B-epoch contrast.
  It does not retrain A. Seed20260927 is explicitly set at the new B entry;
  the unretained original B-entry RNG is not asserted to be recovered.
-618 queries/155 B updates per arm, last accumulation divisor2, same query and
  pixel pairing, same loss/gradient coefficients, LR/775-step schedule and
  fixed final states remain unchanged. No best selection or target labels.
-792 new source-val predictions must be sealed before source-val scoring.
  Frozen identities/physical pixels must be matched. Original complete-tube,
  parent-macro paired CI, event AUROC denominators, retention and tails apply.

## Implementation and acceptance

`vg_tta/optimizer_checkpoint.py` clones tensors while preserving key types and
rejects optimizer states with unmapped IDs. The isolated recovery entry imports
the unchanged worker and replaces only output directory, initialization and
checkpoint save/load functions before execution. Active/old pins are unchanged.
It verifies live Parameter-bound state and exact serialized equality on restore.
Each allocation's entry checkpoint is hardlinked into `resume_points` before
LATEST can be replaced, so the audited restart point remains recoverable.

CPU regression uses real hidden128 adapters and AdamW on all three arms. After
three synthetic updates, save/restore must produce exactly equal parameters,
moments and counters for TWO subsequent updates compared to uninterrupted
controls. CUDA is mocked only for CPU device/RNG entry; this is not a GPU or
task-efficacy test. The real recovery run will also be deliberately paused after
a small complete window and resumed, then checked for continued Adam counters
and moment binding before the long allocation.

## Historical claims requiring correction

Original source-fit002/003 used the same serialization/loading path; their
uninterrupted-optimizer claim is withdrawn. Stored predictions and measured
geometry remain actual observations of the implemented runs. Panel001's raw
gradients and measured Adam deltas remain records, but the claim that its
updates inherited the original moments is invalid. It is not evidence about
the specified inherited-Adam update until repaired/retested. No negative result
is deleted, promoted to a correct-protocol result, or used to reject the route.

Cumulative GPU cap remains null. All failed/replayed/loaded time is counted;
prior total at recovery registration is27390.239283847994s. Root serial GPU,
8GiB free-space floor and finite3600s review windows remain in force.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
