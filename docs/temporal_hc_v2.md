# HC temporal calibration v2

User authorization: mechanism-led HC2 temporal tuning and a new HC1902 full evaluation.
The published v1 and its frozen runtime remain immutable. The VidSTG profile is unchanged.

## Claim and current failure

HC full tIoU changed by−0.0666pp (parent-cluster interval crosses zero); there is no
established temporal improvement. Full vIoU and unobserved space improved. Spatial
observed/guard/link weights do not enter the temporal posterior and cannot fix tIoU.

The old HC64 development set has only31 exact query IDs in officialHC1902, although
62 of its parent IDs overlap. Temperature1.5 improved that old development set but
harmed the prior32-parent recheck. We therefore lock a new input-hash selected official
96-parent panel, one query/parent:64 tuning and32 validation. This is already-exposed
target-domain development followed by a post-selection full rerun, not a pristine test.

## Fixed constraints and hypotheses

Source/native224/eight-observation ReferDINO/ERef/native complete boxes/propagation,
spatial score, physical sampling, FP32, old two-offset hull and original physical MBR
remain fixed. No actioness-based rule,448, conditional decoder, training or new teacher.

1. Endpoint temperature changes uncertainty and thus the hull/MBR decision, not just
   a global score scale. Softer scores can rescue wrong sharp peaks but can also expand
   coverage. Record entropy, expected/selected duration and short/long-event effects.
2. Six layers use the same frozen head and provide different predictions. The upstream
   implementation includes auxiliary temporal losses; early layers must not be called
   untrained. Fixed global layer weighting tests target-domain calibration of the ensemble.
3. Duration-power tilting acts on the already constructed physical posterior. Positive
   power suppresses long intervals; negative power supplies the opposite control. It
   can reduce overcoverage while damaging long events, so no gain is presumed.

Baseline parameters are temperature1, layer slope0, length power0. Coarse stages are
one factor at a time, followed by bounded neighboring refinement, one justified
combination and limited leave-one-axis controls, at most18 tuning profiles. Root reviews
actual results after each stage. One selected profile is tested on the fixed32 once.
Selection/validation protection rules are in [the locked protocol](../results/temporal_hc_v2/protocol.json).
All trial predictions seal before labels. No per-query GT branch or seed selection.

## Capture and evidence

The new panel uses the exact original frozen native capture provider and its original
expert fingerprint. Full v1 expert evidence is content-bound to RGB/caption/eight positions
and reused without weakening keys; new temporal results have a new code/plan fingerprint.
Source, ERef and native intervals must be bitwise identical to v1 before panel scoring.
All96 captures seal before the first tuning labels are opened. Compact logits/geometry
are retained locally; no raw tensors or query text are public.

The maintained main profile is not changed until a locked validation succeeds. A new
full queue must use a new output and immutable snapshot, not resume or overwrite v1.

## Outcome

The finite search and single fixed validation are complete. The selected temperature1.5
failed validation (t−1.5605/v−1.0859pp versus Global6). It is not promoted and no new
full evaluation is started. Default configuration remains unchanged.
[Complete signed results and mechanism analysis](../results/temporal_hc_v2/REPORT.md).
