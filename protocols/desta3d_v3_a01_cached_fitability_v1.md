# A0.1 fixed cached fitability triage

User authorized 2026-09-29. A0 H128/S200 train/dev medians .007694/.002401
failed the direction gate. Distinguish bounded capacity, optimization budget,
multi-query amortization and coefficient-field representation explanations.
These are diagnostic hypotheses, not unique causal identifications.

## Fixed evidence and execution

Reuse exactly sealed A0 Train128 (95 source parents) and Dev64 (16 exposed
diagnosis parents). Source GT supplied the cached privileged evidence and native
oracle direction. No new PTD loading, pixel decoding, native generation, backward
through PTD, label lookup or fresh31/388 access. Read only CACHE.pt, frozen BASIS,
rosters and audit metadata from A0. Old cache and scientific pins remain unchanged.

Frozen feature dimension128, union256, evidence8, native state33, original radius
.13545580427763146. Existing coefficient computation and query-global cosine loss.
Internal width is separate from feature_dim; width256 input is Linear(425,256).
Same construction/RNG order and nonzero output normal std1e-3; H128 must exactly
reproduce A0 initialization. Single seed20260928; AdamW lr.001, wd0, clip1,
batch4 accumulated queries, constant schedule. Only `1-cos(pred,oracle)`.
Same Python Random shuffle/pop order. Missing zero directions retained with zero
loss and unchanged batch denominator; counts reported. No best-step or grid.

## Three locked arms

1. W256-S200: only internal width128 to256, fixed terminal200.
2. W128-S2000: only steps200 to2000, fixed terminal2000. Save step200 strictly
   for exact continuation/control comparison with old A0, never for selection.
3. Q1-overfit: engineering control, H128 terminal2000. Pick minimum SHA256 UTF-8
   `DESTA-A01-v1|q1|<key>` across fixed Train128. Repeat this query four times per
   update to retain the original batch4 accumulation/divisor. Not a method score.

All arms start fresh from the same seed rule, never from A0 final. H128 and Q1
initial parameters must be exactly A0 initial. Evaluate final Train128/Dev64
query-global coefficient cosine means/medians, loss=1-cos, gradient norm range,
terminal training-batch loss, clipping and actual optimizer counters. For Q1
add its own terminal cosine/loss; full-set Q1 scores diagnostic only. Save every
terminal coefficient and full optimizer/initial/final state for independent
NumPy and live-Parameter/integer-key audit. Cached CPU contracts separate from GPU.

## Predeclared routing, no automatic next experiment

- W256 train median>=.3 AND dev median>=.1: capacity candidate, next only Dev64
  native. If both scientific arms pass, width has this predeclared priority;
  report both, do not select by better dev score.
- Otherwise W128-S2000 satisfying both: optimization-budget candidate, next only
  Dev64 native, no intermediate step selection.
- Both fail and Q1 cosine>=.9: stop width/step tuning, next candidate only
  global-context conditioning; single-query representability does not by itself
  uniquely prove the source of amortization failure.
- Both fail and Q1<.9: stop mixer capacity tuning; next cached oracle rank/
  smoothness/structure audit. This does not prove mathematical nonrepresentability.

Current task ends after these three cached fits and independent report. No native,
full618/full447, fresh388, TVG/SVG, trust gate, OPD, target or new architecture run.

## Resources and failure policy

Serial GPU lease, three individual finite900s allocations, disk floor8GiB,
total new output ceiling4GiB. Prior cumulative72405.61955377608s, cap=null.
Measure each worker including imports/cache loading/readouts and nonoverlapping
wrapper overhead; all failed/replayed time preserved. No silent retry. Old files
immutable; engineering repairs get isolated names. No user-visible timing promise;
cached computation expected minutes, preparation/audit accounted separately.
