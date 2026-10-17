# Fixed Vid128 Spatial learning-rate diagnostic

Human authorized a fast bounded parameter comparison on 2026-10-17.
Use all128 Vid parents from the completed T2+B core256 development cohort.
Keep original Source, frozen T2 outputs, marked inputs, Caption, WHO references,
query/round random samples, sigma .4, tau .5, maximum3 steps, Gaussian32,
1792 episodic parameters, response mu .125, original fullstep rejection/rollback
and complete output readout unchanged. Only Adam LR differs: .027 / .0135 / .00675.

Reuse .027 only as exact whole-query sealed predictions with matching input and
reference bindings. Run the other two arms serially, maximum256 new fits.
All128 no-reference/no-fit outcomes remain in the denominator. No new WHO or
Temporal fits. Both new arms' full prediction/root barriers and all128 first
rollout/gradient/initial-state matches to the current-LR control precede new GT
scoring. Report vIoU/tIoU/common dense sIoU, recall, all source-paired signed
deltas/unknowns/t95 intervals, acceptance counts and actual costs.

Select the largest all128 mean vIoU for this exposed development diagnostic;
ties within1e-9 prefer current, then half, then quarter. Publish every arm,
including negative results. Selection is not new independent confirmation;
post-selection intervals are descriptive. No automatic registry promotion or
subsequent mu/sigma/tau/step search. Do not retune using Confirmation128.

Finite2h deadline, one serial GPU worker; MemoryHigh16GiB/Max20GiB/Swap1GiB,
disk floor30GiB and original resource guard. Preserve partial results/failures.
Publish completed code/protocol/anonymous results via Zonglin-He/A after scalar,
actual pixel, privacy, remote SHA/blob/mode/tree/two-ref and archive checks.
