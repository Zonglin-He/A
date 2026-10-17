# Original256: marginal interval training with fixed I1/I2 evidence

Human authorization follows 8cae6e39 negative I2-only ranking experiment. Same
HC128/Vid128 exposed development parents, cross-domain TA-STVG Source, fixed
I1 and I2, actual videos, frozen R1 feedback/rewards and original hyperparameters.
All previous results remain. No new Teacher calls, captions, selection, Prompt,
Source capture, Spatial/Full, external Actionness, sweep or CURRENT promotion.

Only training probability representation changes, consistently for prior,
Teacher target and Student candidate likelihood. For a fixed physical interval:

    P_theta(I) = sum_{z: merge(z)=I} product_o p_theta^o(z_o)
    p_theta^C(i) = P_theta(I_i) / sum_j P_theta(I_j)
    q_T(i) proportional to P_N(I_i) exp(r_i/tau), tau=1
    L = KL(q_T || p_theta^C) + sum_o KL(p_N^o || p_theta^o)

The target is frozen; all compatible latent states contribute gradients.
Compute exact log-sum-exp of four disjoint boundary-supply partitions; no
probability subtraction/clamping or representative-state approximation.
Independently verify mass/objective/gradients by explicitly enumerating state
pairs, Torch autograd and finite differences, including overlapping-offset
endpoint ties. Full Native-grid trust remains the original term, coefficient1;
zero-init512, no L2, LBFGS50/Armijo/history10/tolerances/rollback unchanged.
Missing/truncated/equal signals retain exact Native, with all sources retained.

Final Student readout is the unchanged actual FP32 two-offset Native decoder
after inserting512 parameters. Do NOT switch adapted output to marginal MAP.
Retain unchanged Source marginal-MAP with zero updates as a separate control.
This distinction prevents free decoding gains from being attributed to TTA.

Copy immutable prior I2-only input, no-update and Teacher barriers byte-for-byte
after full file/pixel/Source/query/clock binding. Reuse all512 description/support
pairs; no model reload or Teacher qualification repeated. Refit all256 with the
new objective, then perform actual integrated full-grid readout. Two historical
absent Source prefixes receive same-RGB frozen forwards with Source logits,
hidden/boxes/records/Native exact. Full Source boxes must stay unchanged.

Seal all new Student and control predictions with ROOT before new GT evaluation.
Six arms: Frozen, T2, original B (B0), I2-only B (B1), marginal no-update, new
marginal-trained B (B2). Primary B2-B1 isolates training representation;
report B2-B0/Frozen/T2/no-update, vIoU/tIoU, dense and common-time sIoU, recalls,
all source-level signed/unknown results and paired source t95 by domain. No GT
may control selection, feedback, optimization or acceptance; previously exposed
development GT only after seal. Blind SHA binding of old results does not parse
their scores. No independent-confirmation or cross-backbone claim follows.

One finite serial unit, 4h cap, MemoryHigh16GiB/Max24GiB/Swap1GiB; live host6/GPU4/
disk30GiB floors. Old queues remain untouched. Publish configuration/code/full
anonymous results, negatives/limits; exclude text/media/GT coordinates/weights/
raw tensors. Inspect actual fixed inputs and chart; verify remote HTTP bytes/
SHA/blob/fetched mode/tree and both refs fresh CAS. Archive check/snapshot/check
and close only this finite scope, with no automatic follow-up or promotion.
