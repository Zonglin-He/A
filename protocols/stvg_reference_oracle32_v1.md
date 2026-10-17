# Matched 2x2 reference intervention on the exposed32

The user explicitly authorizes target GT **only for this oracle diagnostic**.
None of these four arms is a deployable target-unlabeled TTA score: even the
DINO row uses a target-GT-availability support mask.

| Arm | Reference | Updater |
|---|---|---|
| A | Original admitted DINO Top1 | Original direct 5L1 + 2GIoU |
| B | Same original DINO Top1 | Original Gaussian OPD |
| C | Same positions replaced by exact target GT boxes | Same direct |
| D | Same positions replaced by exact target GT boxes | Same OPD |

Fixed cohort: the32 previously exposed chronological-prefix sources used in
rollback32 and joint-critic32, HC16/Vid16. The common support is exactly the
intersection of original admitted DINO positions and frames with exact GT boxes:
67 anchors,25 nonempty sources,7 empty sources. No interpolated GT, added anchor,
changed observation clock or per-arm support selection. All32 and the same25
nonempty subset are reported separately; the7 empty sources are exact no-ops.

Every arm starts from the same frozen Source1792 parameter state for its Query.
Query residual, all trainable decoder LN and Adam reset; no cross-Query or
cross-arm writeback. Same Source checkpoint, RGB, two-offset native frame grid,
observations, full-decoder interface and final iterate. Main time readout is the
same frozen Source Native interval; the same saved Source projection is a
secondary CPU-only readout. Neither time readout uses GT or adapted time logits.

Budget: HC Adam lr.01/40 rounds; Vid lr.03/10 rounds; betas(.9,.999),eps1e-8,
weight_decay0,1792 QR/LN. Direct preserves its original sum/4 normalization. OPD
preserves original mean-over-admitted-anchors normalization,Gaussian32 antithetic
samples, HC sigma.025/tau.05, Vid sigma.1/tau.25,detached IoU rewards and
control-variate likelihood. The same Query key/round seed is used in B and D.
Equal maximum rounds are not a claim of equal nonzero gradient calls: original
OPD no-information steps remain no-ops and are recorded. No learning-rate or loss
rescaling search, response gate, rollback, best-step selection, annealing or new
critic. Cross-updater conclusions apply to these fixed configurations and budgets.

All original Source and DINO evidence is hash-bound. Existing DINO references are
reused without new Teacher inference. A fresh Source capture32 is permitted to
reconstruct exact frozen replay inputs; zero outputs must equal sealed Source
boxes. All nonempty arm outputs must match a complete two-decoder readout, and
Source parameter hashes must remain unchanged. Empty outputs reuse Source exact.

Target GT coordinates/support are prepared in a separate CPU process. The model
worker consumes explicitly marked diagnostic inputs; its raw-annotation access
guard remains. Generic fitter flags do not override this GT-assisted provenance.
All128 logical outputs seal before comparative metrics are computed. Target
metrics never select parameters, step, learning rate, observations or a winner
for deployment. Dataset annotations can themselves have uncertainty noted in the
preceding diagnosis; oracle means the benchmark GT, not infallible semantics.

Primary comparisons are C-A and D-B (reference intervention), B-A and D-C
(updater conditional on reference), and (D-B)-(C-A) (factorial interaction).
Report Source and all four arms, paired per-source deltas,positive/negative/zero,
parent-level uncertainty, tIoU/sIoU/vIoU/R@.3/R@.5,observed versus unobserved-native
and benchmark-interpolated support, actual costs and no-op counts. Do not
generalize from this exposed development set or interpret oracle as a method.

Single serial GPU lease; MemoryHigh16GiB/Max20GiB/Swap1GiB,host floor6GiB,
disk floor30GiB,two-hour finite runtime. No other queue is resumed or signaled.
Qwen/full experiment pause and CURRENT/MAIN/PAPER registrations remain unchanged.
Root numerical/path/state audits, fixed real-pixel output review, anonymous public
export, remote HTTP/SHA/blob/fetch mode/tree/two-ref verification and archive
check/snapshot/check are required for completion.

```bash
.conda/tubedetr/bin/python -B scripts/run_stvg_reference_oracle32_v1.py prepare
.conda/tubedetr/bin/python -B scripts/run_stvg_reference_oracle32_v1.py measure
.conda/tubedetr/bin/python -B scripts/run_stvg_reference_oracle32_v1.py score
```

Existing artifacts are immutable. Failed/partial runs are preserved and never
silently restarted, reclassified as negative scientific results, or overwritten.
