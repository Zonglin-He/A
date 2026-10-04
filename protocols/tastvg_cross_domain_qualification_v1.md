# Clean cross-domain qualification of unchanged A

User authorization: attachment 6f0ec7a7, plus pipeline diagnosis. No method promotion.
This is a new namespace; `decota_full_matrix_v1` never launched and is not resumed.

## Fixed scope and inputs

Two directions only: official Vid-trained TA-STVG to HC-STVG-v2; official HC2-trained
TA-STVG to VidSTG. Reuse the historical 413-query/135-source HC parent and the
707-query/384-source Vid parent. Select minimum SHA256(index|caption|original_video_id)
within each source before predictions. All selected physical media hashes are distinct
within a direction. Both native TA loaders lowercase the caption; apply this native input
normalization after selection, while retaining the original caption hash and parent index.
Clean only, original parent sampling grids; no uniform64 revision.
Three source orders sorted by SHA256(seed|source), seeds 2026100401/02/03. 1557 arrivals,
390 scheduled expert arrivals; availability is arrival%4==0. Reset persistent state to
source at each order, not each query. Input dictionaries whitelist metadata and omit
labels, historical outcomes and reference predictions. The cohorts and source parameter
development are historically exposed; this is qualification, not untouched testing.

## Unchanged method and controls

Source Vid bundle: lr .033761698432507946, teacher temperature .34902548789596055, K1.
Source HC2 bundle: lr .006097133675874025, teacher temperature 1, K8. Both rho .05,
student temperature 1, four source-fixed probe directions, nine refreshed candidates,
1792 parameters (256 query residual and norm1/3/4 weight/bias). Uniform5 Sa2VA, Rank-RKL,
ordinary SGD, and original UniversalVTG max(confidence*interval-IoU) Fast are unchanged.
No LR/K/temperature search, no target labels in updates, no new scorer/loss/expert.

Frozen: source checkpoint, no update or Fast. Fast-only: source state, scheduled Fast.
Spatial-only: persistent spatial writes, no Fast. Full A: persistent writes plus Fast.
Target-trained Frozen: separate supervised reference, excluded from fair method comparison.
Full and Spatial-only share the same spatial trajectory because Fast only changes the
sealed interval and never enters the spatial loss; verify this against fast-disabled
live rollouts on the first two scheduled inputs and check all state chains. Fast-only
uses its own source-state native temporal candidates, not the inherited Full candidates.
Thus Full-Fast measures consequences of spatial history including its native temporal
readout interaction, not uniquely box-coordinate effects. Report fixed-interval splits.
All current outputs are sealed before persistent spatial writes. No probe offset persists.

## Execution, seals and diagnostics

Single GPU serial, after the existing correction-scope finite GPU queue completes.
CPU subjects -> spatial expert cache -> temporal expert cache -> source H/Frozen/Fast-only
capture -> target-trained Frozen -> persistent streams. Reuse only exact input/hash receipts;
cache requests across orders, never update/backpropagate expert models. Preserve full spatial
candidates, rewards, pre/post states, gradients, each inner-step output, and temporal support.
All direction outputs and reference predictions seal in a global barrier before CPU GT read.
Failure recovery retains original records and pins engineering revisions. No old queue restart.

GT-only posthoc diagnosis: ideal time/space conditional upper bounds; existing temporal and
spatial candidate oracle; correct candidate lost by expert selection versus unsupported
correct output; Sa2VA empty/event support, expert-frame GT localization; useful preference
not realized by update, loss decrease with GT harm; gross gain/loss and correctness changes
at .3/.5; severe >5/>20pp harm; inherited boxes/native interval/Fast and future nonexpert loss.
Keep good and failure cases, per-step K8 and net fixed-time utility. Ideal branch upper bounds
are conditional and cannot be added as independent causal contributions. Dataset differences
are interpreted alongside observed input/query/event characteristics, not assumed causality.

## Statistics and predeclared decision

Official dense scorers, target source macro; first mean the three eligible order cells per
source, then equal sources. Paired 10000 source bootstrap, seed 20261004; report individual
orders, SD, expert/nonexpert, recency 1/2-3/4-7/>=8 and no-prior-write. Empty bins are N/A
(scheduled-observation distance cannot reach >=8, but actual-write distance can when empty
evidence causes skipped writes; distinguish scheduled observations and actual writes).
Report vIoU/tIoU/sIoU/@.3/.5. Domain gap recovery denominator is Target-trained-Frozen;
report denominator and joint-bootstrap CI, mark nonpositive or <=1e-8 denominators undefined.
Do not clip recovery to [0,1] or treat supervised reference as a mathematical upper bound.
GO requires Full-Frozen paired CI strictly positive in both directions and Full-Fast CI
positive in at least one; otherwise report partial/inconclusive/NO-GO in scope. Nonexpert
persistent contrast is the principal transfer evidence. Never select the best arm, target
bundle or dataset-specific method after these outcomes. HC16/HC1/extra baselines require
separate authorization. Deliver code, protocol, anonymous full results/negative findings,
pipeline report/figures, independent audits, archive check/snapshot/check and verified GitHub.
