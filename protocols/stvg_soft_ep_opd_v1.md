# Soft EP-OPD: finite matched 32-source development protocol

## Scope

This protocol records the completed `primary`, `mu025`, and `mu4` trials in
`artifacts/stvg_soft_ep_opd_v1`. Each trial uses the same 32 historically
exposed development sources (16 HC and 16 VidSTG), the same sealed reference
lock, parent predictions, temporal configuration, and spatial configuration
except for the response coefficient `mu` in M2. The three tested values are
`1.0`, `0.25`, and `4.0`. This is a finite development comparison, not a clean
confirmation set, full-cohort result, or method freeze. The separately
authorized 128-source coordinate-search work is outside this protocol. At
the 2026-10-15 report cut, its source-capture seal contains all 128 sources
(64 new captures and 64 exact reuses), while temporal/teacher preparation is
incomplete and has no complete prediction seal or score. An initial attempt
stopped because `accelerate` was unavailable in that environment; this was
not a GPU memory failure. Its separately recorded continuation is not a
completed result and is not included here.

## Methods and fixed settings

The arms are `F`, `T`, `S_M0`, `TS_M0`, `S_M1`, `TS_M1`, `S_M2`, and `TS_M2`.
M0 is the saved, validated original current-5 fit/output. M1 uses the original
Adam direction with the same candidate acceptance and backtracking policy.
M2 replaces that direction with the saved soft decoder-response direction

`(I + lambda J_U^T J_U) d = d_Adam`,

where `lambda = mu * ||d_Adam||^2 / ||J_U d_Adam||^2`. Other fit settings stay
fixed by domain: HC uses `lr=0.003535533905932738`, `sigma=0.0125`, `tau=0.2`,
five declared steps, 32 antithetic samples, and writeback `0.0625`; VidSTG
uses `lr=0.0075`, `sigma=0.2`, `tau=1.0`, five declared steps, 32 samples, and
writeback `0.125`. Query parameters and Adam are reset per source; this is not
LN carry. The same backtracking schedule `[1, 0.5, 0.25, 0.125]` and locked
response RMS acceptance rule apply to M1 and M2.

For the `mu025` and `mu4` child trials, M1 is byte-exact reuse of the primary
M1 fit and readout; it is not counted as a newly executed fit. Their M2 fits
are new. Empty-reference sources remain represented with the locked no-fit
fallback rather than being replaced or dropped. The norm-matched comparison
is a saved per-step process-direction diagnostic and is not an additional
fitted arm.

## Scoring and aggregation

Scoring occurs only after each trial's prediction seal. The report retains all
32 sources, eight arms, and five metrics: `m_vIoU`, `m_tIoU`, `m_sIoU_dense_GT`,
`m_sIoU_supervised_GT`, and `m_sIoU_unsupervised_GT`. Null values remain null.
Each trial therefore has 256 source-arm rows and 80 domain-arm-metric summary
cells. The 12 locked pair definitions across two domains and five metrics yield
120 paired cells per trial. Paired differences are in percentage points; the
neutral tolerance is `1e-10` pp. Positive, negative, neutral, and unknown
source aliases are all retained. No bootstrap or confidence interval is
computed.

The report highlights full-tube vIoU and selected paired contrasts, while the
machine-readable export preserves every metric and contrast. Anchor-reference
and whole-tube metrics are not interchangeable.

## Saved-step audit boundary

The independent CPU audit checks saved reward/softmax/loss arithmetic, Adam
moments and direction, accepted parameter updates, rejected parameter and
optimizer rollback, lambda/rho, saved CG residual witnesses, and actual
equal-parameter-norm response records. It does not rerun backward, reconstruct
the full decoder Jacobian, or independently replay the full model. The CG
residual is checked against the saved `normal_applied` witness. The original
primary finite-difference qualification has six checks over three fixed
qualification cases; the relative discrepancies are retained in the report
and are not treated as high-accuracy full-Jacobian qualification. The two
single-factor child trials have no new FD checks. The earlier 13-scope
numerical qualification failure remains unresolved and is not converted into
a pass by these results.

## Execution-cost accounting

The anonymous process export retains sealed worker counts and worker-reported
stage wall times. Across the three trials, the barriers record 95 fit
invocations including primary qualification, 70 source captures, 92 actual
full-model final readouts, three cached bitwise M0 qualification replays, and
46 child M1 fit-record reuses. Primary qualification and development stages
reported 52.026 and 216.111 seconds; `mu=0.25` and `mu=4` development stages
reported 187.047 and 204.674 seconds. These stage times are not end-to-end
latency. Teacher and Temporal inference were reused rather than repeated, so
their prior cost is unknown here and must not be treated as zero deployment
cost.

## Interpretation limits

All sources belong to a historically exposed development panel. M2 gains over
M1 or M0 on this panel do not establish generalization or deployment value.
The lower equal-norm response is a process observation, not evidence of
better GT alignment. Reported GT-derived metrics are post-seal evaluation;
they are not used for fit updates, reference choice, or selection of a best
step. No result here changes `methods/CURRENT_METHOD.json`, freezes a method,
or completes the separately authorized 128-source search. Its current
preparation status is disclosed above; this report contains no 128-source
scores.
