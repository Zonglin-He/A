# Reproduction and study boundary

R130 fixes R129's unguarded partial_F at original rho as its development primary.
This is a new explicitly authorized follow-up, not a retrospective rewrite of
R129's failed protected primary. Same source-only epoch100/prepared R91 states,
fixed A256/decoder anchors, registered A/B and complete FAB readout. Only4896 Up
parameters, original feature keep/3steps/Armijo. No source training or expert.

Own-code dependency: immutable R129 commit1dd3ff25b0d22ddc34001143d81a282d9f718e13,
studies/3eed_spatial_support_partial_label_TTA_20261117, and its R127/R123/R97
dependencies. Download upstream native3EED from https://github.com/worldbench/3EED;
private licensed data and prior source/prepared weights are not republished.

Order: prepare130 CPU without GT -> lock design -> run130 --engineering ->
lock execution -> run130 formal -> seal all predictions -> audit130 CPU ->
score130 -> audit_scalars130 -> analyze130 -> report130 -> export130 ->
audit_public130 -> immutable remote byte/tree/parent verification -> CAS main.
All models and CPU scoring use experiment_guard_round130.py. No completed
baseline leaderboard is rerun. Original partial states/predictions are reused;
twelve active engineering replays check exact original-state compatibility.
No C readout opens before new states are fixed. C never selects an update.

The wrong set keeps S and per-group-size-bin C counts exactly fixed; swaps
retained/excluded identities maximally where feasible. Five identity cases
retain exactly the same update and are not counted as effective shuffles.
Both full denominator and252-effective control results are retained. Initial
mass/q0 inclusion differ naturally; correct>wrong does not alone isolate
cross-modal identity evidence from semantic-confidence preservation.

DEV/EXP are exposed development, not independent confirmation. rho0 stays
primary; .5/2 endpoints are all reported, without target-specific selection.
The old18 good/bad records are post-seal explanatory audits, not rule design
or deployment filters. Anonymous frame/scene clusters expose dependence.
All negative cases are retained. FINAL receives no new access/scoring.

Export excludes captions/media, original IDs, coordinates/support arrays,
query indexes, features/weights, credentials and third-party snapshots.
Public ROW/FRAME/CLUSTER values are anonymous table-position labels only.
