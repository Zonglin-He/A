# Reproduction boundary

R129 requires the existing licensed 3EED workspace, three epoch100 source-only
checkpoints, prior R91 prepared modules, and the immutable R127 native-foundation
cache/readout interface. It is not a standalone checkpoint distribution.
Prior own-code dependency: studies/3eed_deployment_consistent_complete_readout_TTA_20261117
(R127, commit605d048c189471530c47876c228e06b0f7b14879), and
R123/R125/R97 interfaces linked there. Upstream 3EED comes from
https://github.com/worldbench/3EED and is not republished.

Order: calibrate129 (saved source-only labels/states); prepare129 (A/B only);
lock design and scope; check_math129; run129 --engineering; run129 formal;
seal every prediction; audit129 independent CPU; score129; audit_scalars129;
analyze129; report129; export129; audit_public129. Model, training and task-scoring
steps use experiment_guard_round129.py. No new source training occurs.
No target labels choose thresholds, candidate sets, trial endpoints or steps.
No C arrays are decoded until all parameter states for that input are fixed.
Archived R127 global states/readouts and protected-alias zero results are reused.

Private registry rows, original IDs, captions/media, coordinates, supports,
query indexes, features, weights and third-party snapshots are omitted.
Public ROW/CLUSTER tags identify only anonymous scalar-table positions.
The 25-arm full task tables retain every positive/negative/unchanged observation.
Source-calibration aggregate thresholds and anonymous gap rows are included;
the private source-file mapping is excluded.
