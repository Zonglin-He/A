# Reproduction and evidence boundary

R133 depends on immutable R132972f5055f5213a8a362ea85c4d7ed9e8f5593d1f,
studies/3eed_update_coverage_B_acquisition_loggap_validation_20261117, plus
R131/R130/R129/R127/R123/R97/R91 dependencies documented there. Upstream3EED:
https://github.com/worldbench/3EED. Licensed input data, source/prepared weights,
features/point indices/coordinates and adaptation states remain private.

Order: prepare133 -> model-code lock -> run133 --engineering -> formal lock ->
run133 -> seal all endpoints -> audit133 -> score133 -> audit_scalars133 ->
analyze133 -> report133 -> export133 -> audit_public133. Models/CPU scoring use
experiment_guard_round133.py; one exclusive GPU, no old paused goal resumed.

Fixed R132 LOG C/S/wrong-C/kappa/original B/interface4896/rho/featurekeep/Armijo.
1step is the actual original first normalized update reconstructed from saved
gradients and GPU norm/accepted fraction. All646x2 original delta3 reassemblies
are bit-exact. 3step readouts/states are inherited, not refitted. Only extra
steps4/5 are differentiated, starting from the actual existing delta3, with no
hidden optimizer state. Original3 readout engineering covers both saved-B and
recomputed-original-B paths. All1798 rows stay,646active/1152inactive.

Primary remains3 by predeclaration. 1/5 are sensitivity, never target-GT parameter
selection, per-case best outputs or per-direction configurations. Same per-step
rho means larger cumulative budget at larger step counts. Actual delta norms,
reconstructed path lengths, accepted radius sums and forward/backward counts are
reported. Incremental measured cache-reuse timing is not standalone TTA latency.
Correct/wrong controls have matched nominal budgets but can accept different
fractions. Identity wrong labels are not counted as effective new supervision.

All16182 task values use R131 saved all256-box IoUs after every new endpoint is
sealed; no raw target GT geometry/source label tuning/FINAL access. Old R132
49good/27bad records are all revisited post-seal, not selected to improve a
configuration. Their set membership/group switches do not certify real object
identity. All new negative/positive/no-op sensitivity outcomes are retained.

Historical target exposure and prior three-source pooled calibration remain;
this is not new blind validation or strict direction-isolated label selection.
Anonymous CONTENT/FRAME/CLUSTER IDs preserve aliases/descriptions/directions;
one Drone recording does not support population scene CIs. Export contains own
code/config and complete anonymous scalars only, no raw IDs/captions/media,
query IDs, tensors, weights, credentials or third-party snapshots. Verify all
immutable remote bytes/trees/parent before CASmain; retain local archive receipt.
No additional tuning, source training, loss or B3 combination is queued.
