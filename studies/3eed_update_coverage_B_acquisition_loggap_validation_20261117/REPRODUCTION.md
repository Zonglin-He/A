# Reproduction and limits

R132 depends on immutable R131 aa2a66ca04d5caf41471e718da2c0e5466d80a12:
studies/3eed_fixed_partial_content_group_expansion_20261117, plus the R130/R129
native source/checkpoint/context dependencies documented there. Upstream3EED:
https://github.com/worldbench/3EED. Licensed media, captions, identifiers,
coordinates, support tensors, features and source/prepared weights are private.

Order: diagnose132 (noGT reason/decision seal, then attach old scalars) ->
prepare132 -> registration code lock -> register132 -> source execution lock ->
source132 -> audit_source132 -> calibrate132 -> scopes132 ->
audit_calibration132 -> target execution lock -> run132 --engineering ->
formal lock -> run132 -> all predictions sealed -> audit132 -> score132 ->
audit_scalars132 -> analyze132 -> report132 -> export132 -> audit_public132.
All models/CPU scoring use experiment_guard_round132.py; one exclusive GPU.

B3 first candidate/old original4results are reused, no old baseline rerun.
Among at most3 preordered distinct candidates, select first passing ORIGINAL
forward/reverse/cycle gates, without scores/GT or threshold relaxation. Every
within-B3 control uses the same selected B. Additional registration costs count.
LOG uses original B ONLY and one source-own real-AB 90percentile log-gap threshold;
no compound B3+LOG arm. Source calibration excludes known protected and R131 A/B
point-content groups before media/model/label reads. All313 source predictions
are frozen and audited before source labels determine kappa. No source weights
are trained; no target GT/calibration/coverage-based parameter choice occurs.

The log policy includes source-AB recalibration; old tau was sourceA-only. A
task difference is not isolated proof that log units alone outperform every
probability-gap calibration. Flat predictions remain unknown; no forced
negative or intersection rule is introduced. Wrong controls preserve S, C
cardinality and group-size counts, but not prior probability mass/q0 inclusion.

All1798 exposed R131 inputs remain, including unknown/no-update/harms. This is
development, not blind independent confirmation or new-scene validation.
Anonymous CONTENT/FRAME/CLUSTER labels preserve dependence across directions;
one Drone recording does not justify scene-population confidence intervals.
Raw target GT/geometry/FINAL are never newly opened or scored; saved R131 all256
IoU vectors are indexed only after the complete target matrix is sealed.
Query switches and IoU switches are separate; geometric groups are not identity
truth. All positive/negative/no-change rows and registration failures are kept.

Only own code, locked protocol and complete anonymous scalar tables are public.
No third-party snapshot, private tensor, raw identifier or credential is exported.
Pre-CAS immutable remote byte/tree/parent audits and archive receipts are retained.
No new experiment, combined configuration or tuning follows automatically.
