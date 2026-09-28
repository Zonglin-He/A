# DESTA-3D v2: one matched output-anchor factor

Question: can a frozen observed-input output teacher reduce the clean damage of
calibration-alignment while preserving its noise gain? This is the same 8 exposed
development parents / 24 clean-noise-blur episodes, not new independent support.

The six previous arms remain sealed. Reuse their predictions byte-identically
only after old seal verification and new actual pixel, feature, query, time,
sourcefit geometry, and teacher-logit identity checks. Add just one arm:
`calibration_alignment_output_anchor`. No GT or loss-based state choice.

Keep fixed B1, FiLM/LN 66816 parameters, frozen gates/backbone/heads/projections,
AdamW 1e-5 wd0 clip1, fixed3 steps, pre-gate losses and alignment .01 unchanged.
Teacher is the initial B1 on the observed input for each episode. The output
anchor also replays the observed input; the existing pre-gate student uses the
same mild photometric view as before. Corrupted episodes never use clean inputs.

Added loss is lambda times (categorical time KL + categorical coordinate KL).
Categories sum, endpoints and frame/coordinate slots mean separately. Lambda
is fixed before target execution to 6.248522551708088, the ratio of the existing
objective gradient norm to the summed output gradient norm on the predeclared
source-only key10016 after3 old calibration steps (probe005 raw audit). This is
one engineering scale choice on one source, not a representative/tuned optimum.
No target-dependent rescaling, hyperparameter grid, or best-step selection.

Native event reference/time tokens fix teacher-forced support. Each student
branch has a fresh multimodal KV prefill and exact official semantic/time/box
incremental probes. Only stateless MLPs are checkpointed; saved activations use
bounded8GiB CPU offload preserving strides. Bias storage is aligned without
changing shape, values or visible positions. Same-state native logits and real
backward passed on source probe005; CPU primitive checks are separate evidence.

Missing/invalid teacher event format: both output terms absent, run the unchanged
three pre-gate steps and retain the episode. Valid event plus missing/nonfinite
coordinate logits: time anchor only. Legal coordinate tokens with invalid zero
geometry remain in the coordinate anchor, matching the model's actual output
distribution. No filtered evaluation support. Actual code/support is recorded.

All3 steps save raw gradient vectors before clipping, actual Adam counters,
output KL, cache/injection/layout audits. Final native outputs, free intervals,
time/coordinate distributions, fixed-prefix coordinate KL, invalid geometry and
exact episodic reset are saved. New24 predictions and reused144 sealed together
before re-opening previously exposed targetGT for offline evaluation. Independent
scalar/vector geometry and second parent/CI/retention/tail readback required.

Readouts: clean harm relative to B1/control, noise and blur separately, corruption
parent macro, net Frozen, v/s/t, interval changes, native-good preservation and
>5pp parent tails. Suppressed improvement is a tradeoff, not hidden. No automatic
64-source expansion or production promotion. HC remains cross-dataset transfer.

Allocation is 3600s engineering guard, 8GiB disk reserve, cumulative GPU cap null.
All failed/replayed/import overhead receipts included. Preserve partials on error;
never silently restart or mutate pins. No target results existed at registration.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
