# Sparse-observation / hard-pseudo-label DESTA v1: expert interface review

Status: code and sealed metadata audited on 2026-09-29. Native grid is still
running. This review neither changes its scientific configuration nor reports
expert correctness, native gains, or a verified failure cause.

## What the current implementation does

- UniversalVTG encodes the existing PTD observations, then fills uniform physical
  2-fps slots by selecting the nearest observation feature. The local official
  `encode_video()` instead extracts its sampled frames from the video itself.
  `feature_fps=2` is an intended interface argument; repeated sparse content is
  the actual difference. Extra raw-video observations have not been tested here.
- GroundingDINO searches all existing observations with the query text. The
  highest-scoring valid box becomes the SAM2 bidirectional tracking anchor.
  SAM2 receives the same sampled sequence, not a dense original video.
- The spatial anchor search does not use the temporal expert interval.
- Scores are retained and used for selection. They are not used to modulate
  adaptation reliability or its absolute step/radius. Temporal scores also
  enter the expert's own candidate postprocessing. They are not literally lost.
- The spatial provider is a detector/tracker fallback, not a task-trained RVOS
  specialist. A negative result cannot establish that all SVG/RVOS guidance fails.

## Complete sealed metadata readback

The standalone standard-library reader verifies consumed evidence hashes,
roster pins, common physical/pixel support, and every nearest-observation index.
It does not decode media or read labels/native predictions. All 64 queries from
the fixed 16 exposed parents are included; none are selected by performance.

| Measurement | Saved evidence |
|---|---:|
| PTD/expert observations per query, min / median / max | 6 / 30 / 32 |
| Temporal slots per query, min / median / max | 6 / 30 / 152 |
| Queries with repeated observation slots | 31 / 64 |
| Per-query repeated-slot fraction, median / mean / max | 0% / 24.9794% / 78.9474% |
| Repeated slots / all slots, pooled across queries | 1573 / 3112 = 50.5463% |
| Median of each query's median observed-frame gap | 0.6006 seconds |
| Largest observed-frame gap | 2.46913 seconds |
| Largest slot-to-selected-observation time offset | 1.20761 seconds |
| Spatial anchor inside / outside predicted temporal interval | 32 / 32 |

Long clips contribute more slots, explaining why pooled repetition exceeds the
equal-query mean. “All experts see heavily repeated frames” would be inaccurate.
An anchor outside the predicted interval is not necessarily the wrong referent:
the referent may exist outside the event, and the temporal prediction may err.
The 32/64 count measures absent temporal conditioning, not a 50% identity error.

Reproduce the readback in a new output file:

```bash
python -B scripts/audit_desta_native_input_support.py \
  --directory artifacts/desta3d_v3/latent_oracle_v1/dual_expert_native_v1 \
  --output /tmp/desta-input-support-review.json
```

## Normalization and confidence: an important distinction

The current update first balances full-space branch gradients, then projects
their sum, then normalizes the combined coefficient direction. A common positive
confidence multiplier applied to that sum cancels under the final normalization.
Likewise, multiplying a branch loss by a positive scalar before normalizing its
gradient cancels within that branch. A single active branch has the same issue.

Branch-specific confidence weights applied *after* full-space normalization can
change the relative direction when both branches are active. They still do not
automatically reduce the absolute update length. A future reliability-dependent
step/radius or explicit no-op would be a separate scientific change. Detector
and TVG scores are not interchangeable calibrated correctness probabilities.

An individual branch's retained R16 fraction does survive full-space balancing
and affects its relative contribution before the final sum normalization.
Therefore it is also inaccurate to say every branch has equal projected force.

## Final-report diagnostic and decision boundary

After the existing stage seals/audits, add temporal/spatial retained-gradient
fractions from saved raw projections/full norms, with zero/missing support
reported separately. No new GPU backward or GT oracle is needed. Because the
stored QC basis is only numerically orthonormal, distinguish the coefficient
ratio from an exact column-space projection ratio. With `p = g @ QC` and
`G = QC.T @ QC`, the latter is
`sqrt(sum((p @ inverse(G)) * p)) / norm(g)` across the entire field.
Use all completed queries/states, separate initial states from trajectory
states, and avoid counting a shared initial gradient 27 times.

A small retained fraction is a geometric diagnostic, not proof of numerical
noise or wrong pseudo-labels. If the smallest tested radius wins, it supports a
local magnitude hypothesis but does not identify label noise as the unique cause.

Continue the registered 27 → 6 → 1 grid with unchanged expert outputs and pins.
Its parameter influence is conditional on these fixed sparse/hard-label experts.
Possible later controls are actual raw-video expert sampling, temporal-conditioned
spatial anchor selection, or calibrated confidence-to-guidance. They are proposed,
untested, and not automatically launched; changing all three together would not
isolate which change mattered. Keep PTD input support, backbone and R16 unchanged
if testing expert input alone, and account for added observations and inference cost.

Experts can provide different learned task information from the same pixels.
Dense input may help but is not a prerequisite for specialization, and it is not
yet an established fix for this experiment.
