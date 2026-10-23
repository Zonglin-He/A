# SimplePath + video-wide Native2D feedback

## Fixed computation

1. Run the original frozen TA-STVG Source224 inference unchanged. Its final-layer FP32 start/end logits from both native offsets form the original physical-hull prior q0. Global6 is not a production input.
2. Use the same eight ReferDINO observations and source-calibrated shared geometric initialization. For each sampled frame, retain the nearest past and future observations. The original two-way affinity product and valid axis-affine fit transform each complete expert box. Keep the geometric mass/variance/scale/intersection checks and initialization fallback.
3. Score each candidate by its mean available-anchor appearance cosine. Connect adjacent complete candidate boxes with their IoU. Exact DP maximizes .5 observed mean + .5 supported-unobserved mean + .25 supported adjacent-link mean. Denominators are the original event observed/unobserved/edge counts. Removed: round-trip reliability thresholds and weights, extent RBF, deviation-from-initialization guard, adjacent feature/mask transport. Both geometric directions remain.
4. Run the same SimplePath over the full sampled timeline to obtain an auxiliary trajectory. Pool the original input_proj appearance grid **before text fusion** inside each complete box expanded by the fixed1.5 context. No new representation model is evaluated.
5. Center/L2-normalize the pooled features over the whole video. Native q0 supplies soft event membership. Estimate **one event/background prototype pair from all frames**, both shrunk toward the video's common mean with kappa64. No splitting by offset or cross-fitting remains. The ratio of shifted cosine compatibilities gives per-frame feedback. This is within-video self-consistency with self-inclusion, not calibrated event probability or independent validation.
6. Sum frame feedback over each legal inclusive interval and apply ONE update proportional to q0 times exp(score), beta1. Use the original physical MBR, then SimplePath at the resulting interval.

The original Source's two-offset sampling is unchanged; removing offset-wise prototype fitting does not remove Source offsets. Network parameters, teacher evidence, sampling, precision, seed and fallback rules remain fixed. No parameter search is part of this adoption.

## Evidence and limits

The user explicitly adopts both simplifications after the exposed modular experiments. On the67-query HC availability panel with old time fixed, SimplePath changed full vIoU +.015794pp/unobserved sIoU +.208066pp (CIs include zero), observed sIoU -1.155595pp. The one-pair temporal arm on exposed128/domain changed HC tIoU/vIoU +.641039/+.302139pp and Vid -.052664/-.043750pp relative to the old method. Severe Source-relative t-or-v union increased19→23 in HC and21→25 in Vid; Vid short-event Source-relative tIoU was -.006256pp. These outcomes remain intact. They are modular tests, not evidence for combined full performance.

The new auxiliary path differs on all67 audited inputs. Old pooled temporal features are therefore invalidated. Unchanged geometric candidates and exact Source/E/native logits/eight positions may be reused with input/hash bindings; actual pre-text grids are repooled on the new path. A real cold/cache bridge verifies Source/q0, raw prefix cells, old-pool recovery and coupled new output. No new expert is called for collection.

Each complete official direction must still be evaluated against the fixed query-equal Source+.9pp m_tIoU requirement. The prior HC1902 full failed (+.747072pp); it remains failed. The new method is selected by the user, not retrospectively declared qualified. Full results cannot select parameters.

## Research-only diagnostics and interfaces

The full collector optionally saves propagated-mask candidates for the fixed range analysis using the already computed anchor/frame affinities. These masks never screen, weight or connect the production path. This optional diagnostic work is not presented as necessary deployment cost.

The reusable feedback interface requires spatially supported native local representations and explicit native temporal candidates/priors, with coordinate/time adapters. Feature dimensions and class vocabularies are unrestricted. Only the TA adapter has been verified; cross-backbone effectiveness is untested.
