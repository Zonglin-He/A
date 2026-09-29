# Minimal C3-T: Frozen / Rerank / Hard / OPD

Completed on the fixed transient deployment setting. The question is whether adaptation adds value over expert reranking; no extra qualification gate or benchmark redesign was run.

16 previously exposed VidSTG parents, one query per source, 15 corrupted inputs plus clean per query. Percent values below; paired changes are percentage points. sIoU uses all valid GT-annotated observed frames; vIoU uses the corrected tube temporal union. Average the 15 corruption conditions within each parent, then bootstrap 16 parents (10,000 samples, seed 20260929).

| Arm | Transient sIoU | Transient tIoU | Transient vIoU | Clean tIoU | Clean vIoU |
|---|---:|---:|---:|---:|---:|
| Frozen | 45.3167 | 35.3431 | 16.0677 | 36.1754 | 16.7440 |
| Rerank | 45.3167 | 42.6600 | 19.3343 | 43.7381 | 19.6060 |
| Hard | 45.3626 | 36.8558 | 16.8038 | 36.8970 | 16.9554 |
| OPD | 45.3049 | 36.6961 | 16.2843 | 37.1625 | 16.5893 |

| Paired comparison | Transient tIoU change [95% CI] | Transient vIoU change [95% CI] |
|---|---:|---:|
| Rerank - Frozen | +7.3170 [+2.5797, +12.6979] | +3.2666 [+0.6539, +6.4638] |
| Hard - Frozen | +1.5127 [+0.1974, +2.9823] | +0.7361 [+0.2114, +1.3313] |
| OPD - Frozen | +1.3531 [-1.1532, +3.6807] | +0.2167 [-0.9966, +1.2301] |
| Hard - Rerank | -5.8043 [-10.9911, -1.2921] | -2.5304 [-5.7493, -0.0039] |
| OPD - Rerank | -5.9639 [-10.3589, -2.3079] | -3.0499 [-6.0893, -0.8172] |
| OPD - Hard | -0.1596 [-2.2292, +1.7527] | -0.5195 [-1.5699, +0.2353] |

## Interpretation

Rerank is at least as good as OPD on both temporal and tube mean scores. Retain simple reranking for this tested recipe; the one-step OPD update has not established added value. This does not rule out other adaptation objectives or scales.

Loss decreased after the fixed step in Hard 240/240 and OPD 240/240 transient cells. No step was selected or rejected by the loss or GT.

| Comparison | tIoU harms >5 pp (cells/240) | vIoU harms >5 pp (cells/240) |
|---|---:|---:|
| Rerank - Frozen | 23 | 19 |
| Hard - Frozen | 3 | 0 |
| OPD - Frozen | 28 | 17 |
| Hard - Rerank | 91 | 53 |
| OPD - Rerank | 84 | 52 |
| OPD - Hard | 32 | 21 |

Clean is a control for generic refinement, not a separate research line. Corrupted-minus-clean paired gain (vIoU): Rerank +0.4045 [-0.6671, +1.5420], Hard +0.5247 [+0.0638, +0.9952], OPD +0.3713 [-0.0504, +0.8530].

## Actual implementation

Frozen and Rerank share native boxes; Rerank chooses from at most eight unchanged TA-STVG decoder/legal-span candidates. The frozen UniversalVTG scores each student candidate by maximum confidence-weighted overlap with its proposals. The teacher interval itself is never emitted. The old clean expert outputs are reused; each corrupted expert output is cached once.

Hard: native Gaussian endpoint loss toward the critic-best student interval. OPD: mean pairwise logistic ranking loss over all strict expert preferences, using student log start/end probabilities. Both use one fixed normalized gradient step of .004 times the joint visual-H norm and edit only H_motion. H_app/H_text and all model parameters are unchanged. No backtracking, projection, protection gate, low rank, reliability network or sweep. The normal two-stage suffix reroutes dynamically and emits its native output; there is no post-update reranking.

The backbone is captured once per episode and is not rerun for gradient/update evaluation. Additional full forwards are solely exact replay and selected reinsertion checks. The existing official TTS/ASA detach Jacobian is preserved.

## Verification and limits

All 256 four-arm outputs were sealed before scoring; only the original 16 authorized GT records were retained from the historical container, solely for offline metrics. This remains a repeatedly exposed development cohort, not a fresh benchmark. The random burst and severity are unchanged even when the event is unaffected. There is one burst realization per source/severity. Spatial specialists and multiseed work remain backlog.

AUDIT.json records exact native reuse, two independent metric implementations, independent critic-score reconstruction, four full reinsertion checks, exact non-motion freeze and identical-input controls. PUBLIC_SCALAR_AUDIT.json reconstructs means/CIs from anonymous rows; CPU_TESTS.txt checks loss gradients, native hard loss and endpoint mapping. No production method was changed.

GPU process allocation: capture 213.159s, critic 172.812s, adaptation 421.973s; total 807.945s (13.47min). Includes loading, decoding, original capture smoke and any failed allocations; excludes CPU development/scoring and previously published clean-cache creation. This is not a fresh-input end-to-end latency benchmark.

The earlier C2.5/C3/C0.6 registration is preserved as superseded history. The latest user instruction cancelled C2.5 gates, C0.6 multiseed and conditional/backtracking rules before adaptation outcomes. Only the required candidate capture was reused. See protocols/tastvg_temporal_fourarm_v1.md.

Anonymous per-cell values, all per-condition summaries, paired contrasts and positive/negative examples are included in ROWS.json, SUMMARY.json, EXCESS.json and EXAMPLES.json. Media, captions, GT coordinates, checkpoints and raw feature/prediction caches remain local.
