# R44: ordinal old-G feedback passes direct use, but this matching-layer transfer does not

The frozen old G contains useful target-domain decision information on the exposed A208 development set. After requiring its nomination and the frozen source choice to belong to two independent complete instance groups, direct use improves balanced mIoU from **19.663563% to 24.332121%**. The finite matching-MLP adaptation tested here does not carry that gain into the held-out-from-fitting V64 observations: mIoU falls from **28.063498% to 25.400646%**, and it also trails the same-information all-A memory control at **25.866269%**. **Stop this fixed 16-step configuration.** Retain the positive direct-G evidence and all unsuccessful adaptation results; do not expand its temperature, steps, layers, or holdouts under this registration.

This is an exposed-development experiment, not an independent confirmation. A208 and the original R24 V64 have appeared in previous work; FINAL384, U and T were not used here. V64 was not used to construct the 32 preferences or fit either head. No new VLM calls, source training, or 3D box updates were made.

## What changed and what was fixed

R43 showed that replacing only the dense-KL student coordinate with exact pre-probability log-odds can severely disrupt calibration. R44 asks a different, bounded question: can a **relative old-G nomination** provide useful supervision without prescribing its absolute posterior distribution?

The old G remains the original literal float64 expression `source_probability + 2*vision + 2*position + 2*relation`. Its raw nomination `qG` and the frozen source `q0` must have a strict G preference and belong to unique, nonempty, distinct complete R37 mask-to-3D groups. Queries belonging to the same group are not negatives. The output box is always the group's original source-score-best representative. Full source 256-box and 256-score compatibility with R37 was checked exactly before reuse. All 208 unrestricted old-G query/box outputs exactly replayed the historical cache; three float64-versus-float32 teacher argmax differences were retained as diagnostics rather than used to select a representation.

Phase 1 produced 32 accepted preferences, 14 Drone and 18 Quad. Its predeclared progression rule was `eligible count > 0 AND whole-A208 balanced gain > 0`, which passed. The rule only allowed this finite phase-2 diagnostic; it did not certify semantic identity or robust transfer. All 32 preferences were used unchanged, including the task-harmful one and six query changes with tied IoU. Task IoU was never used to accept, reject, or weight them. Different detector-mask groups are an instance-association hypothesis, not gold physical-identity labels; improved IoU alone does not prove semantic identity correction.

Phase 2 fitted one original source-initialized query projection MLP per platform: 288→288→288→64, 184,960 parameters. Everything outside this cached matching interface, including source text features and boxes, stayed fixed. AdamW used learning rate 0.0005, zero weight decay, seed 0, and exactly 16 updates per platform. The preference objective was group-max RankNet on exact pre-probability noun log-odds, with temperature 0.1 and fixed denominators 14/18. Relative-score output preservation with beta 1 acted on the exact A\E complement, 90 Drone and 86 Quad observations, anchored to each observation's frozen source `q0`. No dense posterior KL, new hinge, parameter anchor, teacher/EMA, or additional negative candidates was introduced.

On A, adapted group-max scores select a fixed original source representative, retaining ungrouped queries; the raw query winner is separately recorded. On V, the final readout remains the original native float32 noun probability. Thus this is not the stable-readout intervention stopped in R41. The new feedback collection, its complement and denominator, and the complete-group A readout change together relative to R29/R30; this experiment is not a single-factor explanation of their difference.

## Whole-set task results

All entries are mIoU in percent. Platform-balanced averages give equal weight to Drone and Quad. Acc@25, Acc@50, scene-cluster tables, and all per-observation outcomes are retained in the companion JSON files.

| A208, 104 observations per platform | Drone | Quad | Balanced |
|---|---:|---:|---:|
| Frozen source | 12.742985 | 26.584141 | 19.663563 |
| Eligible G, direct fixed source representative | 15.654468 | 33.009773 | 24.332121 |
| Unrestricted historical old G | 15.978627 | 33.476120 | 24.727373 |
| TTA, group-max to fixed source representative | 12.510217 | 26.789717 | 19.649967 |
| TTA, raw native-probability winner (diagnostic) | 12.294377 | 26.812613 | 19.553495 |

Direct eligible G improves 25 observations, harms one, and makes six IoU-neutral changes. The adapted group readout improves 20 and harms 24, while changing 89 queries. Its balanced difference from frozen is **−0.013596 percentage points**. The direct-feedback gain was not recovered by fitting, even on the fitting observations as a whole.

| V64, 32 observations per platform | Drone | Quad | Balanced |
|---|---:|---:|---:|
| Frozen source | 24.512637 | 31.614359 | 28.063498 |
| TTA | 21.839325 | 28.961968 | 25.400646 |
| Same A preferences, E-only memory, no update | 19.021047 | 27.525503 | 23.273275 |
| Same A preferences, all-A memory with zero U values, no update | 20.733564 | 30.998975 | 25.866269 |
| Current V old G, direct, no update (extra current information) | 26.723544 | 35.634447 | 31.178995 |
| TTA plus the same current V G increment (extra current information) | 24.541554 | 32.982055 | 28.761805 |

TTA loses **2.662852 pp** relative to its own frozen source and **0.465623 pp** relative to the all-A zero-U memory control. It exceeds E-only memory by 2.127371 pp, which is insufficient for adoption given the stronger controls. Its 29 V query changes contain seven improvements and eleven degradations. Both platforms degrade relative to source: Drone −2.673313 pp and Quad −2.652392 pp. With extra current G evidence, adaptation still loses **2.417191 pp** relative to directly using that same current G. The latter two arms have information unavailable to the A-only transfer arms and must not be pooled into an adaptation gain.

The nonparametric controls retain native probability units throughout. Each preference contributes the original R30 half-margin correction to its positive/negative groups. Source query×normalized-noun keys retrieve the nearest query per A observation, then CPU float32 top-4 observations with the fixed exponential cosine kernel at temperature 0.1. E-only and all-A-with-zero-U libraries are separately reported. No probability correction is added to log-odds. Actual neighbor rows, query indices, similarities, weights, all 256 scores, and zero-U values are saved. No extra competitor becomes a negative merely because it wins after adaptation.

## Optimization and measurement

The head can change its outputs and both finite objectives decrease; the preferences are not all implemented within this budget. Drone's total loss falls from 32.614586 to 2.795977 (preference 0.478369, preservation 2.317608), with 9/14 positive group log-odds gaps. Quad's falls from 58.989407 to 14.014096 (preference 8.248947, preservation 5.765149), with 10/18 positive gaps. Parameter L2 changes are 1.594423 and 1.624339. The 19/32 satisfied pairs are a fitting diagnostic, not a claim that 19 final targets are correct. Incomplete fitting and interference remain competing explanations; this finite run does not isolate their contributions and does not establish a matching-interface impossibility result.

Source native probabilities replay within 1.05e−6, with exact frozen winners. Both heads and every A/V/memory prediction were sealed before the independent task scorer ran. Two heads and 32 updates, including all prediction/memory work, took **5.235 seconds** in the guarded process, with 352,381,952 peak allocated GPU bytes; the outer timeout was 300 seconds. GPU ownership was released immediately after sealing.

Independent CPU audit passed: 272 cached head vectors; maximum native-probability error 1.24e−6 and log-odds error 6.82e−6; zero CPU64-versus-author query differences. The auditor recomputed **503,316,480** float32 similarity cells for all 128 memory retrievals, verified within-row maxima and legal top-4 membership, and checked values, weights, corrections and scores exactly. Two top-4 boundary ties were retained; saved and recomputed CPU32 indices differed on zero rows. Original boxes were exact for all 1,424 scored selections. All 624 prior phase-1 task values replayed exactly. All **622 negative ordered comparisons** are retained; none was removed to improve averages.

## Decision and limits

**Do not adopt this R44 phase-2 configuration or spend FINAL384 on it.** The first-stage direct-use mechanism remains useful on A208, but this finite shared-MLP fit neither reproduces its whole-A gain nor improves V64 over frozen and the stronger same-information memory control. The result does not justify deleting particular feedback using task GT, adding third-candidate negatives, claiming RGB absence as the cause, or declaring all ordinal adaptation ineffective. No step/temperature/layer search follows this result under the current lock.

The reported implementation is a cached matching-head experiment. It has no new full-grounder closure or deployment adoption claim. A future mechanism would require its own prior registration and matched controls. R24/R30 are historical system references; R44 is not a pure supervision-only comparison with either.

Artifacts: `PLAN_LOCK.json`, `IMPLEMENTATION_LOCK.json`, `ROOT_EXECUTION_AUTHORIZATION.json`, `TASK_SUMMARY.json`, `SCENE_ROWS.json`, `ALL_ROWS_PRIVATE.json`, and `ALL_NEGATIVES.json` in this directory; phase-1 qualification and direct-use results in `../qualification` and `../phase1`; independent arithmetic, loss, neighbor and task audits in `../independent_phase2`. Private feature, state, identifier and prediction payloads remain local and are excluded from public export.

Prediction seal SHA256: `5f45f1af9ed8a7885175579293e1732e6b33c42640213a490ddeaeb218bce23f`.

Mechanism provenance: the pairwise logistic objective follows the RankNet family ([original paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)); the relative-score preservation is this project's adaptation of output preservation. Neither is claimed as a new algorithm or a complete reproduction of an external TTA paper.
