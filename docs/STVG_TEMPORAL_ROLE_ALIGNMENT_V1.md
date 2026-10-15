# Temporal question-role alignment: fixed P1 development result

P1 completed the fixed alignment-only comparison. It improves temporal candidate reward ordering on the evaluable support but does not improve the final question-group vIoU. The prelocked P2 condition is therefore false: no relation-aware recaptioning or additional prompt search was launched. The original P0 remains in use while the separately authorized spatial learning-rate search continues.

## Matched final predictions

All values below are percentages. Each source contributes one query. The 64 VidSTG sources are development data already used in analysis; they are not an independent confirmation cohort.

| Cohort | Method | vIoU | tIoU |
|---|---|---:|---:|
| VidSTG questions, 28 | Frozen | 7.339506 | 33.168457 |
| VidSTG questions, 28 | P0 temporal | 6.505593 | 31.764705 |
| VidSTG questions, 28 | P1 role alignment | 6.443424 | 32.085098 |
| VidSTG all, 64 | Frozen | 12.625615 | 35.244679 |
| VidSTG all, 64 | P0 temporal | 12.706507 | 35.256417 |
| VidSTG all, 64 | P1 questions / P0 declaratives | 12.679308 | 35.396590 |
| VidSTG declaratives, 36 | P0 and P1, identical | 17.529441 | 37.972194 |

P1 minus P0 on the 28 questions is **−0.062169 percentage points vIoU** and **+0.320394 points tIoU**. Four question sources improve vIoU, four worsen and 20 are unchanged, including the original-input fallbacks. tIoU improves for five, worsens for four and is unchanged for 19. On all 64 sources, the changes are −0.027199 points vIoU and +0.140172 points tIoU. Dense spatial IoU is exactly unchanged, as expected from the fixed spatial boxes. HC predictions and all VidSTG declarative predictions retain P0.

The differing tIoU/vIoU movement does not establish that role prompting is universally ineffective. Better interval overlap need not improve complete spatio-temporal overlap when spatial quality varies over time. This finite test establishes that this fixed alignment-only replacement is insufficient to adopt under the prelocked final-prediction rule; it does not isolate the cause of every negative case.

## Candidate ordering

On 18 questions with common non-tied candidate-quality support, query-macro pairwise ordering agreement rises from **56.944444% to 65.277778%**, an increase of **8.333333 points**. The accounting is 99 evaluable candidate pairs, 33 GT-quality ties, 18 unknown pairs, and 10 questions with no valid non-tied comparison. This conditional ranking result must not be described as accuracy over all 28 questions or as complete-Tube improvement. It is postseal diagnosis, never an online candidate Oracle.

## Actual execution and verification

R1 loaded once and performed 88 text-only alignment forwards using the same inherited captions; zero new captions were generated. Three nonempty truncated captions were used by P0 and remain included and flagged in P1. There are 106 actual candidate positions: 88 available and 18 original-input unknown, with six further positions not instantiated in the nominal candidate budget. All 28 question sources remain in final evaluation. No missing feedback is filled with zero.

The current 512-parameter Offset-Trust interface performed 22 new CPU temporal fits from saved model captures. Twenty-two original P0 fits were replayed before launch and recovered all intervals exactly; maximum theta discrepancy was 1.5810797116e−11. The P1 readback repeated all 22 new fits, with theta discrepancy zero. This is cached temporal-head execution, **zero new full-backbone Student GPU forwards**, not a claim of new full Student inference. Native spatial boxes, candidate intervals, source states and spatial reference clocks remain fixed.

All 64 prediction files, source/probability checks and temporal replays were sealed before official GT evaluation. Maximum independent probability arithmetic discrepancy was 9.2572203458e−8. A separate saved-scalar audit recomputed all 192 arm rows into 27 summary cells and nine paired contrasts, including every negative and unchanged source, with aggregation error zero. No bootstrap, GT-reference fit, new Teacher, tracking run, CURRENT promotion or full evaluation occurred.

The first prelaunch reader assumed four candidates even for six three-candidate fallback sources. That runtime and source remain preserved; no model or GT execution occurred under it. The separately locked revision002 corrects candidate-length accounting, empty-description handling and complete-support checks. It does not change candidate generation, the fitting algorithm or the source evidence.

## Decision and ongoing work

P2 required full 28-question vIoU/tIoU support, positive mean P1-minus-P0 vIoU and nonnegative mean tIoU. Support is complete, but the vIoU sign is negative, so P2 was not run. Spatial coordinate search keeps its original temporal lock and separately tests Vid learning rates 0.06 and 0.09 with matched M0/M1/M2 opportunities, followed by the previously authorized coordinates. That ongoing search is not complete and its result is not implied by this report.

The complete anonymous signed scalar evidence is under `results/stvg_temporal_role_alignment/2026-10-15/`. Exact new implementation and the fixed protocol are published together. Raw captions, queries, RGB, boxes, annotations, model states and weights remain private. Existing shared helpers with historical private-path aliases retain their established public versions; this export does not overwrite unrelated research files.
