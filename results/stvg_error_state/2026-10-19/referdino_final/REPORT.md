# Final ReferDINO OPD: temperature calibration, full inference and research freeze

Frozen temperature **tau=0.20**, scales(.05,.25), Adam.01/40steps,32 candidates, native8 positions,2048 per-query temporary states; pretrained weights frozen. Seeds41/42/43 all retained. HC primary. No more algorithm search.

## HC128 exposed development temperature selection

Means are per-parent means of three seed metrics, then128-parent mean. The same predeclared full-vIoU ranking selects temperature; ESS is diagnostic. RH32 is already exposed and is not used for this choice.

| tau | full vIoU % | outside8 actual sIoU % | mean ESS |
|---|---:|---:|---:|
| 0.05 | 27.888703 | 51.818915 | 1.531808 |
| 0.10 | 27.891990 | 51.904038 | 2.371545 |
| 0.20 | 27.928474 | 51.835018 | 4.655576 |

All individual new128×2×3 outputs and losses/ESS/state updates are in [temperature results](TEMPERATURE_RESULTS_PUBLIC.json) and [step diagnostics](TEMPERATURE_STEP_DIAGNOSTICS_PUBLIC.json). Originaltau.05 controls/results are reused exactly; new teacher/Source calls during temperature screening are zero.

## Frozen RF32 full-chain evaluation

RF32 parents were locked before new temperature predictions/GT, disjoint from RC128, prior RH32 and NH16. They were not used to choose this temperature or interface. Earlier full Source/old-method exposure remains: this is not globally untouched test data. All32 including negative, no-update and zero-temporal-intersection cases remain.

| output | full vIoU % | outside8 actual sIoU % |
|---|---:|---:|
| Source_MBR | 25.057508 | 53.568503 |
| E_MBR | 28.884546 | 58.542511 |
| Direct | 28.289246 | 58.030091 |
| Fixed_seedmean | 28.439610 | 58.069079 |
| Live_seedmean | 28.017371 | 57.633211 |

| comparison, pp | full vIoU [95% paired CI] | outside8 actual sIoU [95% paired CI] |
|---|---:|---:|
| Live_seedmean_minus_Source_MBR | +2.959863 [-0.473334, +6.600412] | +4.064708 [-0.841650, +8.747838] |
| Live_seedmean_minus_E_MBR | -0.867175 [-3.259728, +0.779000] | -0.909300 [-3.599010, +1.105581] |
| Live_seedmean_minus_Direct | -0.271876 [-0.886157, +0.087963] | -0.396880 [-1.184609, +0.062743] |
| Live_seedmean_minus_Fixed_seedmean | -0.422239 [-1.504652, +0.493084] | -0.435868 [-1.600138, +0.796901] |
| Fixed_seedmean_minus_E_MBR | -0.444936 [-1.812039, +0.362022] | -0.473433 [-2.133625, +0.516184] |
| Direct_minus_E_MBR | -0.595299 [-2.882257, +0.876191] | -0.512420 [-3.166151, +1.385045] |

Total spatial improvement is Live−Source using the same MBR. Incremental adaptation is Live−E. Independent OPD value additionally requires Live−Direct and Live−freshFixed; these quantities are reported separately and cannot be interchanged. Paired intervals resample parents, not seeds. Full signed per-query/seed data and severe tails are retained in [formal results](FORMAL_RESULTS_PUBLIC.json), with all three per-seed aggregate comparisons in [seed breakdown](SEED_BREAKDOWN_PUBLIC.json). No formal-result parameter switching.

## Cost and execution chain

Measured GPU allocation for this final task: 2707.874s. Formal raw video+text→Source→GroundingDINO4/E/MBR→nativeRefer8→Live(seed41) mean **16.564s**, median15.158s, range[14.469,24.951]s. This includes actual decoding/parsing, expert execution, and their loading/process overhead in the implementation tested. Source loads once per8-query worker; other expert loads are per query. The all-controls time is separately recorded, not mislabeled single-inference latency. Per-query wall begins inside an initialized Python worker; its import/initial CUDA context setup remains in the separate allocation total. Peak memory is torch maximum allocated memory, not global NVML device usage.

Actual formal new calls:32Source queries/64offset forwards, 124GroundingDINO image forwards,32ReferDINO video forwards/256observed images. No new expert type, weights, or training. The E stage keeps the original at-most-four planned-position budget and admission/skip rules; zero-call and zero-accepted-reference queries remain in the full denominator. The two expert dependencies are explicit; historical E evidence is not treated as free. [Full stage timing, peak memory and per-query cost](COST_BREAKDOWN.json).

## Implementation and authority

[Unique research manifest](../../../../methods/RESEARCH_METHOD.json), [single config](../../../../methods/stvg_referdino_opd_v1/config.json), [runnable video/text interface](../../../../methods/stvg_referdino_opd_v1/README.md). Predictor has no experimental roster, scoring or report imports. Same-cache zero E, first update and final trajectory match the legacy implementation exactly, includingtau.05 replication. Every fit resets content states and Adam; no LN carry. Fresh Fixed uses initialmean+D0 and new noise each round, while current student receives gradients. Live refreshes both mean and size context.

This freezes implementation and configuration, not a positive efficacy conclusion. The historical P7-A deployment remains explicitly separate; MAIN/PAPER/WORKING point to the unique new research manifest. Old protocols, predictions and negative results are not moved or rewritten.

## Decision

Final scientific conclusions must follow the above intervals and negative tails. A development mean improvement or softer ESS alone is insufficient to establish a repeatable2–3pp OPD gain. No further temperature, scale, step, learning-rate, expert or interface search is authorized by this frozen version.

RF32 measured decision: Live−E full vIoU -0.867175pp, outside8actual sIoU -0.909300pp. Positive paired lower bounds on both E comparisons: False. Positive paired lower bounds against E, Direct and fresh Fixed on both endpoints: False. A >=2pp incremental Live−E mean with positive full-vIoU lower bound: False. These are this cohort's evidence checks, not universal impossibility statements. Temperature.20 remains frozen under the prior selection rule regardless of formal outcome.

The frozen recipe did not reproduce its development gain on RF32: all three per-seed Live−E full-vIoU means were negative (41: −0.941192pp,42: −0.947014pp,43: −0.713319pp). The paired primary comparisons with E, Direct and fresh Fixed all cross zero. Two severe losses remain in the32-parent primary result: RF028−31.453753pp and RF006−8.072509pp. On RF028, E full-vIoU40.211742% falls to Live seed-mean8.757989%; the pseudo-label cost improves in all three Live fits. This records a loss/task mismatch without identifying teacher correctness or propagation as the unique cause. [Saved severe-tail scalar evidence](SEVERE_TAIL_DETAILS_PUBLIC.json).

Live−Source total spatial +2.959863pp includes E−Source +3.827038pp; it does not establish a2–3pp additional OPD gain. Both total-spatial and incremental confidence intervals remain reported above. The configuration and implementation remain frozen; no further parameter search or severe-case removal follows this result.
