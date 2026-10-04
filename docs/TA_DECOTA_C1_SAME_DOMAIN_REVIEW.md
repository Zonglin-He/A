# C1/Scale06 online DeCoTA in the within-domain corruption setting

This evaluation runs the user-confirmed later **C1/Scale06 online** version. It compares the same official within-domain TA-STVG EMA checkpoint with Frozen, without parameter search or method promotion. Current outputs follow the original after-current-query correction timing; only spatial LN persists.

| Dataset / panel | Frozen vIoU % | DeCoTA vIoU % | Paired change pp [95% CI] | >5 pp harm / cells |
|---|---:|---:|---:|---:|
| Vid search | 16.2145 | 19.2283 | 3.0137 [-1.6502, 8.5130] | 65 / 320 |
| Vid confirm | 32.9534 | 35.0244 | 2.0710 [-4.5533, 8.2613] | 28 / 160 |
| HC2 search | 29.8667 | 31.1982 | 1.3315 [-2.2094, 4.3710] | 39 / 320 |
| HC2 confirm | 26.4778 | 30.0298 | 3.5521 [0.7188, 6.9552] | 8 / 160 |

All primary rows are corruption-only. The aggregation unit is source: average both orders and five conditions within a source, then average sources; paired source bootstrap has 10,000 draws, seed 20261004. Each dataset has 32 development and 16 disjoint confirmation sources. All have historical exposure; confirmation is not fresh.

## Exact method and comparison boundary

- Spatial: query256 + LN1536, unchanged locked Scale06 Adam .03, ten updates, minimum original reference loss over steps0..10, query and Adam reset every query, selected LN displacement writes back at 1/16.
- Temporal: original full66306 head, NLL + hinge, fresh AdamW five steps and eta .25 shrink, then discarded. Vid lr .1 / center .5; HC lr .001 / center1. No new persistent temporal head is introduced.
- DINO-tiny original context/target-token reference acquisition uses up to four native-interval frames on every query. This exceeds recent A’s 25% expert-arrival budget and uses a different expert. No A-versus-DeCoTA budget-matched claim is made.
- Official same-domain VidSTG / HC-STVG-v2 checkpoints, original Paper48 pixels/sampling/two offsets. Existing Frozen H is reused with exact native and original C1 full-forward/trajectory/reinsertion checks. Scale06 was historically locked on Vid; the fixed same spatial rule is newly evaluated on HC2.
- State chains reset independently at dataset/split/condition/order boundaries. Sixteen/thirty-two-query streams are a small-panel online evaluation, not a completed full-dataset benchmark.

## Current correction versus inherited state

| Confirmation corruption | Inherited LN only pp | Spatial after pp | Temporal only pp | Full tIoU change pp | Dense sIoU change pp |
|---|---:|---:|---:|---:|---:|
| Vid | 0.5035 [0.0719, 1.0005] | 4.3812 [0.9166, 9.4812] | -1.9736 [-7.5851, 2.1832] | -2.2910 [-9.3592, 3.1487] | 6.7117 [2.3119, 12.3146] |
| HC2 | 1.3038 [0.8129, 1.8947] | 3.7314 [0.9184, 7.0461] | -0.1475 [-0.6378, 0.3070] | -0.5853 [-1.7893, 0.4723] | 4.7790 [-0.7551, 9.5868] |

These are matched counterfactual readouts relative to Frozen. Their vIoU changes are not additive branch contributions because temporal support changes the scored spatial frames. Inherited-LN-only is generated before this query’s update; full DeCoTA includes current spatial and episodic temporal correction.

## Clean and negative tails

| Confirmation | Clean vIoU change pp | Corruption >20 pp harms | vIoU .3 correct→wrong / wrong→correct | vIoU .5 correct→wrong / wrong→correct |
|---|---:|---:|---:|---:|
| Vid | 0.0903 [-8.3236, 6.2876] | 8 | 8 / 2 | 10 / 14 |
| HC2 | 5.6070 [2.4008, 9.3068] | 0 | 0 / 6 | 0 / 35 |

Order-specific and every-condition tables, source values, gross gain/loss and all 1,152 anonymous metric rows are saved in SUMMARY.json and ROWS.json. UPDATE_DIAGNOSTICS.json keeps each reference-loss path, chosen step, temporal losses and state magnitude. Negative cases are retained; no thresholds or configurations are selected after looking at these results.

## Cost and checks

| Dataset | New DINO forwards | Logical observation requests | Empty reference arrivals | Spatial backward calls | Evidence wall min | Online wall min |
|---|---:|---:|---:|---:|---:|---:|
| Vid | 888 | 1776 | 146 | 4300 | 3.49 | 5.28 |
| HC2 | 1152 | 2304 | 12 | 5640 | 5.02 | 5.65 |

Temporal adaptation is evaluated once per identical query-condition and reused across its two orders; it has no persistent state. Evidence wall time includes expert loading. Online wall time starts after STVG loading and includes the complete prediction loop, host I/O and logging; it is not pure GPU kernel time. Two extra full-backbone smoke checks verify the frozen-cache interface; formal streams reuse encoder caches.

All prediction workers deny GT and scored outcomes. Their global barrier precedes CPU scoring. The root inspected the existing label-container/key schema before launch; those historical labels did not determine the configuration, cohort, order or selection. Independent CPU checks verify the original spatial objective, FP32 Adam displacement, query reset, exact 1/16 LN recurrence, temporal NLL/hinge and last/shrunken states, and official/vectorized dense scorer agreement. Source weights and production registries are unchanged.

A cache-device argument omission failed the first smoke invocation before predictions. Its original worker/log were preserved under recovery/cache_device_argument_001; an explicit CUDA-device argument fixed the adapter plumbing, with no scientific configuration change. The complete paired smoke checks were then regenerated and passed.

The scope of any positive finding is this fixed historically exposed small panel, checkpoint and after-update readout. No fresh-test, general DeCoTA, long-stream retention or automatic production-promotion claim follows.

Artifacts: [protocol](../protocols/tastvg_decota_c1_same_domain_v1.md), [all anonymous results](../results/tastvg_decota_c1_same_domain/2026-10-04/), [root audit](../results/tastvg_decota_c1_same_domain/2026-10-04/ROOT_AUDIT.json).
