# Temporal coverage and boundary-quality review

The 2-support x (oracle + A/B/D) experiment is complete and independently audited. Every Old8 entry and its original A selection is retained in Expanded32. Spatial C is removed. Original experimental A and the deployed CURRENT_METHOD remain unchanged.

## Configuration

VidSTG and HC-STVG-v2 each retain the same 32 development +16 confirmation sources, one query/source, two orders, clean + five fixed 5% transient corruptions, 25% scheduled experts. All 1,152 readouts are covered; 288 expert positions have both supports and six selectors (240 corruption/48 clean). The 864 nonexpert positions are exactly A. All sources have historical exposure; confirmation is source-disjoint within this batch, previously diagnosed, and not fresh.

The reference is **A8**, the saved experimental A with Uniform5 persistent spatial Rank-RKL and original UVTG Fast, not Frozen. Full boxes, 1,792 parameters, Vid K1/HC K8, prior sealed learning rates/temperatures, pixels, expert schedule, sampling and checkpoints do not change. No backbone execution, backward, new expert, new temporal view or learned quality head is used.

Expanded32 preserves the eight original entries in their order and appends 24 deterministic maximin intervals in normalized physical (start,end) coordinates from all existing merged-grid i<j pairs. This is a capacity control, not a trained candidate generator. UVTG A, previous semantic contrast B (outer ratio .25), and new D share each identical support. D uses min(start rise,end fall), w=1 second prelocked, fractional integration, inner clipping to the interval, outer clipping to observed support, and neutral zero for a missing exterior side. Missing sides cannot count as positive transitions. Flat/unavailable curves retain A on the same support; other ties are native first at 1e-12. No width tuning, positive gate, confidence multiplier, C or B+D.

[BAM-DETR](https://arxiv.org/html/2312.00083v2) separates matching from localization quality with an IoU-supervised quality head. This experiment tests an **untrained PE transition proxy**, not that head and not an IoU estimator.

## Capacity on corruption expert arrivals

Values are vIoU percentage points. Oracle changes only offline temporal readout of fixed A boxes.

| Dataset/panel | Expert cells/sources | O8 vIoU % | O32 vIoU % | O32-O8 pp [paired 95% CI] |
|---|---:|---:|---:|---:|
| vidstg/search | 80/16 | 19.9520 | 25.3415 | +5.3895 [+1.7808, +9.8816] |
| hc2/search | 80/14 | 35.7327 | 43.5821 | +7.8494 [+3.3005, +12.7883] |
| vidstg/confirm | 40/8 | 25.2270 | 32.7144 | +7.4875 [+2.0401, +14.1698] |
| hc2/confirm | 40/7 | 37.3005 | 44.0197 | +6.7192 [+0.0000, +15.3967] |

## Actual complete corruption flow relative to A8

This includes scheduled and nonscheduled arrivals; do not scale expert means by .25. Values below are paired delta-vIoU pp with 95% source-bootstrap intervals.

| Dataset/panel | A8 vIoU % | B8-A8 | D8-A8 | A32-A8 | B32-A8 | D32-A8 |
|---|---:|---:|---:|---:|---:|---:|
| vidstg/search | 17.4529 | -0.4085 [-1.0458, +0.0404] | -0.5089 [-0.9860, -0.1342] | -1.3958 [-4.1014, +0.2939] | -1.6654 [-4.4450, +0.2623] | -3.3540 [-6.6746, -0.7154] |
| hc2/search | 30.9637 | -0.2161 [-0.8022, +0.2926] | -0.6981 [-1.4429, -0.0519] | -2.8308 [-6.1462, -0.1310] | -1.5321 [-3.5313, +0.1274] | -3.2151 [-5.9379, -0.9618] |
| vidstg/confirm | 32.6465 | -0.0542 [-0.4990, +0.3754] | -0.3703 [-1.0335, +0.1789] | -2.1184 [-5.7419, +0.2247] | -3.1872 [-7.0864, +0.0734] | -4.0018 [-7.5360, -1.1844] |
| hc2/confirm | 26.7630 | -0.3278 [-0.9475, +0.2046] | -0.7196 [-1.7285, +0.0735] | -1.7977 [-3.7569, -0.3433] | -1.8520 [-4.8826, +0.8004] | -3.4176 [-7.1499, -0.6435] |

## Same-support actual comparison

| Dataset/panel | B32-A32 vIoU pp [CI] | D32-A32 vIoU pp [CI] | D32-A32 tIoU pp [CI] |
|---|---:|---:|---:|
| vidstg/search | -0.2696 [-0.9273, +0.3503] | -1.9582 [-4.0094, -0.5076] | -5.9330 [-9.7396, -2.6584] |
| hc2/search | +1.2988 [-0.5618, +3.4090] | -0.3843 [-2.7865, +1.9299] | -1.9483 [-6.6289, +2.5058] |
| vidstg/confirm | -1.0688 [-3.4092, +0.7336] | -1.8834 [-4.5684, +0.3843] | -4.0888 [-9.9494, +0.8973] |
| hc2/confirm | -0.0543 [-2.1391, +2.0304] | -1.6199 [-4.6130, +0.7633] | -3.1578 [-8.6331, +1.1022] |

## Oracle regret, ordering and harms

All following rows are corruption expert arrivals. Regret is oracle-vIoU minus selected-vIoU on the same support; it is nonnegative. Pairwise ordering compares every strictly unequal GT candidate pair, gives score ties half credit, excludes cells with no strict pairs, and uses source aggregation. vIoU ordering and tIoU ordering are distinct diagnostics. Severe harm is an arrival delta below -5 pp relative to A8; destroyed Fast gains count old A8>native followed by a lower readout.

| Dataset/panel | Arm | vIoU % | tIoU % | regret pp | pair-v % | pair-t % | better/harm/same vs A8 | harm>5pp | old-positive Fast destroyed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vidstg/search | A8 | 17.8323 | 42.3285 | 2.1197 | 76.45 | 77.80 | 0/0/80 | 0 | 0/35 |
| vidstg/search | B8 | 16.1984 | 35.2900 | 3.7536 | 55.75 | 54.75 | 11/32/37 | 9 | 23/35 |
| vidstg/search | D8 | 15.7968 | 35.1226 | 4.1552 | 55.65 | 56.14 | 8/30/42 | 11 | 23/35 |
| vidstg/search | A32 | 12.2491 | 37.1956 | 13.0925 | 76.50 | 76.94 | 29/24/27 | 16 | 9/35 |
| vidstg/search | B32 | 11.1707 | 28.6067 | 14.1708 | 52.95 | 54.43 | 26/43/11 | 20 | 23/35 |
| vidstg/search | D32 | 4.4163 | 13.4638 | 20.9252 | 50.07 | 51.28 | 7/53/20 | 36 | 30/35 |
| hc2/search | A8 | 30.1973 | 53.7567 | 5.5354 | 62.34 | 63.00 | 0/0/80 | 0 | 0/37 |
| hc2/search | B8 | 29.0654 | 50.1258 | 6.6673 | 52.60 | 52.79 | 19/29/32 | 9 | 17/37 |
| hc2/search | D8 | 27.2492 | 48.6052 | 8.4835 | 47.10 | 47.12 | 23/42/15 | 13 | 34/37 |
| hc2/search | A32 | 18.7459 | 38.1845 | 24.8362 | 75.86 | 76.24 | 14/41/25 | 38 | 19/37 |
| hc2/search | B32 | 23.0498 | 40.7992 | 20.5323 | 59.15 | 58.72 | 25/41/14 | 27 | 20/37 |
| hc2/search | D32 | 15.7429 | 27.0602 | 27.8392 | 53.97 | 53.56 | 23/51/6 | 39 | 31/37 |
| vidstg/confirm | A8 | 22.0480 | 46.0983 | 3.1790 | 53.93 | 56.34 | 0/0/40 | 0 | 0/11 |
| vidstg/confirm | B8 | 21.8314 | 46.4966 | 3.3956 | 57.90 | 60.89 | 12/13/15 | 1 | 2/11 |
| vidstg/confirm | D8 | 20.5667 | 42.6696 | 4.6603 | 51.80 | 53.06 | 14/21/5 | 7 | 10/11 |
| vidstg/confirm | A32 | 13.5743 | 30.1775 | 19.1402 | 74.39 | 75.37 | 5/24/11 | 11 | 7/11 |
| vidstg/confirm | B32 | 9.2990 | 21.2048 | 23.4154 | 51.77 | 51.66 | 6/34/0 | 21 | 6/11 |
| vidstg/confirm | D32 | 6.0409 | 13.8222 | 26.6736 | 49.43 | 48.96 | 1/39/0 | 26 | 10/11 |
| hc2/confirm | A8 | 33.7574 | 68.8674 | 3.5431 | 70.50 | 70.75 | 0/0/40 | 0 | 0/24 |
| hc2/confirm | B8 | 32.3744 | 64.2764 | 4.9260 | 69.06 | 69.51 | 12/13/15 | 8 | 12/24 |
| hc2/confirm | D8 | 30.6501 | 62.2735 | 6.6504 | 52.87 | 52.91 | 13/23/4 | 12 | 21/24 |
| hc2/confirm | A32 | 25.5395 | 47.8601 | 18.4802 | 78.52 | 78.66 | 0/20/20 | 17 | 10/24 |
| hc2/confirm | B32 | 24.0264 | 42.2673 | 19.9933 | 56.87 | 57.63 | 17/20/3 | 18 | 11/24 |
| hc2/confirm | D32 | 18.5280 | 33.9557 | 25.4917 | 55.20 | 55.20 | 9/31/0 | 24 | 22/24 |

A lower regret and a higher selected vIoU on identical support are the same paired difference; they are not two independent pieces of evidence. Gross gains/losses, .3/.5 correctness, paired regret CI, both order values, leave-one-source-out ranges, all clean and nonexpert rows are in the complete anonymous ROWS/SUMMARY files.

## Boundary support and clean control

| Dataset/panel | D32 missing exterior candidates / mean32 | D32 chosen both-positive fraction | Clean D32-A8 pp [CI] | Clean D32-A32 pp [CI] |
|---|---:|---:|---:|---:|
| vidstg/search | 8.488 / 32 | 100.00% | -3.4140 [-6.8205, -0.6268] | -1.9236 [-4.0916, -0.2784] |
| hc2/search | 8.686 / 32 | 100.00% | -2.4974 [-4.6725, -0.6359] | -0.2635 [-3.2657, +2.8247] |
| vidstg/confirm | 7.925 / 32 | 100.00% | -4.6284 [-9.0366, -1.3213] | -2.0616 [-4.7185, +0.1924] |
| hc2/confirm | 8.971 / 32 | 100.00% | -2.6929 [-6.3706, +0.0427] | -0.2636 [-3.7666, +2.9316] |

## Positive and negative arrival cases

These examples diagnose this fixed rule; they do not set thresholds or candidate allocation. Source IDs are anonymous within the current cohort. Full curves, all scores, normalized candidate intervals and candidate GT metric values (without annotations) are available in anonymous ROWS.

- vidstg, largest D32 harm vs A32: source 40, exposure_5, order2, arrival 8; A32 v=41.1210%, D32 v=11.3521%, paired -29.7689 pp; D transitions +0.008782/+0.005174, chosen indices 2→21.
- vidstg, largest D32 gain vs A32: source 36, frame_drop_5, order1, arrival 8; A32 v=5.8550%, D32 v=22.4054%, paired +16.5504 pp; D transitions +0.014812/+0.024331, chosen indices 12→20.
- hc2, largest D32 harm vs A32: source 33, frame_freeze_5, order1, arrival 0; A32 v=43.4904%, D32 v=1.7231%, paired -41.7673 pp; D transitions +0.026402/+0.016028, chosen indices 21→19.
- hc2, largest D32 gain vs A32: source 47, occlusion_5, order1, arrival 4; A32 v=0.0000%, D32 v=27.7266%, paired +27.7266 pp; D transitions +0.031345/+0.034770, chosen indices 27→1.

## Decision and limits

- Expanded-support capacity: `not_positive_CI_both_confirmation`.
- D32 vs A32 on both confirmation corruption flows: `not_positive_mean_both_confirmation`.
- Retain original A. Neither Expanded32 nor D is integrated or promoted; no new experiment is started.
- Capacity is restricted to this one deterministic 24-interval addition and these fixed A boxes. A negative proxy cannot reject all unlabelled boundary estimation. B concerns this frozen PE global semantic signal, not all event semantics. D still uses that same signal, so it is not independent boundary evidence. One-second windows and missing exterior context are disclosed limitations.
- Expert membership varies by order; inference is clustered by independent source, not by cells. Small expert source counts and earlier GT exposure limit generalization. No fresh-test, new persistent-transfer or universal temporal-bottleneck claim follows.

## Verification and resources

The predecessor A8/B8 predictions and scores reproduce exactly. All old support and A-selected intervals are retained. Both support generation and all decisions seal before this experiment’s GT access. Independent root scoring uses official dense metrics; the public auditor separately recomputes allocation, every B/D score/selection, candidate arithmetic and source-bootstrap summaries. All frozen inputs and production-method hashes are verified before and after scoring.

- PREPARATION: 4.246 seconds worker wall time; CUDA initialized=false.
- GENERATION_RESOURCES: 5.403 seconds worker wall time; CUDA initialized=false.
- SCORE_CHECKS: 79.992 seconds worker wall time; CUDA initialized=false.
- FINAL_ROOT_AUDIT: 86.440 seconds worker wall time; CUDA initialized=false.

No GPU-kernel time is claimed from worker wall time. All new model/expert calls and backwards are zero. No private captions/media/GT coordinates/features/parameters are in the public export.

## Post-hoc interpretation of the fixed rule

These are descriptive checks on sealed anonymous rows, not new online gates or width tuning. The reproducible calculation is POSTHOC_DIAGNOSTICS.py; it reads no GT coordinates or raw features.

| Confirmation corruption experts | Capacity-positive sources | D32 interval <2 seconds | D32 median length seconds | Both transitions >1e-12: harmed vs A8 / total |
|---|---:|---:|---:|---:|
| vidstg | 6/8 | 18/40 | 2.2356 | 39/40 |
| hc2 | 2/7 | 15/40 | 6.5696 | 31/40 |

The HC capacity interval is [0,15.3967] pp and its mean gain is concentrated in two of seven expert sources. The measured extra opportunity is real for those sources, but not a demonstrated positive interval across the confirmation panel. Vid capacity has a strictly positive paired interval.

D can reward a short local semantic peak with a rise and a fall while missing much of the event. Both transitions being positive is therefore not sufficient evidence for correct event endpoints. The worst HC confirmation case chooses a narrow new interval with positive transitions .026402/.016028 yet loses 41.7673 vIoU pp against A32. The largest HC gain (+27.7266 pp versus A32) is retained in the report and anonymous rows; it does not remove the negative mean or severe tails.

UVTG pairwise-vIoU ordering increases from 53.93% to 74.39% in Vid and 70.50% to 78.52% in HC when the support changes, while top-1 quality falls. The pair populations differ across supports; more correct pair orderings cannot establish a better top-1 selector. D has inferior pairwise ordering and larger same-support regret than UVTG on Expanded32.

This scope supports retaining A and rejecting the proposed one-second PE transition scorer as a method component. The next unresolved question is unlabelled boundary localization quality, not another mix of spatial plausibility and temporal semantic confidence. No follow-on experiment is launched automatically.

An audit-only engineering recovery is retained: one exact-zero diagnostic transition obtained opposite signs at ~1e-17 under vector versus scalar integration. Revision001 compares the floats independently and checks the strict-sign flag from the saved, verified operands. All candidate scores, selections, oracle values and predictions remain unchanged. There was one such diagnostic sign case, no model or selection repair.
