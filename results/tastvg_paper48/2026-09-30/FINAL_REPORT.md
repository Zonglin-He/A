# Paper48: completed P0–P5 evidence

All authorized P0–P5 model executions and scoring have completed. The final HC-STVG-v2 panel is included; the removed 48-hour deadline was not used to skip it. The frozen J0.1 recipe was not tuned using these outcomes. This report accompanies the final public code/results export; the local publication receipt records the subsequent remote verification.

## Main readout

Sparse temporal specialist reranking plus spatial parameter adaptation improves mean performance in the larger VidSTG panel. Transfer to future nonexpert arrivals is positive there, while corruption-specific advantage over clean refinement is not established. The smaller HC2 panel has positive point estimates with confidence intervals spanning zero. Availability sweeps do not establish monotonic benefit.

All deltas below are percentage points (pp) of dense mean vIoU, Ours minus Frozen. Intervals are 10,000-replicate source-bootstrap 95% intervals, retaining repeated conditions/orders within sources. Subsets of P1 are not independent confirmations.

| Phase / population | Sources | Arrivals | Corrupt all ΔvIoU [95% CI] | Corrupt future nonexpert ΔvIoU [95% CI] |
|---|---:|---:|---:|---:|
| P1 VidSTG, two orders, clean + five 5% conditions | 670 | 8,040 | +0.4712 [+0.1905, +0.7756] | +0.3787 [+0.3100, +0.4523] |
| P2 nested VidSTG, two orders, clean + five families × 1/5/10% | 128 | 4,096 | +0.4706 [+0.0728, +0.8993] | +0.1380 [+0.0472, +0.2170] |
| P3 nested VidSTG, 0% specialists | 64 | 768 | 0.0000 (exact Frozen parity) | 0.0000 |
| P3 nested VidSTG, 25% specialists | 64 | 768 | −0.0477 [−0.5050, +0.3118] | +0.0356 [−0.0006, +0.0746] |
| P3 nested VidSTG, 100% specialists | 64 | 768 | +0.2113 [−1.7555, +2.0614] | Not applicable: every arrival is expert-assisted |
| P5 HC-STVG-v2, one order, clean + five 5% conditions | 128 | 768 | +0.4939 [−0.1964, +1.3960] | +0.4095 [−0.2424, +1.4758] |

P1 future nonexpert comprises 626 unique sources and 5,020 corrupt cells across the two orders; P2 comprises 121 sources / 2,880 cells; P3 at 25% comprises 61 sources / 480 cells; P5 comprises 96 sources / 480 cells. Different expert positions across orders cause some sources to contribute only in one order to the nonexpert subset. The prescribed source aggregation is used throughout.

P1 clean all ΔvIoU is +0.4108 pp, and paired corruption-minus-clean gain is +0.0604 [−0.0454, +0.1665] pp. P5 clean all ΔvIoU is +0.4214 [−0.2166, +1.2689] pp; its paired excess is +0.0724 [−0.1302, +0.2884] pp. Neither interval establishes a corruption-specific recovery advantage. Separate clean streams are not tests of clean retention after a corrupted stream.

## P5: HC2 binding and results

The cohort contains 128 original sources, one hash-chosen query per source, one locked source order and six conditions, selected from the local official HC-STVG-v2 validation confirm512 pool (167 sources). It is not a full 3,482-query validation benchmark or globally fresh data. Prior project exposure and annotation-bearing metadata intake are disclosed. The intake used whitelisted query/media metadata for selection and model input, with no GT-based selection. Dense labels were read for scoring only after all 768 predictions were sealed.

The official same-domain TA-STVG checkpoint SHA256 is `47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036`. The existing project runtime uses 224px and nominal 64 frames, not the official 420px benchmark setting. The 1,792 updated parameters, plain SGD learning rate .005, one step, probe seed and relative radius .05, specialist schedule and output-before-spatial-update rule remain frozen. Probe support is centered on the HC2 checkpoint; no Vid-trained adapted state is transferred.

| Input | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |
|---|---|---:|---:|---:|---:|
| Clean | Frozen | 58.3682 | 29.9901 | 42.1875 | 13.2812 |
| Clean | Ours | 58.3890 | 30.4115 | 42.9688 | 14.8438 |
| Corruption | Frozen | 57.1348 | 29.1984 | 42.1875 | 12.1875 |
| Corruption | Ours | 57.1732 | 29.6923 | 43.9062 | 14.2188 |

For corrupt P5 arrivals, mean dense spatial IoU changes by +0.3119 [−0.3503, +1.2499] pp overall and +0.4276 [−0.4523, +1.7286] pp on future nonexperts. Nonexpert temporal IoU is unchanged. Overall corrupt vIoU losses greater than 5 pp occur for 4/128 source averages and 30/640 cells; future nonexpert losses occur for 1/96 source averages and 11/480 cells. No P5 corrupt source average or cell has a vIoU loss greater than 20 pp. Positive means do not remove these negative cases. One order provides no between-order SD estimate.

The final audit verifies 768 state links, 168 SGD updates, 192 temporal critic selections and 1,536 independent dense metric comparisons with maximum discrepancy zero. Twelve complete native spatial reinsertions pass. Root readback verifies all 768 sealed receipt/payload hashes, scalar identities and seal-before-GT chronology. Anonymous scalar reaggregation passes 14,851 checks including source bootstrap, clean/corrupt pairing, quartiles and negative tails. Spatial/temporal specialist stages each have 192 receipts; spatial has 42 fresh inference calls because identical sampled inputs reuse matching cache, while temporal has 192 fresh calls. These are research-cache counts, not deployment latency.

Two P5 engineering attempts remain excluded and preserved. First, raw `vsync=0` decoding could not provide an officially sampled end frame; the corrected P5 decoder follows the official HC full RGB output timing then selects the unchanged indices. Second, an off-grid frame-freeze donor still used the former decoder in the online process; revision009 binds that path identically to the specialists. The 12 affected freeze-prefix predictions and scratch are retained outside the valid run, and that stream was restarted from its initial state. All 128 sources were checked: both specialists' 384 input receipts and the retained 256 clean/drop predictions have matching pixel hashes. No manual frame padding, index replacement, source substitution, GT-based repair or method tuning was used. Online allocations total 1,178.122 seconds including the failed attempt; this is not an uncached latency measurement.

## P0 ablation and P4 efficiency

P0 adds only Raw-RKL on 16 exposed development sources, five locked orders, six conditions: 480 new arrivals and 102 updates. Its future ΔvIoU is +0.000180 ± 0.000042 pp versus frozen Final's +0.025505 ± 0.020209 pp on that historical sampled-grid readout. This is not a norm-matched comparison and is not merged with the official dense main table. It does not establish universal failure of raw rewards.

P4 measures 100 fixed clean queries in a persistent stream, two arms / 200 uncached native executions and 50 fresh specialist executions. Each native/specialist call starts a new process and loads/verifies its model; costs include decoding, loading, switching, computation and serialization on one RTX 5090 32GB. These are cold-start end-to-end measurements, not warm-service latency.

| P4 arm / arrivals | Mean seconds | Median seconds | p95 seconds |
|---|---:|---:|---:|
| Frozen, 100 | 8.3874 | 8.3564 | 8.8556 |
| Ours, all 100 | 18.4230 | 9.3278 | 45.5977 |
| Ours, expert 25 | 46.2577 | 44.4986 | 55.5899 |
| Ours, nonexpert 75 | 9.1448 | 9.1003 | 9.7371 |

Peak PyTorch allocated memory is 7.8306 GiB for Frozen and 9.3943 GiB for Ours; it is not total device or reserved memory. All 100 Frozen predictions exactly match corresponding P1 native boxes and interval indices, and all 100 persistent state links and 50 fresh expert receipts pass. The public timing audit has 1,160 checks. Archived loader/precision attempts are not mixed into the valid timing table.

## Scope and files

The valid execution totals are 480 P0 + 8,040 P1 + 4,096 P2 + 2,304 P3 + 768 P5 = 15,688 online arrivals, plus 200 P4 native arm-query executions. This count excludes failed/restarted prefixes and is not a count of unique sources.

All component tables, anonymous scalars, audits and configuration revisions are in `results/tastvg_paper48/2026-09-30/`. `ALL_PHASES_ROOT_REVIEW.json` records the complete review, while prior completion/publication records remain historical snapshots. This final report supersedes their pending-P5 descriptions. The earlier 451,728-arrival B1 and paper-matrix queues remain stopped, without partial GT scoring. External baselines, Pairwise, mixed/dynamic streams, cross-domain experiments and TubeDETR were not run as part of this scope. The production method registry is not changed.
