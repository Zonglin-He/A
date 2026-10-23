# Cycle-reliability screening and weighting: fixed cached spatial ablation

## Question and intervention

Retain the exact bidirectional affinity/complete-box affine geometry and remove only return-cycle reliability screening and weighting. Geometrically available directions receive reliability1, unavailable directions0. Keep affine mass/ESS/variance/scale/finite-box validity checks. Rebuild side-averaged support, candidate projection, node weights and adjacent pair weights. Recompute adjacent link Dice from the changed merged support with actual cached Source transport; the old link is not reused.

Both arms use the frozen kappa64 Full interval and identical Native2D features/posterior, E, candidates, expert8 observations, original MBR, score coefficients and DP. This is a spatial ablation, not an end-to-end feedback-ROI change. No gate threshold, scoring weight, observation or teacher tuning. Production is unchanged.

## Available evidence and scope

67 queries/67 parents match official HC inputs exactly and have complete archived Source spatial tensors/masks. Membership is determined solely by input/cache availability before GT access. All geometry tensor fields and original Full boxes reconstruct bitwise for all67. The broader compact full/panel caches lack source adjacency needed to rebuild link after removing side weights; no Source recapture was authorized or performed. This exposed availability subset is not the wholeHC1902 or a fresh confirmation. No Vid result is claimed.

All67 predictions sealed before a separate CPU official-label score. Spatial evaluation uses actual sampled GT-supported positions with the same observed/unobserved partition, within-query average then equal-query mean; no arm-dependent temporal support selection. Unobserved valid67, observed valid66.

## Metrics

| Arm | Fixed unobserved sIoU% | Complete vIoU% | Observed sIoU% | tIoU% |
|---|---:|---:|---:|---:|
| Source | 50.047065 | 27.535844 | 52.196678 | 54.180620 |
| Current | 55.087238 | 31.211470 | 61.051451 | 54.460918 |
| NoCycle | 54.868079 | 31.060033 | 61.042464 | 54.460918 |

| NoCycle-Current | Mean pp | 95% parent-cluster CI | Improved / harmed / tied |
|---|---:|---|---|
| unobserved_actual_sIoU | -0.219159 | [-0.901584, +0.408188] | 22 / 19 / 26 |
| v | -0.151437 | [-0.767830, +0.361950] | 21 / 22 / 24 |
| observed_actual_sIoU | -0.008988 | [-0.146827, +0.119173] | 2 / 2 / 62 |
| t | +0.000000 | [+0.000000, +0.000000] | 0 / 0 / 67 |

Time is exactly identical between the two spatial arms for every query. No temporal improvement can explain their complete-v difference. Both primary mean CIs span0; these data do not establish that cycle reliability is indispensable. NoCycle R@.3 rises49.2537%->53.7313% while R@.5 stays17.9104%; these threshold counts do not overturn the paired mean uncertainty.

## Severe damage: explicit baselines and both signs

Severe means complete vIoU OR fixed unobserved sIoU declines>5pp. This is a spatial/v union, not the t/v union used in the separate temporal study. Count a query once; metric counts are not additive.

| Comparison | Severe complete v | Severe unobserved s | Union |
|---|---:|---:|---:|
| NoCycle-Current | 1 | 4 | 4/67 (5.9701%) |
| NoCycle-Source | 11 | 11 | 13/67 (19.4030%) |
| Current-Source | 11 | 12 | 14/67 (20.8955%) |
| Current-NoCycle (reverse sign) | 1 | 1 | 1/67 |

NoCycle creates4 severe losses relativeCurrent and1 severe improvement. RelativeSource, the severe union is Current14 versusNoCycle13, and severe complete-v is11 for both. Therefore it would be wrong to claim that cycle reliability uniformly reduces Source-relative damage; that aggregate tail count slightly favors removal in this subset.

## Duration groups

| Group | N | NoCycle-Current unobserved s pp | NoCycle-Current complete v pp | Severe union vsCurrent |
|---|---:|---:|---:|---:|
| short_le5s | 19 | -0.628084 | -0.375592 | 2 |
| long_gt5s | 48 | -0.057293 | -0.062709 | 2 |

## Concrete changes

Removing cycle checks re-enables551 full-timeline nodes and854 candidate slots before event clipping. Final DP changes235 node actions,231 unobserved. These are not all GT-scored positions.

| Query alias | Delta unobserved s pp | Delta complete v pp | Changed nodes | Selected previously cycle-ineligible slots |
|---|---:|---:|---:|---:|
| H033 | -5.342817 | -0.941286 | 2 | 0 |
| H049 | -7.064784 | -1.355378 | 5 | 5 |
| H053 | -13.944522 | -16.152012 | 8 | 6 |
| H059 | +8.628875 | +7.690997 | 6 | 0 |
| H066 | -6.788804 | -2.614999 | 2 | 0 |

H049 selects5 previously cycle-ineligible slots whose old node confidence is about1.4e-6 on average, with unobserved-s-7.0648pp. H053 selects6 such slots among8 changed nodes and loses16.1520pp complete-v. These cases demonstrate that dropping screening can admit harmful geometry even when affine validity holds. H059 improves7.6910pp complete-v/8.6289pp unobserved-s without opening a previously ineligible selected slot, showing that removing the weights can also help. The joint ablation does not separately estimate the gate and weight effects. It is not an identity-correctness test.

## Decision and proof

Keep the active frozen method unchanged. Small mean losses with CIs spanning0 do not prove necessity, while the new severe losses and the severe rescue both remain visible. No stable overall benefit from removal is established here. This component can remain an implementation detail of reliable propagation rather than an independent claimed contribution.

All67 original geometry tensors, output boxes and DP actions reproduce exactly; candidate boxes/features/side availability remain exact in NoCycle. Reconstruction idempotence and unavailable-side non-admission pass67/67. 201 official-versus-independent dense metric checks max8.88e-16. CPU source-only preflight14.41s, replay9.26s, scoring1.04s;0models/RGB/encoder/expert/GPU/training. These cached research costs do not measure deployment speedup; reverse correspondence still supports the preserved geometry.

[Protocol](protocol.json), [full metrics/CIs/signed rows](report.json), [all signed pair outcomes](signed.tsv), [all severe negatives](severe.tsv), [both-sign case audit](AUDIT.json). Private media/captions/GT coordinates/masks/features/weights remain private.
