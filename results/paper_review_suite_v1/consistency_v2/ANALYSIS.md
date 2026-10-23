# Matched entropy and photometric-consistency TTA baselines

All384 predictions sealed before independent CPU scoring. 768 official/dense metric checks; maximum discrepancy1.2212453270876722e-15. Native clean/augmented replay and post-reset readouts were exact; final readback verifies all384 prediction/trace/photo-context hashes and exact original Source boxes/intervals/metrics.

The fixed adapters update only the original temporal decoder LayerNorm affine9728 scalars with five Adam.001 steps per arm/query. Spatial boxes, cached first-pass gating/context and native MBR readout are fixed. PhotoConsistency minimizes endpoint JS against an actual RGBfloat32×.8→uint8 Source view. This is a restricted temporal LN baseline, not canonical BN-Tent or CoTTA. Fig1 is unchanged.

|Fixed confirmation|Arm|ΔtIoU pp vs Source|ΔvIoU pp vs Source|v95%CI pp|Severe t / v / union once|
|---|---|---:|---:|---|---|
|hc2128|Entropy|-0.134552|-0.217783|[-0.7238483039902049, 0.2785501325838481]|10 / 5 / 10|
|hc2128|PhotoConsistency|-0.046107|-0.034903|[-0.319805368888442, 0.20993796657461256]|4 / 2 / 4|
|hc2128|Full|+1.328928|+2.645083|see original full / matched panels|11 / 18 / 25|
|vidstg128|Entropy|-1.281317|-0.448877|[-1.0431995416739774, -0.0065820515482891894]|12 / 4 / 12|
|vidstg128|PhotoConsistency|+0.004427|+0.037199|[-0.10614026515333107, 0.17819771317126823]|3 / 0 / 3|
|vidstg128|Full|+0.383248|+1.103289|see original full / matched panels|13 / 18 / 28|

Both objectives decreased on all384 episodes. PhotoConsistency shows no established confirmation benefit (both v CIs cross zero); Entropy lowers Vid confirmation t/v with negative paired CIs. These outcomes concern the fixed temporal-only update scope and do not establish universal failure of all consistency/entropy TTA. Full also uses expert spatial evidence; comparison with these restricted baselines is a system comparison, not equal-information optimization causality.

Actual new work:384 brightness-view Source forwards,3840 optimizer steps,16896 additional TimeDecoder calls,0expert. Predictor wall1982.275953s. The384 clean Source acquisitions belong to the earlier shared-support study and are reused here; neither this time nor cache reuse is a deployment acceleration claim. Model startup and inherited clean acquisition are separate cost provenance.

Screen64 and confirmation128 parents/domain are disjoint, but the full official datasets were historically exposed. They are fixed new contrasts, not historically unseen data. Every screen/confirmation signed row, objective change, severe union and short/long group is retained. No parameter selection followed these results.

The original horizontal-flip plan was cancelled before any Source/TTA/prediction/label operation because25captions include directional words. Its protocol and cancellation remain preserved. The independently fingerprinted brightness v2 changed no step count, optimizer, update scope, selection, metric or readout.
