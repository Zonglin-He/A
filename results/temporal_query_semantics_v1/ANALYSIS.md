# Same-target different-action query temporal experiment

VidSTG32independent videos,64real declarative queries: same annotated actor target track and interaction-object track, same physical-videoSHA/inputsegment/nativeframegrid, different nonstatic action predicates and clearly distinct official GT event intervals. SHA-first onepair/video selected from157eligiblepairs/71videos before model outcome access. Full catalog10303queries;442duplicate identical relation records resolved by official query index+target+event;34ambiguous relation queries excluded during eligibility, never silently assigned. Metadata and GT-based cohort eligibility is offline analysis, not unseen testing/inference or target tuning. HC lacks a reliable cross-query target-track identity field and is not claimed to supply same-person pairs.

Current unchanged E-ROI kappa0/beta1/oneprototype update/jointMAP; no firstauxiliaryDP. This experiment evaluates temporal intervals only, without regenerating spatial paths or mixing spatial gains. SourceNative and matched JointPrior distinguish decoding change from visual update. SourceROI is the expert-free temporal feature alternative. Own inputs/outputs exactly reproduce earlier full-query cached temporal results. Natural captions can differ in attributes as well as action; query-dependent Source boxes/priors/Eregions/eightplannedpositions can differ. Counterfactual feature/prior swapping controls numerical dependence conditional on these upstream states, not full Source language causality.

|Arm|Mean tIoU % [95% video CI]|Correct-vs-swapped GT assignment margin pp [95% CI]|Both events tIoU>=.3 % [95% CI]|Identical pair intervals % [95% CI]|
|---|---|---|---|---|
|SourceNative|+20.523306 [+15.731531, +25.454015]|+0.502879 [-3.262692, +3.897893]|+3.125000 [+0.000000, +9.375000]|+3.125000 [+0.000000, +9.375000]|
|JointPrior|+20.969638 [+16.226105, +25.926945]|+0.055124 [-3.831478, +3.403307]|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|
|SourceROI|+20.384490 [+15.949234, +24.827397]|+0.686568 [-4.038600, +4.104422]|+3.125000 [+0.000000, +9.375000]|+18.750000 [+6.250000, +34.375000]|
|CorrectedROI|+19.898830 [+15.572818, +24.237305]|+0.049714 [-4.577302, +3.290746]|+0.000000 [+0.000000, +0.000000]|+12.500000 [+3.125000, +25.000000]|
|CrossFeatures|+20.085130 [+15.696849, +24.531129]|+0.218217 [-4.450687, +3.618081]|+3.125000 [+0.000000, +9.375000]|+9.375000 [+0.000000, +21.875000]|
|CrossPrior|+19.866913 [+15.040240, +24.904962]|-0.218217 [-3.618081, +4.450687]|+0.000000 [+0.000000, +0.000000]|+9.375000 [+0.000000, +21.875000]|
|CrossBoth|+19.849116 [+15.099362, +24.813444]|-0.049714 [-3.290746, +4.577302]|+0.000000 [+0.000000, +0.000000]|+12.500000 [+3.125000, +25.000000]|
|CrossEvidence|+20.152464 [+15.792888, +24.556317]|+1.108013 [-0.024656, +2.252194]|+0.000000 [+0.000000, +0.000000]|+6.250000 [+0.000000, +15.625000]|
|CommonPrior|+19.139905 [+14.479738, +23.895734]|+0.228587 [-0.276302, +0.908605]|+0.000000 [+0.000000, +0.000000]|+53.125000 [+37.500000, +68.750000]|

Assignment margin = mean tIoU under correct query-event matching minus mean tIoU after swapping the two GT events, in percentage points. Positive distinguishes the two actions better than wrong assignment. Identical pair intervals necessarily give zero assignment margin even if mean tIoU is nonzero. Both-events threshold.3 was fixed before evaluation.

CrossFeatures uses partner E-ROI features with own q0; CrossPrior uses own features with partner q0; CrossBoth is partner features and q0 and reproduces the partner update. CrossEvidence applies partner original evidence to own q0. CommonPrior uses identical symmetric(q0A+q0B)/2 for both queries but their own features. These are mechanism interventions, not candidate methods or new baselines for adoption.

|Comparison|Delta tIoU pp [95% CI]|Improved % [95% CI]|Harmed % [95% CI]|Severe decline>5pp % [95% CI]|Rescue>5pp % [95% CI]|
|---|---|---|---|---|---|
|visual_gain|-1.070808 [-3.291101, +0.888341]|+37.500000 [+26.562500, +48.437500]|+40.625000 [+28.125000, +54.687500]|+14.062500 [+4.687500, +25.000000]|+14.062500 [+6.250000, +23.437500]|
|total_gain|-0.624476 [-3.279950, +1.825788]|+32.812500 [+23.437500, +42.187500]|+42.187500 [+28.125000, +56.250000]|+18.750000 [+9.375000, +29.687500]|+17.187500 [+7.812500, +26.562500]|
|own_feature_gain|-0.186300 [-0.751373, +0.256350]|+17.187500 [+9.375000, +25.000000]|+25.000000 [+14.062500, +35.937500]|+4.687500 [+0.000000, +9.375000]|+1.562500 [+0.000000, +4.687500]|
|own_evidence_gain|-0.253634 [-1.601092, +1.022079]|+28.125000 [+17.187500, +39.062500]|+28.125000 [+18.750000, +39.062500]|+10.937500 [+4.687500, +18.750000]|+10.937500 [+4.687500, +18.750000]|

visual_gain = CorrectedROI minus JointPrior(same interval readout); total_gain = CorrectedROI minus SourceNative(includes readout change); own_feature_gain = CorrectedROI minus CrossFeatures(same own prior); own_evidence_gain = CorrectedROI minus CrossEvidence(same own prior). No total-vs-visual attribution substitution.

|Input/evidence diagnostic|Mean [95% CI]|
|---|---|
|same_eight_positions_percent|+3.125000 [+0.000000, +9.375000]|
|q0_total_variation|+0.599027 [+0.477310, +0.718556]|
|E_features_exact_percent|+0.000000 [+0.000000, +0.000000]|
|E_features_relative_L2|+0.110397 [+0.076403, +0.149872]|
|feature_swap_L_mean_abs|+0.085689 [+0.056774, +0.115977]|
|prior_swap_L_mean_abs|+0.182237 [+0.114425, +0.267939]|
|text_metadata_q_exact_percent|+100.000000 [+100.000000, +100.000000]|
|text_metadata_L_exact_percent|+100.000000 [+100.000000, +100.000000]|
|CorrectedROI_exclusive_action_L_contrast|+0.018156 [-0.049772, +0.085046]|
|CrossFeatures_exclusive_action_L_contrast|+0.051550 [-0.013463, +0.117163]|
|CrossPrior_exclusive_action_L_contrast|-0.051550 [-0.117163, +0.013463]|
|CommonPrior_exclusive_action_L_contrast|-0.015976 [-0.041300, +0.005854]|

Exclusive-event L contrast averages each query evidence on its own exclusive GT-action frames minus the partner exclusive GT-action frames;27/32pairs have sampled native support for both exclusive actions. Missing support is unscorable, never0; all32remain in interval metrics. No independent-frame bootstrap.

## Direct language-input contract: scope correction

Actual frozen one_step signature: (features, q0, *, kappa=64.0, beta=1.0) . The update has no text argument. The sealed arrays labeled text_metadata_* were identical-F/q0 repeat calls: their caption default was unused. Those exactness rates must NOT be described as a new caption-substitution response experiment. They are deterministic replay controls only.

ROOT separately executed64actual calls to the real one_step API using the partner action query as a caption keyword. All64 correctly raised unsupported-keyword TypeError before any numerical/model computation. This confirms the absence of a direct text entry, not a test of whether Source changes under an edited action description. We do not fabricate a text adapter or claim new end-to-end action-text counterfactuals. The zero-feature control exactly reproduces q0.

The32real query pairs already have actual query-conditioned Source/expert outputs in the audited caches. Their natural query differences can enter via Source priors/boxes/Eregions/pooled features, but attributes and eight planned positions can change too. The feature/prior factorial probes test dependence on those upstream states conditional on fixed cached inputs. They do not independently prove action semantics inside Eq6-8.

## ROOT conclusion

On this prespecified32-video same-actor/object different-action cohort, matched JointPrior tIoU20.969638% becomes19.898830% after the unchanged E-ROI visual update; difference−1.070808pp,95%videoCI[−3.291101,+.888341]. Correct-query versus swapped-GT assignment margin+.049714ppCI[−4.577302,+3.290746], with17positive/11negative/4ties; no resolved action-specific matching advantage. Own versus partner features at the same own prior difference−.186300ppCI[−.751373,+.256350]; own versus partner evidence−.253634ppCI[−1.601092,+1.022079], also unresolved. Both events reach tIoU>=.3 in0/32pairs after update; this small, deliberately difficult, label-selected diagnostic is not a population zero-capability claim.

Swapping prior changes55/64intervals(85.9375%), swapping features33/64(51.5625%). Under a common symmetric prior,17/32pairs output identical intervals and mean pair interval overlap rises to.945470. Exclusive-action evidence contrast on27scorablepairs is+.018156CI[−.049772,+.085046]. These are consistent with material prior conditioning and do not establish independent action-semantic recognition; unequal intervention strength prevents a quantitative causal contribution ranking. Source and the unchanged method can remain useful on broader benchmarks even though this targeted cohort supplies no positive independent semantic evidence.

## Scope and validity

All64counterfactual-array sets sealed before separate offline GT analysis, followed by an independent process rederiving every array bitwise, checking original interval/evidence identity, independently implementing prototype/posterior arithmetic and intervalIoU, verifying actor/object/frame identity, and recomputing all video bootstrap means/CIs.256old officially scored temporal values are reused and cross-checked, not rerun as a new full official tube score. Independent math max discrepancy 1.9984014443252818e-15; 99 scalar bootstrap statistics verified. Three subprocesses and outer controller actually exited0. All signed32pair/64query data, harms/rescues and costs retained.

Previously exposed VidSTG targeted diagnostic,32videos,pointwise10000video bootstrap seed20261104,no familywise correction, no unknown-test/equivalence/action-semantic/generalization claim from a null effect. Natural-query conditions can change upstream Source/expert evidence; feature/prior interventions are mechanistic conditional probes. No parameter, seed, query or method selection based on outcomes; no current hard-anchor fix/paper change/oldqueue restart. Same-target guarantees derive from catalog track IDs, not matching clothing words. Private captions/GTcoordinates/intervals/feature/posterior arrays and identity manifests remain private. Public only protocol/code/derived scalars/hash. No new Source/expert/GPU/model/optimizer, inherited acquisition nonzero; cached CPU work is not deployment latency.

## Sampling/prior-support prerequisite check (offline GT oracle only)

- grid_oracle_t_percent: 94.416445,95%videoCI[93.22218516881311, 95.57661275130599].
- prior_support_oracle_t_percent: 93.661138,95%videoCI[91.74983516374499, 95.29853493067955].
- grid_oracle_both_t03_percent: 100.000000,95%videoCI[100.0, 100.0].
- prior_support_oracle_both_t03_percent: 100.000000,95%videoCI[100.0, 100.0].
- GT_event_duration_seconds: 4.632159,95%videoCI[3.2552946084170102, 6.21745592294117].
- GT_native_no_sample_queries_percent: 0.000000,95%videoCI[0.0, 0.0].
- actual_both_t03_percent: 0.000000,95%videoCI[0.0, 0.0].

Both events can reach tIoU>=.3 on the existing grid in32/32pairs, and within the original strictly-positive q0support in32/32pairs; actual E-ROI update succeeds in0/32pairs. 0/64GT events have no native sampled frame inside. Thus distinguish absent sampled evidence from decoder support; intervals can bracket events even without an internal sample. Oracle calculations use GT after all predictions/separateaudit and do not alter predictions or selection. Independent full endpoint enumeration matches vectorized upper bounds, maxerror0.0. No label-assisted adaptation or oracle-method performance claim.
