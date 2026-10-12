# Controlled LN State Intervention

One fixed historically exposed diagnostic panel: the explicitly selected P9/P0 previous-P6 32 parents in HC and 32 in VidSTG. No score-based case selection; zero query overlap with the separate TENT/SAR calibration 32. This is a controlled mechanism study, not a held-out performance test.

Three arms per query: Source LN, inherited LN from original order1, inherited LN from original order2. Only six LN tensors (1536 coordinates) change at entry. Query Residual starts at exact zero and Adam is fresh. The original joint 1792-coordinate OPD update, final round, clean pixels, source checkpoint, decoder inputs, Omega0 teacher evidence, admission, positions and antithetic noise remain fixed. Actions and gradients can depend on state. There is no new cross-query writeback.

All24 actual complete qualification episodes used two genuinely fresh frozen captures per fixed query and exact complete repeats. All192 formal episodes/4800 rounds are new full fits. The128 inherited formal fits exactly reproduce the corresponding old original Full episodes;12 formal fits exactly match their qualification fits. The21 empty formal episodes (7 Vid queries x3) have no sampler or optimizer and remain in every table. All predictions were sealed globally before raw GT scoring. No new DINO call, teacher, gate, tuning or production/main experiment.

Primary U comprises genuine annotated decoder frames within the GT event, excluding every actually visited Omega0 observation (including rejected teachers). Secondary common-U also excludes Omega1 visits for P0 comparability. No dense interpolation is used to manufacture genuine U. Before/After are computed separately for each initial state, with an identical mask across arms. Empty U stays unknown. Full dense s/v/t and the observed region are diagnostics. HC preserves the original coordinate lower-bound clipping.

All estimates use32 parent units per domain,10000 fixed paired parent bootstrap draws, seed20261013 and percentile95% intervals. Three prespecified contrasts and negative/neutral/unknown results are fully reported. Intervals and fixed1/5/20pp tiers are descriptive, with no multiple-comparison adjustment or held-out claim.

|Target|Initial state|U Before (%)|U After (%)|Update utility (pp),95% CI|Positive/negative/neutral/unknown|Harm <-5pp/<-20pp|
|---|---|---|---|---|---|---|
|hc2|Source|+47.932 [+38.912, +56.645]|+47.013 [+37.027, +56.805]|-0.919 [-7.308, +4.176]|18/13/1/0|4/3|
|hc2|Inherited_order1|+51.467 [+42.021, +60.576]|+47.398 [+37.219, +57.352]|-4.070 [-11.328, +1.542]|16/15/1/0|6/3|
|hc2|Inherited_order2|+52.133 [+42.580, +61.494]|+47.360 [+37.293, +57.279]|-4.774 [-11.359, +0.599]|12/19/1/0|9/4|
|vidstg|Source|+35.980 [+28.032, +43.889]|+47.784 [+38.316, +56.815]|+11.804 [+4.412, +18.462]|22/2/8/0|1/1|
|vidstg|Inherited_order1|+40.716 [+31.825, +49.411]|+48.755 [+39.083, +58.236]|+8.039 [+1.074, +14.686]|18/6/8/0|4/3|
|vidstg|Inherited_order2|+39.869 [+31.078, +48.712]|+47.398 [+37.414, +56.922]|+7.529 [+0.518, +14.026]|18/6/8/0|4/2|

|Target|Contrast (first -> second)|Mean utility difference (pp),95% CI|Mean absolute difference (pp),95% CI|Flips|Both >=1pp/>=5pp|Difference >=5pp/>=20pp|Unknown|
|---|---|---|---|---|---|---|---|
|hc2|Source_to_Inherited_order1|-3.151 [-4.694, -1.821]|+3.623 [+2.448, +5.030]|4/32|2/0|8/0|0|
|hc2|Source_to_Inherited_order2|-3.855 [-5.488, -2.046]|+5.021 [+3.728, +6.395]|8/32|6/0|15/0|0|
|hc2|Inherited_order1_to_Inherited_order2|-0.704 [-2.264, +0.939]|+3.357 [+2.243, +4.589]|6/32|3/0|6/0|0|
|vidstg|Source_to_Inherited_order1|-3.765 [-6.822, -0.727]|+6.500 [+4.146, +9.073]|4/32|3/1|13/2|0|
|vidstg|Source_to_Inherited_order2|-4.275 [-7.796, -0.987]|+6.971 [+4.384, +10.083]|4/32|4/1|15/1|0|
|vidstg|Inherited_order1_to_Inherited_order2|-0.509 [-3.383, +1.819]|+3.827 [+1.959, +6.372]|2/32|1/0|7/1|0|

The initial-state effect is evaluated on each arm's own Before-to-After utility, not on final accuracy alone. A signed mean near zero can coexist with material causal changes in opposite directions. Under the predeclared rule, a positive lower interval bound for absolute utility difference plus material sign reversals or >=5pp differences supports receiving-state dependence on this panel.

- hc2: fixed-panel material LN state dependence supported = True; qualifying prespecified contrasts: Source_to_Inherited_order1, Source_to_Inherited_order2, Inherited_order1_to_Inherited_order2.
- vidstg: fixed-panel material LN state dependence supported = True; qualifying prespecified contrasts: Source_to_Inherited_order1, Source_to_Inherited_order2, Inherited_order1_to_Inherited_order2.

This is a causal intervention on the saved entering LN vectors while the remaining source model/input/evidence/update/noise are fixed. The exact repeat and original-reproduction controls eliminate a capture or replay change as the source of these differences. It does not prove generalization to unseen parents, identify which LN subcomponents cause the effect, establish a universal harm of inheritance, or provide a reliable unlabelled decision signal. No GT-selected state or step is used online.

Inherited states already have different Before accuracy. For example, Vid order1 starts at40.716% versus35.980% for Source, and finishes at48.755% versus47.784%, while its update utility is8.039pp versus11.804pp. Smaller update utility therefore does not automatically mean worse final prediction. This total LN-initialization effect does not separately identify initial-position/headroom effects versus changed gradients or decoder sensitivity. Same noise also does not mean same candidate boxes: the original state-dependent mean changes sampled actions and their teacher rewards.

The result freezes this experiment. No automatic gate, predictor, prototype, new loss, wider state search, paper drafting or CURRENT_METHOD promotion follows. P7-A remains frozen. Cancelled TENT/SAR formal evaluation remains cancelled with its original accepted prefix/optimizer/process identity in T standby.

Independent root: 192 complete fits, 4800 rounds, 40051200 independent Adam scalar checks, 2100480 exact replayed sample coordinates, 6650 independent outcome/statistic checks. Full model forward was actually run; the independent CPU root does not independently replay the full decoder Jacobian. Actual native head VJP was checked during qualification.

Complete anonymous64-parent x3-arm Before/After/utility/real-mask counts/empty-episode/full dense diagnostics are in ROWS.jsonl.gz and ALL_PARENT_THREE_ARM_READOUTS.csv. All domain/arm/contrast/transition/tail/CI statistics are in SUMMARY.json. Private model states, raw fits, samples, teacher/GT geometry, media and text stay local.
