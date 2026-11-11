# R68 — bounded fusion/update-strength optimization and random repeats

One user-authorized DEV336 development search, not new algorithms, source training or independent confirmation. Same Waymo-only epoch100, A256 boxes, originalABC/source features, native first-noun readout, R57 transport and UNKNOWN, .07temperature, source matching head, existingR66 feedback groups/directions. Exactly12 primary conditions:5gamma,4eta,3randomrepeats. No cross-axis combinations or adaptive grid expansion. Historical R66/R67 fixed-comparison decisions remain intact; this newly authorized round may develop parameters on already exposedDEV336. FINAL384/full5912/source training not accessed.

## Locked search and actual execution

Gamma={0,.5,1,1.5,2} modifies only the outerconfidence clip(gamma*c,0,1) in R59. Internalrho, targetdirection construction, originalcorrespondence/masses/UNKNOWN all fixed. Gamma0 returnsABC, gamma1 returnsR59; both endpoints checked before taskscoring. Intermediate scores may change without changing Top1 because argmax is piecewise constant.

Eta={.25,.5,.75,1} loads actualsource+eta*(acceptedR66-source) float32 parameters of the existing184960-parameterMLP. Eta1 uses exactaccepted snapshot. Originalpositive/negative/tested groups and initialgradient direction never recalculated. Each eta rechecks originalABconditionalset+sourceKL loss decrease with same permitted criterion; failure restores source, no per-input GT strength choice. All47feedback inputs pass all4eta levels, giving188accepted nonzero parameter edits across conditions;289no-feedback percondition stay in denominators. No new optimizer/gradient steps. Every parameterstate/ABacceptance seal precedes C readout. Independentfunctional audit reproduces arrays and acceptance before opening C.

Random repeat0 is the actualR67 result. Two new fixedhash prefixes are locked before inference. SA3 remains256D+256R atoriginal512budget, sourceSA2 andanchors unchanged, R59gamma1 unchanged. Reuse R67's independently audited SA2 captures; no new fullbackbone/source forward needed. Additional2016suffixreplays and independent2016manualsampler/SA3/nativeSA4/FP/encoderdecoderreplays. This reuses a proven interface, not a new samplingalgorithm. Seeds are repeats: no bestseed selection, no three-branch ensemble prediction.

Primary metric is overallplatform-equalDEV336mIoU. Selectbestglobalgamma andbestglobaleta independently, ties within1e-10 prefer smallest coefficient; compare both with three-repeatmean. Familyties prefer no-updategamma, thenrandompolicy, theneta. C is diagnostic, not a veto for the R59deployment-readout objective; this does not revise R66's prior joint-gate record or prove cross-sampling transfer. No allscene/zeroharm gate.

## Complete task table

E denotes actualweightdelta scaling atoriginalgamma1; G denotes unadaptedhead outerfusionstrength. C_E are diagnostics, not four additional hyperparameter searches. R59/R66 are explicit inherited reference outputs. Units percent/pp; all negatives retained.

| method | Drone_mIoU | Quad_mIoU | balanced_mIoU | Acc25 | Acc50 | gain_vs_R59_pp | improved | worsened | query_changes |
|---|---|---|---|---|---|---|---|---|---|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 | -0.780785 | 14 | 21 | 67 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 | 0.000000 | 0 | 0 | 0 |
| R66 | 16.657288 | 28.979993 | 22.818641 | 42.857143 | 22.023810 | 0.207525 | 5 | 2 | 12 |
| C_source | 16.085318 | 29.342198 | 22.713758 | 43.154762 | 21.726190 | 0.102642 | 15 | 15 | 76 |
| G_0p0 | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 | -0.308656 | 8 | 10 | 38 |
| G_0p5 | 16.333644 | 28.691211 | 22.512427 | 43.154762 | 21.726190 | -0.098688 | 3 | 4 | 19 |
| G_1p0 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 | 0.000000 | 0 | 0 | 0 |
| G_1p5 | 16.610028 | 28.421280 | 22.515654 | 42.559524 | 22.023810 | -0.095462 | 4 | 5 | 16 |
| G_2p0 | 16.610028 | 28.279697 | 22.444862 | 42.559524 | 22.023810 | -0.166254 | 2 | 5 | 15 |
| E_0p25 | 16.378010 | 28.992784 | 22.685397 | 43.154762 | 21.726190 | 0.074281 | 2 | 1 | 4 |
| E_0p5 | 16.181115 | 28.992784 | 22.586950 | 42.857143 | 21.726190 | -0.024166 | 3 | 2 | 9 |
| E_0p75 | 16.185573 | 28.992784 | 22.589178 | 42.857143 | 21.726190 | -0.021938 | 4 | 2 | 11 |
| E_1p0 | 16.657288 | 28.979993 | 22.818641 | 42.857143 | 22.023810 | 0.207525 | 5 | 2 | 12 |
| C_E_0p25 | 16.085318 | 29.342198 | 22.713758 | 43.154762 | 21.726190 | 0.102642 | 15 | 15 | 76 |
| C_E_0p5 | 15.850826 | 29.342198 | 22.596512 | 42.559524 | 21.726190 | -0.014604 | 15 | 18 | 80 |
| C_E_0p75 | 15.850826 | 29.342198 | 22.596512 | 42.559524 | 21.726190 | -0.014604 | 15 | 18 | 81 |
| C_E_1p0 | 15.850826 | 29.342198 | 22.596512 | 42.559524 | 21.726190 | -0.014604 | 15 | 18 | 81 |
| random_s0 | 16.834981 | 28.722168 | 22.778575 | 44.047619 | 21.726190 | 0.167459 | 10 | 7 | 41 |
| random_s1 | 16.479126 | 29.115523 | 22.797324 | 43.452381 | 22.023810 | 0.186208 | 5 | 7 | 37 |
| random_s2 | 16.563661 | 28.763916 | 22.663788 | 43.452381 | 22.321429 | 0.052672 | 6 | 5 | 38 |

## Development selection

- state: R68_DEV336_ONLY_GLOBAL_CONFIGURATION_SELECTED_NO_CONFIRMATION
- best_gamma: 1.0
- best_gamma_mIoU: 22.61111580683559
- best_eta: 1.0
- best_eta_mIoU: 22.818640568916397
- best_eta_minus_best_no_update_gamma_pp: 0.2075247620808085
- random_three_repeat_mean: 22.746562372000938
- selected_family: R66_parameter_delta_eta
- selected_coefficient: 1.0
- selected_development_mIoU: 22.818640568916397
- random_seed_chosen: False
- random_repeats_ensembled: False
- C_used_for_selection: False
- final_holdout_accessed: False
- source_training: False
- new_TTA_robustly_adopted: False
- no_cross_axis_combination: True
- one_finite_table_only: True
- postscore_grid_expanded: False

Besteta minus bestno-updategamma: **+0.207525 pp**. This compares separately optimized existing branches under the authorized finite grids; it does not test eta with a tuned gamma combination. Selectedconfiguration is development-only. No protected-data confirmation was run and no robust newTTA adoption is claimed.

Within the five tested outerfusion strengths, the original gamma=1.0 remains best; within the four tested actualparameter scales, eta=1.0 remains best. Thus this finite search found no higher-mIoU replacement for the original strengths. Eta.5/.75 fall below R59 while eta.25 is mildly positive, so the tested update axis has not established a broad positive operating region. All three prespecified random repeats exceed R59 (mean gain +0.135447pp), supporting a repeatable direction on this exposed cohort, not independent-data generalization. Per-level concentration is retained below; repeated positivity does not mean broad instance-level improvement.

## Random repetition, not seed selection or ensemble

- state: THREE_PRESPECIFIED_REPEATS_NOT_BEST_SEED_OR_ENSEMBLE
- seed_mIoU: [22.778574624497733, 22.797324292366465, 22.66378819913861]
- mean_mIoU: 22.746562372000938
- min_mIoU: 22.66378819913861
- max_mIoU: 22.797324292366465
- std_population: 0.05902858223537959
- mean_Drone_mIoU: 16.625922429503746
- mean_Quad_mIoU: 28.86720231449813
- gain_vs_R59_pp: 0.13544656516534914
- all_three_over_R59: True

| platform | scene_alias | n | repeat_mean_mIoU | repeat_min_mIoU | repeat_max_mIoU | repeat_mean_gain_vs_R59_pp |
|---|---|---|---|---|---|---|
| drone | D_S01 | 106 | 11.928209 | 11.828733 | 12.041321 | 0.166597 |
| drone | D_S02 | 62 | 24.657498 | 24.295403 | 25.247293 | 0.366457 |
| quad | Q_S01 | 62 | 9.884710 | 9.665637 | 10.025807 | 0.252594 |
| quad | Q_S02 | 49 | 36.190555 | 34.845945 | 37.277685 | 0.122974 |
| quad | Q_S03 | 57 | 43.219312 | 42.931836 | 43.794263 | -0.290458 |

The average is an average of three independently evaluated policies, not an output ensemble and not three times the deployment input budget. Min/max are repeat variability, not an independent-scene confidence interval. The originalseed's +.167459pp is retained regardless of whether later repeats improve. Do not substitute bestseed for policymean.

## Concentration and coefficient ranges

The largestpositive contribution and the remaining335-contribution use the original336denominator. No input is dropped or used to make a GT-dependent trigger. Strong single-case dependence is diagnostic, not a new adoption veto. Complete levels retained; no monotonic or smooth-curve requirement.

| method | gain_vs_R59_pp | largest_positive_full336_contribution_pp | remainder_full336_gain_pp | improved | worsened |
|---|---|---|---|---|---|
| G_0p0 | -0.308656 | 0.076801 | -0.385456 | 8 | 10 |
| G_0p5 | -0.098688 | 0.078060 | -0.176748 | 3 | 4 |
| G_1p0 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |
| G_1p5 | -0.095462 | 0.152181 | -0.247643 | 4 | 5 |
| G_2p0 | -0.166254 | 0.152181 | -0.318435 | 2 | 5 |
| E_0p25 | 0.074281 | 0.078060 | -0.003779 | 2 | 1 |
| E_0p5 | -0.024166 | 0.078060 | -0.102226 | 3 | 2 |
| E_0p75 | -0.021938 | 0.078060 | -0.099998 | 4 | 2 |
| E_1p0 | 0.207525 | 0.234305 | -0.026781 | 5 | 2 |
| random_s0 | 0.167459 | 0.146305 | 0.021154 | 10 | 7 |
| random_s1 | 0.186208 | 0.183443 | 0.002766 | 5 | 7 |
| random_s2 | 0.052672 | 0.183443 | -0.130770 | 6 | 5 |

## All five historically exposed scene clusters

| method | platform | scene_alias | n | mIoU | gain_vs_R59_pp |
|---|---|---|---|---|---|
| source | drone | D_S01 | 106 | 10.584991 | -1.176621 |
| source | drone | D_S02 | 62 | 24.011174 | -0.279868 |
| source | quad | Q_S01 | 62 | 10.117522 | 0.485406 |
| source | quad | Q_S02 | 49 | 32.998900 | -3.068681 |
| source | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| R59 | drone | D_S01 | 106 | 11.761612 | 0.000000 |
| R59 | drone | D_S02 | 62 | 24.291041 | 0.000000 |
| R59 | quad | Q_S01 | 62 | 9.632116 | 0.000000 |
| R59 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| R59 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| R66 | drone | D_S01 | 106 | 12.536308 | 0.774696 |
| R66 | drone | D_S02 | 62 | 23.702835 | -0.588206 |
| R66 | quad | Q_S01 | 62 | 10.020492 | 0.388376 |
| R66 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| R66 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| C_source | drone | D_S01 | 106 | 11.328022 | -0.433590 |
| C_source | drone | D_S02 | 62 | 24.218760 | -0.072281 |
| C_source | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_source | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_source | quad | Q_S03 | 57 | 44.325635 | 0.815865 |
| G_0p0 | drone | D_S01 | 106 | 10.711738 | -1.049873 |
| G_0p0 | drone | D_S02 | 62 | 24.326767 | 0.035725 |
| G_0p0 | quad | Q_S01 | 62 | 10.506319 | 0.874203 |
| G_0p0 | quad | Q_S02 | 49 | 35.070901 | -0.996679 |
| G_0p0 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| G_0p5 | drone | D_S01 | 106 | 11.489957 | -0.271655 |
| G_0p5 | drone | D_S02 | 62 | 24.614785 | 0.323744 |
| G_0p5 | quad | Q_S01 | 62 | 10.025685 | 0.393569 |
| G_0p5 | quad | Q_S02 | 49 | 35.070901 | -0.996679 |
| G_0p5 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| G_1p0 | drone | D_S01 | 106 | 11.761612 | 0.000000 |
| G_1p0 | drone | D_S02 | 62 | 24.291041 | 0.000000 |
| G_1p0 | quad | Q_S01 | 62 | 9.632116 | 0.000000 |
| G_1p0 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| G_1p0 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| G_1p5 | drone | D_S01 | 106 | 11.927999 | 0.166388 |
| G_1p5 | drone | D_S02 | 62 | 24.614785 | 0.323744 |
| G_1p5 | quad | Q_S01 | 62 | 9.949950 | 0.317834 |
| G_1p5 | quad | Q_S02 | 49 | 34.241251 | -1.826330 |
| G_1p5 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| G_2p0 | drone | D_S01 | 106 | 11.927999 | 0.166388 |
| G_2p0 | drone | D_S02 | 62 | 24.614785 | 0.323744 |
| G_2p0 | quad | Q_S01 | 62 | 9.566304 | -0.065812 |
| G_2p0 | quad | Q_S02 | 49 | 34.241251 | -1.826330 |
| G_2p0 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| E_0p25 | drone | D_S01 | 106 | 11.749633 | -0.011979 |
| E_0p25 | drone | D_S02 | 62 | 24.291041 | 0.000000 |
| E_0p25 | quad | Q_S01 | 62 | 10.055151 | 0.423035 |
| E_0p25 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| E_0p25 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| E_0p5 | drone | D_S01 | 106 | 11.788683 | 0.027072 |
| E_0p5 | drone | D_S02 | 62 | 23.690758 | -0.600284 |
| E_0p5 | quad | Q_S01 | 62 | 10.055151 | 0.423035 |
| E_0p5 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| E_0p5 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| E_0p75 | drone | D_S01 | 106 | 11.788683 | 0.027072 |
| E_0p75 | drone | D_S02 | 62 | 23.702835 | -0.588206 |
| E_0p75 | quad | Q_S01 | 62 | 10.055151 | 0.423035 |
| E_0p75 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| E_0p75 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| E_1p0 | drone | D_S01 | 106 | 12.536308 | 0.774696 |
| E_1p0 | drone | D_S02 | 62 | 23.702835 | -0.588206 |
| E_1p0 | quad | Q_S01 | 62 | 10.020492 | 0.388376 |
| E_1p0 | quad | Q_S02 | 49 | 36.067581 | 0.000000 |
| E_1p0 | quad | Q_S03 | 57 | 43.509770 | 0.000000 |
| C_E_0p25 | drone | D_S01 | 106 | 11.328022 | -0.433590 |
| C_E_0p25 | drone | D_S02 | 62 | 24.218760 | -0.072281 |
| C_E_0p25 | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_E_0p25 | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_E_0p25 | quad | Q_S03 | 57 | 44.325635 | 0.815865 |
| C_E_0p5 | drone | D_S01 | 106 | 11.307484 | -0.454128 |
| C_E_0p5 | drone | D_S02 | 62 | 23.618477 | -0.672565 |
| C_E_0p5 | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_E_0p5 | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_E_0p5 | quad | Q_S03 | 57 | 44.325635 | 0.815865 |
| C_E_0p75 | drone | D_S01 | 106 | 11.307484 | -0.454128 |
| C_E_0p75 | drone | D_S02 | 62 | 23.618477 | -0.672565 |
| C_E_0p75 | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_E_0p75 | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_E_0p75 | quad | Q_S03 | 57 | 44.325635 | 0.815865 |
| C_E_1p0 | drone | D_S01 | 106 | 11.307484 | -0.454128 |
| C_E_1p0 | drone | D_S02 | 62 | 23.618477 | -0.672565 |
| C_E_1p0 | quad | Q_S01 | 62 | 10.117087 | 0.484971 |
| C_E_1p0 | quad | Q_S02 | 49 | 36.238137 | 0.170556 |
| C_E_1p0 | quad | Q_S03 | 57 | 44.325635 | 0.815865 |
| random_s0 | drone | D_S01 | 106 | 11.914572 | 0.152960 |
| random_s0 | drone | D_S02 | 62 | 25.247293 | 0.956252 |
| random_s0 | quad | Q_S01 | 62 | 10.025807 | 0.393691 |
| random_s0 | quad | Q_S02 | 49 | 34.845945 | -1.221636 |
| random_s0 | quad | Q_S03 | 57 | 43.794263 | 0.284493 |
| random_s1 | drone | D_S01 | 106 | 11.828733 | 0.067121 |
| random_s1 | drone | D_S02 | 62 | 24.429797 | 0.138756 |
| random_s1 | quad | Q_S01 | 62 | 9.962687 | 0.330571 |
| random_s1 | quad | Q_S02 | 49 | 37.277685 | 1.210104 |
| random_s1 | quad | Q_S03 | 57 | 42.931836 | -0.577934 |
| random_s2 | drone | D_S01 | 106 | 12.041321 | 0.279709 |
| random_s2 | drone | D_S02 | 62 | 24.295403 | 0.004362 |
| random_s2 | quad | Q_S01 | 62 | 9.665637 | 0.033521 |
| random_s2 | quad | Q_S02 | 49 | 36.448035 | 0.380454 |
| random_s2 | quad | Q_S03 | 57 | 42.931836 | -0.577934 |

## Audits, limits and decision

All predictions sealed before a single saved-original-candidate-IoU lookup perarm; no rawGT geometry rescoring, newannotation/media scoring, oraclefeedback or FINALaccess. Independent functional parameter loads/acceptance and all20readoutarms audited; savedtask summaries/negative comparisons/scenes independently checked; derived globalchoice/repeatmeans/pointconcentration verified. Same originalA256 and allUNKNOWN/fallback/no-feedback rows retained. DEV336 consists of168/platform in2Drone+3Quad already exposed scenes; performance optimization is allowed here but remains development evidence.

This round optimizes known functions, not mechanistic reliability. A bettereta does not prove R66's relative-regioncontrast is a semanticlabel; a bettergamma is a stronger sameinformation no-update baseline. Randomvariability cannot be assigned to featureFPS, which remains its preserved negative historical configuration without new scale/quota/site search. C diagnostics cannot be called newphysicalview/independentinput; it is another sampling of the samecloud. Do not silently turn favorablecurrentreadout scores into a transferclaim.

Stop this finite table here. Preserve every alternative and select only the explicitly reported globaldevelopmentconfiguration. No immediate expandedgrid, no mixingR66withnewrandom, no newloss/module, and no automaticconfirmation on protecteddata. Originalsource/feature/weight/media/description/coordinate/ID/secrets and upstreamsnapshots excluded from publicpackage. Independentprivacy scan, everyimmutableremotebyte and all oldstudy/root object preservation precede mainCAS. Collectiontimes include cacheI/O/audits and are not isolated per-policy deploymentlatency.

Publication preparation attempt1 failed on stale copied publisher/auditor script filenames; the subsequent exporter did not start because authorization was not created. Both are local metadata failures, preserved with no scientific prediction/score/selection change and no network write. Corrected attempt2 is independently audited.
