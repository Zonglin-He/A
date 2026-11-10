# R61 — controlled matching-head and last-decoder interventions

**Decision:** R60 was not merely underpowered. Making its same selected channels fully inactive changes decisions but loses overall mIoU; increasing to32does not fix it. Last-decoder language-reading strength and visual-head6show small task-relevant signals, but every fixed intervention remains below R59direct fusion. No broad harmful component or new TTA method is established.

## Fixed experiment

Waymo-only epoch100; exposed DEV336,168per platform,2Drone/3Quadsceneclusters; originalA256boxes and R57ABCsamplings/transport/UNKNOWN.34prespecifiedarms,26actualcontrolled interventions. No GT in intervention construction, no fitting/gradient/pseudo-label/source training/newloss/external model. No FINAL384or full5912access.

Original A prefix reused; old B/C prefix was missing, recovered using the exact saved seeds and pointindices. Engineering4andfull672sourceforwards; all originalboxes/scores/h/query/tokens/mask/pointindicesbitexact; source fingerprint unchanged. Every controlled intervention replays allABCviews, outputs originalAboxes only, and restores all source parameters. Source-zero replay hiddenerror0and maxABCprobabilityerror2.22e-16; every originalTop1preserved. Physical matching-head/out-projection column assignments and native residual-output interventions are separate diagnostic mechanisms, not optimized testtime updates.

Matching tests: R60noise8/common8fully off;noise32highest same excess ranks fully off;random8sameR60destinations;first/secondhidden32largest source-energy contributions removed. Noise32can include nonpositive-excess ranks; expansion is a sensitivity intervention, not a claim all32are harmful. Lastdecoder self/language/vision/FFNresidual gains0,.5,1.5; each of8visual heads separately suppressed through its36out-projection input columns. All downstream attention/norm/FFN/native matching is recalculated. No best arm is chosen per input during inference.

## Complete DEV336 task results

IoUbetter/worse refers to continuousIoU, not independent instance-identity truth. Rescue/break counts refer to crossingIoU>.25relative to frozenABC. All cases including no changes remain in denominators.

| Arm | Drone mIoU% | Quad mIoU% | Balanced mIoU% | Delta ABC pp | Acc25% | Acc50% | Query changes | IoU better/worse | Rescued25/broken25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | -0.472129 | 40.773810 | 21.130952 | 37 | 7/13 | 4/9 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | +0.000000 | 42.261905 | 21.428571 | 0 | 0/0 | 0/0 |
| r58_online_ema_teacher | 15.924842 | 28.841619 | 22.383230 | +0.080770 | 42.559524 | 21.428571 | 27 | 7/7 | 3/2 |
| r58_online_ema_student | 15.868511 | 29.104569 | 22.486540 | +0.184079 | 42.857143 | 21.726190 | 56 | 12/11 | 4/2 |
| r59_direct_all | 16.385568 | 28.836664 | 22.611116 | +0.308656 | 43.154762 | 21.726190 | 38 | 10/8 | 5/2 |
| r60_noise | 15.736332 | 28.868588 | 22.302460 | +0.000000 | 42.261905 | 21.428571 | 0 | 0/0 | 0/0 |
| r60_common | 15.740789 | 28.868588 | 22.304689 | +0.002229 | 42.261905 | 21.428571 | 2 | 1/0 | 0/0 |
| zero_replay | 15.736332 | 28.868588 | 22.302460 | +0.000000 | 42.261905 | 21.428571 | 0 | 0/0 | 0/0 |
| mlp_noise8_off | 15.708431 | 28.876921 | 22.292676 | -0.009784 | 42.559524 | 21.130952 | 10 | 4/3 | 1/0 |
| mlp_noise32_off | 15.728097 | 28.755425 | 22.241761 | -0.060699 | 42.559524 | 21.130952 | 11 | 4/3 | 1/0 |
| mlp_common8_off | 15.715598 | 28.687923 | 22.201760 | -0.100700 | 41.964286 | 21.428571 | 14 | 3/3 | 0/1 |
| mlp_random8_off | 15.561972 | 28.823825 | 22.192898 | -0.109562 | 41.964286 | 21.130952 | 4 | 1/2 | 0/1 |
| mlp_h1_energy32_off | 15.672911 | 28.716313 | 22.194612 | -0.107849 | 42.261905 | 21.428571 | 17 | 3/6 | 1/1 |
| mlp_h2_energy32_off | 14.724086 | 28.931579 | 21.827832 | -0.474628 | 41.071429 | 21.130952 | 39 | 6/11 | 1/5 |
| dec_self_off | 15.800220 | 28.874403 | 22.337311 | +0.034851 | 42.857143 | 21.130952 | 10 | 5/1 | 2/0 |
| dec_self_half | 15.948001 | 28.868588 | 22.408295 | +0.105834 | 42.857143 | 21.428571 | 4 | 3/0 | 2/0 |
| dec_self_boost | 15.891198 | 28.826343 | 22.358770 | +0.056310 | 42.261905 | 21.428571 | 7 | 3/3 | 1/1 |
| dec_language_off | 15.545132 | 28.823825 | 22.184478 | -0.117982 | 41.964286 | 21.130952 | 8 | 1/3 | 0/1 |
| dec_language_half | 15.723950 | 28.871107 | 22.297528 | -0.004932 | 42.261905 | 21.428571 | 6 | 2/1 | 0/0 |
| dec_language_boost | 16.140706 | 28.868588 | 22.504647 | +0.202187 | 43.154762 | 21.428571 | 6 | 4/1 | 3/0 |
| dec_vision_off | 15.542305 | 28.874403 | 22.208354 | -0.094106 | 42.261905 | 21.130952 | 9 | 2/2 | 0/0 |
| dec_vision_half | 15.561972 | 28.874403 | 22.218187 | -0.084273 | 42.261905 | 21.130952 | 4 | 2/1 | 0/0 |
| dec_vision_boost | 15.923583 | 28.826343 | 22.374963 | +0.072503 | 42.559524 | 21.428571 | 7 | 3/3 | 2/1 |
| dec_ffn_off | 15.717242 | 28.874403 | 22.295822 | -0.006638 | 42.559524 | 21.130952 | 10 | 3/2 | 1/0 |
| dec_ffn_half | 15.707095 | 28.868588 | 22.287842 | -0.014618 | 42.559524 | 21.130952 | 6 | 1/2 | 1/0 |
| dec_ffn_boost | 15.933619 | 28.796761 | 22.365190 | +0.062730 | 42.559524 | 21.428571 | 9 | 4/4 | 2/1 |
| dec_vision_head0_off | 15.545132 | 28.871107 | 22.208119 | -0.094341 | 42.261905 | 21.130952 | 8 | 2/2 | 0/0 |
| dec_vision_head1_off | 15.740789 | 28.868588 | 22.304689 | +0.002229 | 42.261905 | 21.428571 | 1 | 1/0 | 0/0 |
| dec_vision_head2_off | 15.736332 | 28.868588 | 22.302460 | +0.000000 | 42.261905 | 21.428571 | 2 | 0/0 | 0/0 |
| dec_vision_head3_off | 15.696704 | 28.871107 | 22.283906 | -0.018555 | 42.261905 | 21.428571 | 6 | 1/3 | 0/0 |
| dec_vision_head4_off | 15.736332 | 28.868588 | 22.302460 | +0.000000 | 42.261905 | 21.428571 | 0 | 0/0 | 0/0 |
| dec_vision_head5_off | 15.561972 | 28.868588 | 22.215280 | -0.087180 | 42.261905 | 21.130952 | 4 | 1/1 | 0/0 |
| dec_vision_head6_off | 16.062046 | 28.874403 | 22.468224 | +0.165764 | 42.857143 | 21.428571 | 10 | 5/2 | 2/0 |
| dec_vision_head7_off | 15.740789 | 28.868588 | 22.304689 | +0.002229 | 42.261905 | 21.428571 | 2 | 1/0 | 0/0 |

## Weak intervention versus wrong component

The same R60noise8selector at original soft attenuation changes0queries. Full removal changes10queries(4IoUimprovements/3worsenings)but mIoUdecreases0.009784pp. Expanding to32changes11queries but decreases0.060699pp. Common8full removal changes14queries,decreases0.100700pp. First/secondhidden32removal changes17/39queries,and decreases0.107849/0.474628pp. Median changed-matrix norm fractions are16.73%fornoise8,33.30%fornoise32,and34.37%forsecondhidden32; this is not a nonzero-but-negligible numerical edit. It does not identify a beneficial selector or demonstrate arbitrary edits of all matched magnitude would fail.

Thus larger perturbations prove matching-head sensitivity but do not repair overall grounding. R60's lack of effects had an amplitude component, yet amplitude alone does not explain lack of beneficial task edits. Do not continue that channel-damping rule by just increasing scale.

## Which last-decoder computation has evidence?

Language residual gain gives22.184478%(off),22.297528%(half),22.302460%(native),22.504647%(1.5x). The direction is monotone in overall mIoUacross these fixed probes.1.5xrescues3of126recoverableerrors,breaks0originalgood25predictions,and changes6queries(4IoUbetter/1worse). Gains occur inbothDroneclusters,Quadmeanunchanged; two scene means improve,noneworsen. This is a small exposed-development signal,still0.106468ppbelowR59directandnot independent confirmation.

Visualhead6off gives22.468224%(+0.165764pp),rescues2,breaks0at25,and changes10queries(5IoUbetter/2worse);3scene means improve,noneworsen. Its rank effects below are mixed,so it is not globally certified as a harmful head. Turningoff the entirevisualresidualinstead loses0.094106pp. Keep output contribution versus attentionQ/K changes distinct: this round does not test a learnedQ/Kupdate.

Selfattentionhalf is+0.105834ppbut its dose relation is not monotone;FFNgains are smaller/mixed. If local-interface work continues, the most justified first probe is finaldecoder cross_l output/read-strength,withvisualhead6as a secondary probe. Matchingchanneldeletionis lowerpriority. No combo withR58orR59was run,and no combo benefit is implied. Closing lastdecoder language does NOT remove earlier language information; closing its vision path does NOT remove earlier point information.

## Recoverability and finite intervention envelope

ABChas142good25predictions and194errors.126errorshave at leastone original candidateIoU>.25;68have noneand cannot be rescued atthisthresholdby any ranking-only method. These labels are offline geometric task criteria,not instance identity annotations.

| Family | Tested arms | Recoverable errors with any query change | Recoverable errors rescued by any arm25 | GT-best tested arm or ABC fallback mIoU% |
|---|---:|---:|---:|---:|
| matching | 6 | 20 | 2 | 22.698688 |
| decoder_self | 3 | 5 | 3 | 22.515393 |
| decoder_language | 3 | 5 | 3 | 22.509696 |
| decoder_vision | 11 | 7 | 3 | 22.499874 |
| decoder_ffn | 3 | 6 | 3 | 22.503688 |
| all_controlled | 26 | 22 | 4 | 22.805084 |

Even selecting the best of the26controlledarms perinputwithGTrescues only4/126recoverableerrors;only22/126changeTop1under any ofthese probes. The GT-best-envelope withABCfallback is22.805084%. This finite-arm diagnostic is NOT a deployable method or a universal architecture capacity bound; interactions/different directions were not exhaustively tested. It weakens a broad 'just suppress a few bad terminal channels/branches' explanation. It is consistent with many incorrect preferences already being stable in the frozen upstream representation, but does not locate an earlier fault by itself.

## Actual ranking, not probability softening

On126recoverableerrors,the bestIoU>.25candidate's baseline rank medianis25.0. A softened negative probability gap can improve without rank improvement,so we added an explicitly postseal descriptive rank check; no method or predictions changed. Language1.5ximproves best-good rankon30,worsenson12,leaves84unchanged. Visualhead6off improves18,worsens23,leaves85unchanged despite its smallTop1task gain. Source singleAcan improve medianrankwhile reducingtotalmIoU;R59can improvetotalmIoUwith medianrank26.0. Therefore rank summaries explain behavior,not replace the registered overall task metric.

| Arm | Recoverable rank improved | Worsened | Same | Median best-good rank after |
|---|---:|---:|---:|---:|
| source | 83 | 26 | 17 | 13.5 |
| frozen_ABC | 0 | 0 | 126 | 25.0 |
| r58_online_ema_teacher | 38 | 35 | 53 | 24.0 |
| r58_online_ema_student | 50 | 37 | 39 | 22.5 |
| r59_direct_all | 64 | 40 | 22 | 26.0 |
| r60_noise | 5 | 9 | 112 | 25.0 |
| r60_common | 13 | 20 | 93 | 25.0 |
| zero_replay | 0 | 0 | 126 | 25.0 |
| mlp_noise8_off | 27 | 21 | 78 | 25.5 |
| mlp_noise32_off | 25 | 24 | 77 | 24.5 |
| mlp_common8_off | 30 | 37 | 59 | 24.5 |
| mlp_random8_off | 23 | 16 | 87 | 25.0 |
| mlp_h1_energy32_off | 57 | 19 | 50 | 24.5 |
| mlp_h2_energy32_off | 40 | 45 | 41 | 24.0 |
| dec_self_off | 27 | 30 | 69 | 25.0 |
| dec_self_half | 18 | 22 | 86 | 25.0 |
| dec_self_boost | 26 | 15 | 85 | 25.5 |
| dec_language_off | 20 | 30 | 76 | 25.5 |
| dec_language_half | 20 | 22 | 84 | 25.5 |
| dec_language_boost | 30 | 12 | 84 | 25.0 |
| dec_vision_off | 24 | 29 | 73 | 25.5 |
| dec_vision_half | 14 | 22 | 90 | 25.5 |
| dec_vision_boost | 34 | 14 | 78 | 25.0 |
| dec_ffn_off | 38 | 24 | 64 | 25.5 |
| dec_ffn_half | 27 | 19 | 80 | 25.5 |
| dec_ffn_boost | 18 | 24 | 84 | 25.5 |
| dec_vision_head0_off | 20 | 18 | 88 | 25.5 |
| dec_vision_head1_off | 13 | 19 | 94 | 25.5 |
| dec_vision_head2_off | 17 | 11 | 98 | 25.0 |
| dec_vision_head3_off | 23 | 18 | 85 | 25.5 |
| dec_vision_head4_off | 14 | 20 | 92 | 25.5 |
| dec_vision_head5_off | 12 | 15 | 99 | 25.0 |
| dec_vision_head6_off | 18 | 23 | 85 | 25.5 |
| dec_vision_head7_off | 7 | 17 | 102 | 25.0 |

## Five exposed scenes

| Platform | Anonymous cluster | n | Arm | mIoU% |
|---|---:|---:|---|---:|
| drone | 1 | 106 | frozen_ABC | 10.711738 |
| drone | 1 | 106 | r59_direct_all | 11.761612 |
| drone | 1 | 106 | mlp_noise8_off | 10.392212 |
| drone | 1 | 106 | mlp_noise32_off | 10.423382 |
| drone | 1 | 106 | dec_language_boost | 11.084392 |
| drone | 1 | 106 | dec_vision_head6_off | 10.943817 |
| drone | 2 | 62 | frozen_ABC | 24.326767 |
| drone | 2 | 62 | r59_direct_all | 24.291041 |
| drone | 2 | 62 | mlp_noise8_off | 24.797449 |
| drone | 2 | 62 | mlp_noise32_off | 24.797449 |
| drone | 2 | 62 | dec_language_boost | 24.785372 |
| drone | 2 | 62 | dec_vision_head6_off | 24.812567 |
| quad | 1 | 62 | frozen_ABC | 10.506319 |
| quad | 1 | 62 | r59_direct_all | 9.632116 |
| quad | 1 | 62 | mlp_noise8_off | 10.528899 |
| quad | 1 | 62 | mlp_noise32_off | 10.199682 |
| quad | 1 | 62 | dec_language_boost | 10.506319 |
| quad | 1 | 62 | dec_vision_head6_off | 10.522075 |
| quad | 2 | 49 | frozen_ABC | 35.070901 |
| quad | 2 | 49 | r59_direct_all | 36.067581 |
| quad | 2 | 49 | mlp_noise8_off | 35.070901 |
| quad | 2 | 49 | mlp_noise32_off | 35.070901 |
| quad | 2 | 49 | dec_language_boost | 35.070901 |
| quad | 2 | 49 | dec_vision_head6_off | 35.070901 |
| quad | 3 | 57 | frozen_ABC | 43.509770 |
| quad | 3 | 57 | r59_direct_all | 43.509770 |
| quad | 3 | 57 | mlp_noise8_off | 43.509770 |
| quad | 3 | 57 | mlp_noise32_off | 43.509770 |
| quad | 3 | 57 | dec_language_boost | 43.509770 |
| quad | 3 | 57 | dec_vision_head6_off | 43.509770 |

## Audit, cost and failure history

Independent functional reconstruction checks sourcecheckpointsubset,oldABCstates,selectors,and9072noninheritednative readouts. Maxindependenthiddenerror1.67e-06,scoreerror3.33e-07;allselectedqueriesexact. Difference from producerbitexactzerois due to functionalmodule operation ordering and is within the prespecified tolerance,no tolerance change. All original boxes preserved. One postseal saved-IoUscore11424values,zero new rawGT/geometrycalls. Independent aggregates/sceneserror7.11e-15pp,5159directednegativecomparisons exact. Independentlychecked8736process/error/marginrecords and11424rankrecords. OriginalGOODandallnegativecasesretained.

SourceB/Ccapture81.97s,controlleddownstreamreplay105.25s,independentnativeaudit58.31s,serialsingleGPU. Fullcapturepeak678437888bytes,replaypeak39556096bytes;nooptimizersteps. StandardunusedRobertaPooler initialization warning occurs before strict sourcecheckpointload; fullsource fingerprint and all oldforward outputs verify the actual source,so no training was initiated. One postseal diagnostic serialization attempt failed on numpyint64; preserved draft/error and fixed explicitPythonint only,no predictions/scores/tolerances changed. No full/source model inference failed.

## Scope and stopping decision

NoTTArecipeadopted,no automatic newloss/fitting/fulltarget/holdout/source training. PreserveR59directfusionandR58references. Retaincross_lread-strengthandvisualhead6onlyasdevelopmentmechanismprobes. The answer is not simply 'make the edit stronger':strongerterminalperturbations can change errors but mostly do not fix them,and the best limited signals remain belowdirectuse. Further work needs a better-founded entity/readout mechanism or an explicitly tested earlier-stage issue,not anotherblindamplitudeincrease. DEV336has historical label exposure and onlyfiveclusters;no independent confirmation or frame-independentCI is claimed.
