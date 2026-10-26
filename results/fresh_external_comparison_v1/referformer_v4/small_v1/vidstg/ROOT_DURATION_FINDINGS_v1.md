# vidstg: root short/long interpretation

GT event duration groups were used only for offline reporting after complete cohort seal; no samples are excluded. Pointwise parent-cluster95% CIs below are not familywise-adjusted or causal duration effects.

## short_le5s: 78 queries

- CurrentOurs-CurrentSimple: t-0.552505pp CI[-2.401443869986275, 1.3012908728656365]; v-0.246146pp CI[-1.3495918038329964, 0.7845087915369285]; v improve/harm/tie26/24/28.
- AlternativeOurs-AlternativeSimple: t-0.728254pp CI[-2.6436942626233053, 1.1319952272439024]; v-0.320108pp CI[-1.3576400521812761, 0.6189768465339988]; v improve/harm/tie26/26/26.
- AlternativeSimple-CurrentSimple: t+0.000000pp CI[0.0, 0.0]; v+0.322126pp CI[-1.4995547667237008, 2.4079966021680317]; v improve/harm/tie24/24/30.
- AlternativeOurs-CurrentOurs: t-0.175750pp CI[-1.1551360894835945, 0.619557663273123]; v+0.248164pp CI[-1.4819025551145242, 2.2629084556549364]; v improve/harm/tie21/35/22.
## long_gt5s: 50 queries

- CurrentOurs-CurrentSimple: t+8.749011pp CI[4.203408202871081, 13.965947699051359]; v+4.150660pp CI[1.464663153371928, 6.932496408754038]; v improve/harm/tie31/15/4.
- AlternativeOurs-AlternativeSimple: t+7.490064pp CI[3.014762309564963, 12.725718766938261]; v+3.927966pp CI[1.865728340824131, 6.205912158564138]; v improve/harm/tie33/10/7.
- AlternativeSimple-CurrentSimple: t+0.000000pp CI[0.0, 0.0]; v+0.603098pp CI[-3.5012326518747496, 4.778201163564348]; v improve/harm/tie26/20/4.
- AlternativeOurs-CurrentOurs: t-1.258947pp CI[-3.4249593903483118, 0.4900047124991056]; v+0.380404pp CI[-4.770746193866906, 5.540298032698887]; v improve/harm/tie22/25/3.

Interpret each duration-specific CI directly; positive aggregate gain does not establish gain in both strata. Differences in expert quality and temporal-event prediction remain part of pipeline comparisons, not isolated propagation effects.

Single external expert per branch does not equalize parameters, prior knowledge, input resolution, processed frame count or complete uncached compute. Original current masks are used only in the separate baseline, not the replacement inference. All negative short/long rows remain in the full report and signed TSVs.
