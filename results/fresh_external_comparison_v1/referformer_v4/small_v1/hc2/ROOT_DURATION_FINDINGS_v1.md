# hc2: root short/long interpretation

GT event duration groups were used only for offline reporting after complete cohort seal; no samples are excluded. Pointwise parent-cluster95% CIs below are not familywise-adjusted or causal duration effects.

## short_le5s: 42 queries

- CurrentOurs-CurrentSimple: t+3.124530pp CI[-1.6311748558741703, 7.81248523796055]; v+1.355058pp CI[-2.1842523244173018, 4.798607515702112]; v improve/harm/tie20/14/8.
- AlternativeOurs-AlternativeSimple: t+2.907383pp CI[-1.6001344585129067, 7.466073895056797]; v+2.219717pp CI[-0.6409076194236949, 5.227387416681587]; v improve/harm/tie20/14/8.
- AlternativeSimple-CurrentSimple: t+0.000000pp CI[0.0, 0.0]; v-1.977448pp CI[-5.616232443544686, 1.2583126844986692]; v improve/harm/tie12/17/13.
- AlternativeOurs-CurrentOurs: t-0.217146pp CI[-1.3491803681518024, 0.8993673508532353]; v-1.112788pp CI[-3.8133305916591222, 1.3799783419544822]; v improve/harm/tie14/20/8.
## long_gt5s: 86 queries

- CurrentOurs-CurrentSimple: t+11.064312pp CI[7.3082225729666765, 14.90077110091356]; v+6.648633pp CI[4.22969959043269, 9.151529164869348]; v improve/harm/tie59/21/6.
- AlternativeOurs-AlternativeSimple: t+10.873626pp CI[6.971191334474249, 14.805344112495174]; v+5.582799pp CI[3.401998182960214, 7.873904474707214]; v improve/harm/tie59/21/6.
- AlternativeSimple-CurrentSimple: t+0.000000pp CI[0.0, 0.0]; v-4.699846pp CI[-8.26005892562682, -1.4795802494425374]; v improve/harm/tie33/47/6.
- AlternativeOurs-CurrentOurs: t-0.190685pp CI[-1.1830187781435788, 0.76311279354163]; v-5.765680pp CI[-9.616978510570371, -2.2793291252194883]; v improve/harm/tie29/52/5.

Both experts have positive long-event structured-minus-simple mean t/v gains with CIs above zero; both short-event gain CIs cross zero. Alternative-minus-current complete pipeline v loss is clear in long events and uncertain in the smaller short group. The aggregate advantage over each simple fusion must not be presented as established for both duration groups.

Single external expert per branch does not equalize parameters, prior knowledge, input resolution, processed frame count or complete uncached compute. Original current masks are used only in the separate baseline, not the replacement inference. All negative short/long rows remain in the full report and signed TSVs.
