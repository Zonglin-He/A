# hc2: standalone retained192 roi review control

Exactly192queries, 192 parents. This pre-existing retained-raw cohort includes previously exposed development and confirmation sources; it is not the full1902/10303 domain, an unseen test, or a prefix accuracy estimate. All192 prediction configurations sealed before this independent official scoring. Separate actual ROI repooling/cold gamma chain, prediction/input/parent/receipt SHA and all-mean audits passed.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|44.661062|23.524963|38.541667|13.541667|
|CorrectedROI|54.096368|30.508756|48.437500|21.354167|
|SourceROI|54.186687|30.610749|48.437500|21.875000|
|GlobalAppearance|53.838140|30.064630|47.916667|20.312500|
|ShiftedROI|52.320639|28.964275|47.916667|18.229167|
|RandomROI|53.423511|29.865006|46.875000|19.270833|

Parent-paired10000-bootstrap pointwise95% CIs, no familywise significance claim:

- CorrectedROI-Source: t+9.435307pp CI[6.847394977493687, 12.057960342977392]; v+6.983792pp CI[4.670941002160638, 9.408689551568013]; severe t/v/union28/28/43, rescues115.
- CorrectedROI-SourceROI: t-0.090319pp CI[-0.5827423440941807, 0.4115242160893628]; v-0.101994pp CI[-0.3737336343452556, 0.16356479464478887]; severe t/v/union13/3/13, rescues10.
- CorrectedROI-GlobalAppearance: t+0.258228pp CI[-0.8546939091942599, 1.3827561733017666]; v+0.444126pp CI[-0.13509895507983177, 1.0697658848130802]; severe t/v/union31/12/31, rescues34.
- CorrectedROI-ShiftedROI: t+1.775729pp CI[0.3123402989173088, 3.288457467294164]; v+1.544481pp CI[0.5947487086983662, 2.5593010465603787]; severe t/v/union27/10/27, rescues40.
- CorrectedROI-RandomROI: t+0.672857pp CI[-0.7244549016938525, 2.105240345320305]; v+0.643750pp CI[-0.12950462321074635, 1.490361380984762]; severe t/v/union30/12/30, rescues36.

Short<=5s/long ground-truth-duration strata are offline only:

- short_le5s, n59, CorrectedROI-Source: v+2.124738pp CI[-1.15469959112641, 5.448543783850435]; t+4.064864 CI[-0.5664566682276667, 8.714811986279681].
- short_le5s, n59, CorrectedROI-SourceROI: v-0.009257pp CI[-0.4380673058144906, 0.3848501283328553]; t+0.512071 CI[-0.4503486423548181, 1.634689594830791].
- short_le5s, n59, CorrectedROI-GlobalAppearance: v+0.202652pp CI[-0.46416234735972445, 0.9361216269267582]; t+0.368943 CI[-1.595685081587725, 2.2297758116475404].
- short_le5s, n59, CorrectedROI-ShiftedROI: v+0.215069pp CI[-1.2556604706657637, 1.4928742330722793]; t+0.371723 CI[-1.9694697184033454, 2.5585347135191188].
- short_le5s, n59, CorrectedROI-RandomROI: v+0.035530pp CI[-0.7247334392876719, 0.8536712757538423]; t+0.439278 CI[-2.221287885384201, 2.7965755151821043].
- long_gt5s, n133, CorrectedROI-Source: v+9.139312pp CI[6.224700975393956, 12.121539018028786]; t+11.817683 CI[8.754668120039954, 14.967892370695123].
- long_gt5s, n133, CorrectedROI-SourceROI: v-0.143133pp CI[-0.48855704913407944, 0.18704617216798802]; t-0.357544 CI[-0.9160019979548257, 0.18077326180097283].
- long_gt5s, n133, CorrectedROI-GlobalAppearance: v+0.551246pp CI[-0.23479146684961916, 1.4065707450485403]; t+0.209114 CI[-1.1659466242105512, 1.5820439691224595].
- long_gt5s, n133, CorrectedROI-ShiftedROI: v+2.134220pp CI[0.9375242978792183, 3.4813321480022177]; t+2.398559 CI[0.5184264114346124, 4.372408976574491].
- long_gt5s, n133, CorrectedROI-RandomROI: v+0.913561pp CI[-0.13421877106821, 2.0277222468153058]; t+0.776475 CI[-0.8726254008134087, 2.4224540947093574].

All appearance arms retain identical q0/eight observations, exact MAIN spatial candidate bank and coefficients, centering/unit/prototype/kappa0/beta1. Only temporal pooling changes, each computes own posterior/interval/finalDP. GlobalAppearance pools all valid native grid cells. ShiftedROI translates the corrected trajectory by a predeclared SHA-selected horizontal quarter image; RandomROI fixes a SHA-derived querywise center with original per-frame widths/heights. Context1.5 and original out-of-grid global fallback remain, with counts reported. These counterfactual controls can alter coverage/temporal smoothness and are not semantic ground truth or pure action-feature attribution. Source is untouched native-time Source boxes, distinct from the SourceROI full chain. No control, scale or main method is selected using these scores.

Global-fallback frame counts: {'CorrectedROI': 22, 'SourceROI': 0, 'ShiftedROI': 190, 'RandomROI': 0}.

All signed positive/negative rows, severe tails/rescues and short/long CIs are retained. Conditional replay elapsed clocks include file SHA, resource checks and actual original-grid pooling; not process CPU or complete deployment latency. New model/Source/expert/optimizer calls are zero. Original acquired Source/eight-expert contexts and old gamma preprocessing are inherited nonzero costs, not free observations. Shared original Vid producer/TTA costs must not be added twice.
