# TA-STVG: replace fused motion feedback with native local appearance

Fixed previously exposedHC32/Vid32, one candidate, no parameter selection. All64 outputs sealed before independentCPUofficiallabels. Original Source/E/Global6/eightobservations/space stayfixed.

| Direction | Representation | tIoU (%) | Complete vIoU (%) | Δt vs current(pp) | Δv vs current(pp) |
|---|---|---:|---:|---:|---:|
|hc2|Source|46.0910|22.6217|+0.0633|-1.9646|
|hc2|NativeG6|44.0627|23.6196|-1.9650|-0.9668|
|hc2|CurrentRegion|46.0277|24.5863|+0.0000|+0.0000|
|hc2|Native2D|48.8882|25.8101|+2.8605|+1.2237|
|hc2|Native3D|47.7062|25.5246|+1.6785|+0.9383|
|hc2|Fused2D|46.1648|24.6357|+0.1371|+0.0494|
|vidstg|Source|41.9852|18.6846|-3.4325|-3.3295|
|vidstg|NativeG6|45.4902|21.7566|+0.0725|-0.2576|
|vidstg|CurrentRegion|45.4176|22.0141|+0.0000|+0.0000|
|vidstg|Native2D|45.3198|21.5424|-0.0979|-0.4717|
|vidstg|Native3D|44.8771|21.4727|-0.5405|-0.5414|
|vidstg|Fused2D|46.2308|22.3879|+0.8132|+0.3738|

Native2D is original input_proj(ResNet appearance) grid BEFORE text/vision fusion. Native3D/Fused2D are explanatorycontrols, not candidates automatically promoted. CurrentRegion is original fused motion memory, restored bitwise. No new externalencoder/teacher/observation/training. All4views collected in the same one nativeSource forward/query; newSource64 recorded as research cost, not deployment latency.

The generic algorithm does not need motion or a class vocabulary. This experiment evaluates one concrete TA pre-fusion2D choice; it does not prove every spatially-addressable representation will transfer or improve event recognition. OnlyTA tested; noMLLM/crossbackbone empiricalclaim.

See report.json for all pairedparent-bootstrapCIs, group/boundary/length/severe/fixedspace values and signed.tsv/severe.tsv for every sign. No statisticalequivalence/adoption gate is invented after outcomes.

## Measured interpretation

Native pre-fusion2D does not show a broad mean collapse on these fixed exposed32/32
panels. Relative to the frozen current postfusion-motion readout: HC t+2.8605pp
(CI[1.4081,4.4425]) and completev+1.2237pp(CI[.2984,2.2783]); Vid t-.0979pp
(CI[-2.1563,1.7093]) and completev-.4717pp(CI[-1.5895,.4533]). These are paired
feasibility estimates, not demonstrated statisticalequivalence or untouchedtest.
HC time18wins/5harms/9ties, v15/6/11. Vid time10/10/12, v9/11/12.

The Vid mean hides two substantial time+v failures relative to CurrentRegion:
RECHECK014 t-18.2609/v-10.8734pp and RECHECK013 t-15.7928/v-9.7728pp. Relative
Global6, severe>5ppt time4->5 and completev0->3. HC severeGlobal6 time2->1 andv2->1.
Thus not every decline is small. Fixedspacev falls.3724pp onVid versus fullv.4717pp,
consistent with interval changes carrying most of the observed loss; frozen spatial
parameters are unchanged. This is output decomposition, not proved event semantics.

Native2D beats Source mean t/v by HC+2.7972/+3.1884 and Vid+3.3346/+2.8578pp.
Source-relative short/long t deltas: HC+2.6108/+2.8402(n6/26), Vid+.4027/+6.2665
(n16/16). The six-query HCshort group is especially small; these are descriptive.

Prefusion3D also improves HC means, while postfusion2D staysclose to CurrentRegion.
OnVid, postfusion2D is stronger than prefusion2D. Fusion context/feature choice can
matter in different directions; these controls do not establish a universal best
layer/stream and are NOT promoted after looking at outcomes. Correct2Dregion does
not reliably beat samebudgetreflectedregion(CIscross0), so location-specific event
causation remains unestablished. Wrongtime is much worse inbothpanels.

The supported framework statement is native spatially-addressable local content
feedback with model-specific coordinate/time adapters, independent of a required
class vocabulary, independent motion branch or mandatory textfusion. Current TA
full remains on its EXACT selected postfusion-motion method; no automatic swap or
retuning follows this bounded comparison. Otherbackbone/MLLM effectiveness untested.

Completed64originalSource reads took298.61s plus7.36s independentCPUscore. The failed
v1 added ONE abortedSource read before zero savedpredictions; keep its overhead and
failure separately. No newexpert/observation/optimizer/networktraining or external
encoder. Four representations were recorded during the SAMESource forward/query;
research acquisition time is not deployment latency or encoder speedup.
