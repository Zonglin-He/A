# ReferDINO native8 capability32 and cost gate

Completed/measurement valid. Original NH16/NV16 exposed development parents, one query each; not independent confirmation. One official ryt_mevis_swinb frozen checkpoint. Original TA-STVG Source/E/finalMBR unchanged. Full original queries; no DINO unresolved-what skip. Original4+4 deterministic physical coverage positions locked before inference/GT. Native infer/segmentation decoder and same-query mean-score track selection unchanged; no OPD/training/threshold sweep.

## Cost gate

Four pilots NH001,NH002,NV001,NV002 all originally execute DINO4. Both models one duplicate-input warmup then measured once each query. Shared raw RGB decoding/model loading excluded; preprocessing/H2D/full forward including native mask decoder/track selection/box CPU readout included. FP32, native360 Refer input versus original800/1333 DINO input. No mask upsampling/video visualization, since output task is boxes.

Corrected pooled Refer8/DINO4 expert-time ratio **0.431514** <=1.5: pass. Original audit=True DINO timing gave0.363997; audit hashes were inside its model prehook but outside Refer timer. Preserve original timing; rerun same four T queries audit=False, no Refer/prediction changes, no quality selection. Additional20 T calls counted. This correction followed global output sealing; original gate had allowed28 additional queries, corrected result independently remains pass.

| Pilot | DINO4 expert s (without audit hashes) | Refer8 expert s | ratio | Refer peak allocated GiB |
|---|---:|---:|---:|---:|
|NH001|0.434268|0.169793|0.39099|1.7864|
|NH002|0.452757|0.173135|0.38240|1.8557|
|NV001|0.485669|0.249974|0.51470|2.1096|
|NV002|0.484391|0.208456|0.43035|2.1096|

All completed queries Refer expert total6.381041s, forward5.649813s; max allocated2.5260GiB. Aggregate GPU allocation upper bound79.995144s includes main66.086555+timing correction13.908589, both with3s guards/load/decode/hash costs. Actual Refer32+1 warmup query forward, T40 total image forwards, Source/OPD/training0. Setup and historical cache costs excluded from latency comparisons; runtime build required missing Ninja and linker-space fixes, original external code unchanged.

## Three frozen readouts

E_MBR is original baseline. Ref_E_MBR uses only native selected singleton boxes at original4, recomputes unchanged source-calibrated E from the same Source with no new confidence filter. Ref_direct8_MBR linearly interpolates the native selected common track8 in physical xyxy time; endpoint hold, same final MBR. Interpolation is not independent per-frame visual supervision. All32 three outputs sealed before new GT.

| Absolute (%) | HC vIoU | Vid vIoU | HC original4-excluded actual sIoU | Vid original4-excluded actual sIoU |
|---|---:|---:|---:|---:|
|E_MBR|35.30548|15.75593|56.66036|45.44847|
|Ref_E_MBR|36.34060|12.64721|56.91843|42.79957|
|Ref_direct8_MBR|37.55630|14.81362|55.08722|47.95761|

Differences in percentage points, query mean then parent mean; paired10000 bootstrap seed20261018. All16 parents in primary full/actual-node metrics, unknowns not fabricated.

| Difference | HC | Vid |
|---|---:|---:|
|Ref_E_MBR − E, v|+1.0351 [-5.8442, +8.3894], n=16|-3.1087 [-12.9961, +4.3853], n=16|
|Ref_E_MBR − E, unobserved_actual_s|+0.2581 [-10.6068, +11.8082], n=16|-2.6489 [-19.4162, +16.0745], n=16|
|Ref_direct8_MBR − E, v|+2.2508 [-4.7166, +11.3331], n=16|-0.9423 [-10.1127, +6.1845], n=16|
|Ref_direct8_MBR − E, unobserved_actual_s|-1.5731 [-13.3896, +12.4941], n=16|+2.5091 [-15.5706, +23.2927], n=16|

All temporal outputs exact unchanged. Main intervals cross0; no reliable2–3pp complete improvement, no dual-positive near5pp capability criterion. HC direct has4 improve/12 worsen; Vid6 improve/7 worsen/3 unchanged. This is a capability/readout study, not an OPD failure.

## Actual selected anchors and transfer

| Diagnostic | HC | Vid |
|---|---:|---:|
|actual teacher − E at all8|+13.7636 [+1.3189, +28.5471], n=16|+12.2824 [-6.4149, +33.7452], n=14|
|actual teacher − E at original4|+18.3860 [+3.4085, +35.6135], n=15|+13.9235 [-5.7880, +38.0507], n=12|
|actual teacher − E at additional4|+2.6114 [-14.4546, +25.9901], n=5|+6.1593 [-24.1024, +42.4585], n=7|
|E propagated − same teacher at original4|-15.9878 [-27.9176, -5.8401], n=15|-9.4852 [-18.1169, -2.3403], n=12|

HC original4 local advantage is measured, but E preserves much less; direct interpolation also does not improve outside-original4 actual nodes. These are distinct effects, not proof of a single cause. Added4 known support only5 HC/7 Vid parents; unannotated observations remain unknown.

| Conditional actual-node subset, direct − E | HC | Vid |
|---|---:|---:|
|outside8|-1.7709 [-13.5811, +12.3101], n=16|+1.6997 [-16.5867, +22.4906], n=16|
|far_center_E|+28.1174 [+11.1612, +47.1802], n=8|+19.8294 [-6.6974, +53.4229], n=8|
|scale_E|+22.5714 [-1.0746, +54.8775], n=4|+5.5164 [-13.2754, +35.1549], n=7|
|E_good|-14.6408 [-20.5998, -9.4275], n=14|-14.4590 [-35.8451, +2.6711], n=9|
|E_very_bad|+27.1825 [+6.7204, +51.0306], n=7|+34.9634 [-3.5275, +73.4543], n=5|

Far center means any axis error>.5 GT dimension; scale ratio>2 or<.5; E_good IoU≥.5 and E_very_bad<.1. GT defines only post-seal explanation subsets. These supports overlap and differ, never replace the main denominator. Box IoU/large shifts cannot identify wrong identity versus severe geometry independently; no identity-ID recovery percentage is claimed.

## Five formerly skipped what queries

All five now execute native Refer8 full-query inference, not evidence they understand questions.

| Readout − E | full vIoU pp | original4-excluded actual sIoU pp |
|---|---:|---:|
|Ref_E_MBR|-4.7447 [-10.2269, +0.8996], n=5|-10.5805 [-19.4696, -0.0167], n=5|
|Ref_direct8_MBR|+0.1332 [-6.3981, +6.5991], n=5|-3.5053 [-16.1288, +9.1181], n=5|

## Complete signed per-parent results

| Parent | Ref4→E v pp | Direct8 v pp | Ref4→E actual s pp | Direct8 actual s pp | native expert s | peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
|NH001|+0.88545|+4.53474|-4.80954|-2.04770|0.169793|1.7864|
|NH002|+1.04027|-0.76597|+1.95352|-16.46710|0.173135|1.8557|
|NH003|+1.07001|-2.76946|+1.90937|-8.73218|0.236281|2.1096|
|NH004|+3.82684|+1.09337|+3.19064|-0.38683|0.323061|2.5260|
|NH005|+0.03778|-1.00414|+0.22632|-11.58802|0.174831|1.7862|
|NH006|+26.42372|+41.86135|+38.67340|+49.95415|0.265729|2.2784|
|NH007|-26.80010|-12.92238|-49.51111|-46.44624|0.166951|1.7862|
|NH008|+0.38518|-1.11283|+1.85557|-5.91513|0.171026|1.7792|
|NH009|-1.58365|-4.35698|-17.61671|-22.98620|0.159211|1.7864|
|NH010|+1.63257|-6.71439|+1.39405|-7.87597|0.206308|1.9399|
|NH011|-0.12551|-4.46288|-1.97109|-0.02861|0.174964|1.7862|
|NH012|+37.71515|+46.48960|+59.75338|+74.00263|0.170488|1.7861|
|NH013|-21.83572|-14.65468|-24.44980|-16.21505|0.164587|1.8557|
|NH014|-3.49189|-4.50098|-4.05990|-5.83952|0.179430|1.7861|
|NH015|-0.32252|-4.03598|-0.22207|-6.36945|0.180037|1.7791|
|NH016|-2.29556|-0.66528|-2.18690|+1.77095|0.175690|1.7861|
|NV001|+0.00000|+0.00000|-52.36218|-52.86444|0.249974|2.1096|
|NV002|+0.00000|+0.00000|-1.48923|-10.29378|0.208456|2.1096|
|NV003|-2.83490|-2.51932|-4.87129|-3.91202|0.182788|1.7861|
|NV004|+0.00000|+0.00000|-7.71818|+14.10084|0.211299|2.1096|
|NV005|-64.07155|-57.11065|-72.07534|-68.37377|0.173869|1.7861|
|NV006|+4.32270|+4.48782|+9.20581|+9.15795|0.219734|1.7861|
|NV007|-3.68448|-1.39345|-14.51382|-6.16476|0.233882|2.1096|
|NV008|-0.59961|+0.46572|-0.43131|+1.86181|0.209719|1.7861|
|NV009|-11.35727|+11.07887|-17.12580|+15.59342|0.202345|1.7861|
|NV010|-0.04211|+0.36314|-0.17336|+1.58843|0.173286|1.7861|
|NV011|+14.42125|+17.57880|+75.94348|+93.21485|0.208278|2.1096|
|NV012|+25.45541|+25.78831|+80.65778|+89.94503|0.159346|1.7861|
|NV013|-1.06142|-1.06142|-22.82952|-22.82952|0.230487|2.1096|
|NV014|-0.26360|-0.75834|-1.41569|-3.55193|0.229921|2.1096|
|NV015|-12.79260|-11.31974|-17.28163|-15.53651|0.171411|1.7861|
|NV016|+2.76865|-0.67680|+4.09780|-1.78949|0.224727|2.1086|

Positive large cases NH006/NH012 and NV011/NV012 kept alongside NH007/NH013/NV005/NV015 severe losses. HC direct mean depends strongly on NH006/NH012. See POST_SEAL_ANALYSIS for leave-one-largest sensitivity without deleting any main sample.

## Decision and verification

Cost passes; whole-trajectory quality criterion does not. Do not promote the teacher or start a new OPD/threshold/observation sweep. Preserve HC local teacher advantage, E transfer loss, sparse interpolation limits and good-E damage separately; one native8 configuration does not disprove all ReferDINO or OPD. Any later interface research requires a new explicit mechanism/protocol and matched direct/fixed/live comparison, not attributing this teacher change to OPD.

Root independent audit checks32 native choices/32 position rules/108 original RGB hashes/32 E constructions/32 interpolation reconstructions/288 metrics/192 actual-node metrics/101 known teacher anchors/36 bootstrap contrasts. No independent second GPU decoder replay; source-level native selection plus saved score argmax and selected-time-score consistency checked. Original native source checkout clean, all learned checkpoint keys covered, only nonpersistent bert.position_ids unexpected, CUDA kernel synthetic-reference check passes. Both owned units exit0/inactive; registries unchanged.

Official sources: https://github.com/iSEE-Laboratory/ReferDINO and https://huggingface.co/liangtm/referdino . Code/protocol/config/full signed and unknown scalar results/cost correction/audit published; raw original text/IDs/media/GT or candidate coordinates/weights/tensors excluded.
