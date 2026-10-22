# Actual native spatial-by-event-text interaction

Exposed HC64. Original full-time complete-boxDP defines auxiliary target trajectory; full spatial output remains unchanged. Fractional complete-box/cell coverage replaces BOTH2D/3D features at that location with same-frame valid outside-footprint mean (FP32 thenFP16); padding unchanged. Control180reflects the exact same weights within validrectangle, preserves area but may overlap target. Full valid mean fallback recorded for fully covered frames, no sample rejection. Feature intervention is not photorealistic object removal or a proved causal effect. Original motion features retain originalclipcontext.

Actual six native reads perquery: Full0/Identity0 reproduce archived text-conditioned hidden/logits/q/Source boxes64bitwise, then FullTarget/IdentityTarget/FullControl/IdentityControl. Same frozen original FP16fusion/FP32two-pass/timehead6/native offsethull. Event-vs-identity difference of spatial responses, corrected by matchedcontrol; positive q0*exp(interaction), coefficient1/no epsfloor/gain/GTgate. All64 no exclusions. Original physicalMBR/complete-boxDP andexpert8 unchanged. All64seal before independentCPU labels.

Same-information DirectInteraction; uncorrected target interaction; spatial-only/identity-only/text-only; native sharpening; time-reflected interaction; Source/Native/noop. This differs from prior pre-MLP TTS erasure, global temporal mean null and text-only ratios; no new query/optimizer/teacher. Not a reproduction of trained D-TSG/CAEM.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|InteractionResidual|26.1684|50.6715|25/36/3|26|
|DirectInteraction|17.1369|34.3938|16/48/0|46|
|TargetInteraction|25.5172|48.8161|23/41/0|32|
|SpatialOnly|29.0199|53.5893|30/32/2|27|
|IdentityOnly|27.3429|49.2172|27/34/3|28|
|TextOnly|23.6563|45.3587|13/49/2|40|
|Sharpen|28.7258|53.7518|16/38/10|20|
|ReverseInteraction|24.6624|47.7843|28/34/2|28|
|NoInteraction|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−G6 t-5.2898pp CI[-9.603550525918468, -1.1130774048331185], v-3.5268pp CI[-6.487066841512405, -0.7792847996300879]; Source t-4.1515, v-0.7254pp.
Primary Source shortt-4.0571pp; longt-4.1756pp.
Primary−DirectInteraction t+16.2777pp CI[9.946267922683324, 23.052830116533375].
Primary−TargetInteraction t+1.8554pp CI[-2.153282046304673, 5.857039799354572].
Primary−SpatialOnly t-2.9178pp CI[-7.4586601527573455, 1.7038638557027646].
Primary−IdentityOnly t+1.4543pp CI[-5.631823996563048, 8.440830886760882].
Primary−TextOnly t+5.3128pp CI[0.2501631487860462, 10.142640762181129].
Primary−Sharpen t-3.0803pp CI[-7.452346436127831, 1.1406297298925268].
Primary−ReverseInteraction t+2.8871pp CI[-4.311395673542682, 10.043259984661475].

Not qualified; no spatialmask/gain/ratio scan or posthoc controlpromotion.
Worker81.36s inclloading/research recording; calls{'text_forwards': 64, 'fusion_offsets': 768, 'native_replay_queries': 384, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. Research cost not deployment end-to-end timing. Official/dense checks1408 maxerror1.11e-15.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [Source groups](source_duration_groups.json), [protocol](protocol.json).
