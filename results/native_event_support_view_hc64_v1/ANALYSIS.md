# Positive target-preserving feature view before native fusion

Exposed HC64. Full original caption and full physicaltime axis retained. Original complete-boxDP auxiliaryfulltime path, extent1.5context, valid cellswithanyboxoverlap. Keep original2D/3D values inside; fill outside valid cellswith perframe inside mean FP32castbackFP16; exclude outside attentionkeys BEFORE originalcrossmodal fusion. Originalpadding unchanged. EmptyvisibleROI frame fallbackwholevalidgrid recorded,noqueryrejection. NoRGB/visualencoder/text/expert/observations/optimizer. Noqueryinjection or foregroundscoresasfeedback.

Nativeencoder safetycorner0,0 remains unmasked by ORIGINALsource; if outsideROI itsfeaturealready insidemean. Initialtarget/controlkeepcounts equal byreflection, actualcounts maydifferby1 aftersafetycorner. No frozen source/hash modification. Original motionfeaturesretain fullclipcontext; viewnotpixelzoom,newdetail,purecausalisolation orphotorealisticcropping. Difference oldTubeReplay: actualcontext BEFOREfusion/queryconstruction instead ofdecoderattentionprior.

ActualFull/SupportBoth/ReflectedBoth/SupportMaskOnly/SupportFillOnly reads, originalFP16fusion/FP32native2passes/6head/twooffsethull. Full64hidden/logits/q/Sourceboxesbitwise. Primary fixed halfglobal+halfsupport, noinverse-nullratio. SameinformationDirectSupport, reflectedfusion/direct, mask-only/fill-onlycomponentcontrols, nativeself/source/noop. All64active;all64sealbefore independentCPU labels. Actualfinalspace originalsamegeometry/DP/evidence8; fixedspace separatelyreported.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|SupportViewFusion|29.4496|55.2700|30/26/8|16|
|DirectSupportView|26.1941|49.1827|24/39/1|31|
|ReflectedViewFusion|28.7024|53.8812|22/32/10|22|
|DirectReflectedView|24.5415|46.4622|25/38/1|32|
|MaskOnlyFusion|30.2082|56.0583|26/29/9|13|
|FillOnlyFusion|30.1406|56.5246|31/25/8|14|
|NativeSelfMixture|29.6952|55.9613|0/0/64|0|
|NoView|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−G6 t-0.6913pp CI[-3.9418071816743745, 2.0273039442961047], v-0.2456pp CI[-1.9326321672222493, 1.2469886579376768]; Source t+0.4470, v+2.5558pp.
Primary Source shortt+1.6448pp; longt+0.1417pp.
Primary−DirectSupportView t+6.0873pp CI[3.156194797050177, 9.137079393972812].
Primary−ReflectedViewFusion t+1.3888pp CI[-2.2401236726021265, 4.68934568985214].
Primary−DirectReflectedView t+8.8078pp CI[3.647866621403707, 13.891160820139367].
Primary−MaskOnlyFusion t-0.7883pp CI[-2.3233266109452737, 0.6297110595051714].
Primary−FillOnlyFusion t-1.2546pp CI[-3.187012903742528, 0.5482086488811799].
Primary−NativeSelfMixture t-0.6913pp CI[-3.9418071816743745, 2.0273039442961047].

Not qualified; no context/fusionweightscan orposthoc controlpromotion.
Worker70.80s inclloading/researchrecording; calls{'fusion_offsets': 640, 'native_replay_queries': 320, 'new_text_encoder': 0, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. Deploymentwouldaddone sourcefeaturefusion/readoutview, butactualdeploymentcostnotmeasured here. Study5reads notend-to-end deploymenttiming. Official/densechecks1280 maxerror8.88e-16.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [Source groups](source_duration_groups.json), [protocol](protocol.json).
