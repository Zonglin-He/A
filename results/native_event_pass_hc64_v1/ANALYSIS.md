# Actual native first-pass and refined second-pass temporal endpoint reading

Official recipe: training calls positive iteration/max_iter; extra native refinement occurs only with negative argument; evaluation omits argument and uses negative default. This is code evidence, not private checkpoint training-history proof. Hypothesis: target-domain refinement can damage trained first-pass boundary evidence. This study reads native endpoint distributions before/after refinement, with no NEW foreground/actionness feedback or threshold changes.

Actual original FP16 Full fusion and FP32 two-pass source replay, original cached Full text/visual values. Non-replacing observer captures existing first/second hidden at0/1 then2/3 for two offsets; hook removed even on failure. Each pass six-layer original time head, normalized native offset hull perlayer then mean. Primary fixed halfFirst6+halfSecond6, no cross-pass hull. Source/refined-second64 hidden/logits/q/boxes bitwise. All64 primary, no grammar/quality/GT exclusions, original spatial geometry/evidence8/DP. All64 sealed before independent CPU labels.

Same-information DirectFirst6, time-reflected first, exact second self-mixture, all64native selfhull expansion, archived52selfhull/12native reference, Source/Native/nochange controls.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|RefinementMixture|29.7955|56.1842|6/4/54|0|
|DirectFirst6|29.7129|56.0588|12/11/41|3|
|ReversedFirstMixture|24.1097|44.0806|22/33/9|27|
|SecondSelfMixture|29.6952|55.9613|0/0/64|0|
|NativeSelfHull|30.4811|57.4826|39/23/2|10|
|ArchivedSelfHull|30.5252|57.4780|33/18/13|7|
|NoChange|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−G6 t+0.2229pp CI[-0.018979876157741292, 0.5073308510885418], v+0.1003pp CI[-0.030745852180273942, 0.24915975667697823]; Source t+1.3612, v+2.9017pp.
Primary Source shortt-0.0299pp; longt+1.7158pp.
Primary−DirectFirst6 t+0.1254pp CI[-0.3273318155072445, 0.5919674777419263].
Primary−ReversedFirstMixture t+12.1036pp CI[5.841087653382962, 18.851882481405635].
Primary−SecondSelfMixture t+0.2229pp CI[-0.018979876157741292, 0.5073308510885418].
Primary−NativeSelfHull t-1.2984pp CI[-2.7250679889974396, 0.17788463335603005].
Primary−ArchivedSelfHull t-1.2938pp CI[-2.6264409662884383, 0.10575995341309222].

Not qualified; no pass-weight/layer/gate scan or posthoc promotion of DirectFirst6.
Worker24.88s inclloading/research recording; calls{'new_text_encoder': 0, 'fusion_offsets': 128, 'native_replay_queries': 64, 'original_decoder_calls': 256, 'original_head_six_layer_calls': 128, 'extra_head_six_layer_calls': 256, 'new_decoder_executions_beyond_original': 0, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. First hidden already computed in original deployment. Two additional first-pass head/posterior calls can expose this signal; actual deployment cost NOT measured here. Research repeats second head for no-op validation and must not claim its runtime as deployment speed. Official/dense checks1152 maxerror6.66e-16.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [Source groups](source_duration_groups.json), [protocol](protocol.json).
