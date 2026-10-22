# Native event decoder with joint offset context

Fixed exposed HC64, all64 queries. Interleave original Full-caption actual final-pass7 kwargs into65 physical nodes; each query/content/time/memory/position/padding value is unchanged. No additional encoder/text/fusion/expert/observation/optimizer. Six-layer original frozen time decoder/head; single legal interval posterior per layer, mean6, fixed half Global6/half joint; original MBR and spatial DP.

Native decoder previously performs temporal self-attention separately for each offset. This test permits cross-offset interaction. Same-grid BlockedOffsetFusion masks ONLY cross-parity self-attention edges, keeping all inputs and legal boundary support. It isolates temporal interaction from changed pooling and cross-parity endpoint hypotheses. Existing motion features already contain their original clip context; no new independent visual evidence is created.

Original time positions remain unchanged (adjacent parity nodes can share original slot). This is an intervention, not bitwise equivalent acceleration. Blocked and separate native calls have different kernel packing; actual hidden error is reported. Original separate native hidden/logits/posterior and saved Source/Native/spatial aliases must still match bitwise64.

NativeMarginalFusion averages original two legal offset tables without hull. SelfHullFusion is explicitly the previous grammar-supported52/64 alias, not the main all64 matched control. ReverseJointFusion reverses per-frame memory/position/padding relative to unchanged query/time. Source/NativeG6/NoOp retained. All64 predictions sealed before independent official-label CPU scoring.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|JointOffsetFusion|28.9036|53.9931|14/42/8|17|
|JointDirect|25.8296|48.3870|10/53/1|41|
|BlockedOffsetFusion|29.2333|54.5753|16/37/11|13|
|NativeMarginalFusion|29.1492|54.5192|14/39/11|12|
|SelfHullFusion|30.5252|57.4780|33/18/13|7|
|ReverseJointFusion|29.2367|54.1174|22/24/18|18|
|NoOp|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−Global6: v-0.7916pp CI[-1.3642668693193256, -0.22810425140258844], t-1.9682pp CI[-3.0049270741368996, -0.9448592207464904]. Primary−Source: v+2.0098pp, t-0.8299pp.
Joint−identical blocked-grid: v-0.3297pp CI[-0.7116605019772164, -0.04099788665457908], t-0.5821pp CI[-1.0490003985274652, -0.20104323837714966].
Primary shortt-0.3571pp,longt-2.3789pp vsG6; vsSource short-0.5105,long-0.9113pp.

Not qualified for main/full promotion. No position, density or fusion-weight sweep.
Actual blocked-vs-original native hidden maximum difference2.38419e-06; original separate no-op remains bitwise. Official/dense checks1152 maxerror9.99e-16. Worker21.36s including loading/research recording, not deployment end-to-end. Calls{'native_decoder_calls': 320, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_expert': 0, 'new_observations': 0, 'new_optimizer': 0}.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [all Source duration groups](source_duration_groups.json), [protocol](protocol.json).
