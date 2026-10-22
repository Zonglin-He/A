# Same query optimizer, new target-semantic feedback

Earlier RawQuery and ClauseQuery results stay negative; this does not propose a new query interface or more steps. Direct target-semantic posterior now improves HC64t+4.315/v+1.898pp with36wins14harms/two severe time harms and positive Source short+long. Its fixedVid32 still harms more queries and short events. This study tests whether fitting that better measured feedback through the frozen full-caption native time model adds useful correction, not whether the pseudo-loss can decrease.

EXACT historical disposable shared-offset u256, source-query RMS scaling, Adam lr.01,10steps, normalizedRMS radius.1, laststate. All128 queries execute ALLthree optimizer arms and ALL10steps. No beststep, confidence/GT veto, fewer updates or new learned weights. Original timeDecoder7kwargs/strides preserved. Differentiable six-layer two-offset oldhull posterior versus same sealed target; zero-state native math restored, original no-adaptation hidden/logit/posterior bitwise. All Source weights frozen; only temporary query state optimized.

Same-information DirectTarget exactparentalias; ReflectedQuery fits its matched-location teacher with identical optimizer and DirectReflected; GlobalSemanticQuery samebudget onparentglobal semantics andDirectGlobal. Original source/native/no-update controls. No new teacher, RGB/visual/text/fusion/ASA computation. Native model family may fit wrong feedback; measure actual task/harms and advantage over identical direct feedback.

All128 predictions seal before independent CPU officiallabels. Panels are exposed development evidence, no untouched-test claim. Five reused engineering checks cover exact hull, finite-difference gradient, no-information no-op, score consistency and expanded input layout.

|Panel|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---|---:|---:|---:|---:|
|HC64|NativeG6|29.6952|55.9613|—|—|
|HC64|TargetQuery|29.8774|56.3663|20/9/35|1|
|HC64|DirectTarget|31.5936|60.2765|36/14/14|2|
|HC64|ReflectedQuery|29.8711|56.2660|16/7/41|1|
|HC64|DirectReflected|30.4808|58.7320|33/19/12|6|
|HC64|GlobalSemanticQuery|29.7052|56.1564|16/11/37|0|
|HC64|DirectGlobal|30.9658|58.8883|31/17/16|6|
|HC64|NoUpdate|29.6952|55.9613|0/0/64|0|
|HC64|Source|26.8938|54.8230|—|—|

HC64: primary−G6 t+0.4050pp CI[-0.015319211229469214, 0.8277001648582994]; v+0.1822pp CI[-0.01285030559035767, 0.3876980082311646]. Primary−Source t+1.5433/v+2.9836pp.
Source short-0.3886/long+2.0358tpp; fixed screen=False.
Primary−DirectTarget: t-3.9102pp CI[-6.0854594510593705, -2.0316145678206183].
Primary−ReflectedQuery: t+0.1003pp CI[-0.2207437076720565, 0.4453938910769019].
Primary−GlobalSemanticQuery: t+0.2099pp CI[-0.15435925544647325, 0.5760218987777473].
|hc2_recheck|NativeG6|27.8907|51.6010|—|—|
|hc2_recheck|TargetQuery|28.0598|51.6823|8/4/20|1|
|hc2_recheck|DirectTarget|27.8940|52.0803|14/11/7|5|
|hc2_recheck|ReflectedQuery|27.9352|51.3641|9/5/18|3|
|hc2_recheck|DirectReflected|28.1065|52.7487|15/9/8|7|
|hc2_recheck|GlobalSemanticQuery|27.8086|51.6165|7/5/20|1|
|hc2_recheck|DirectGlobal|27.3123|51.9587|15/12/5|5|
|hc2_recheck|NoUpdate|27.8907|51.6010|0/0/32|0|
|hc2_recheck|Source|23.4655|50.4639|—|—|

hc2_recheck: primary−G6 t+0.0813pp CI[-0.8971616514819365, 0.8758476493586235]; v+0.1690pp CI[-0.18897102740678579, 0.5341938877529622]. Primary−Source t+1.2184/v+4.5943pp.
Source short+2.2224/long+0.6925tpp; fixed screen=False.
Primary−DirectTarget: t-0.3980pp CI[-2.6556424842941473, 1.8797965909765462].
Primary−ReflectedQuery: t+0.3182pp CI[-0.30164261016431504, 1.0397020145761615].
Primary−GlobalSemanticQuery: t+0.0659pp CI[-0.9907670026404656, 0.997246831086166].
|vidstg_recheck|NativeG6|23.9576|35.6554|—|—|
|vidstg_recheck|TargetQuery|23.7767|35.4307|4/9/19|0|
|vidstg_recheck|DirectTarget|25.3553|36.3389|9/16/7|6|
|vidstg_recheck|ReflectedQuery|24.2671|36.0469|11/6/15|0|
|vidstg_recheck|DirectReflected|25.2854|36.3672|11/11/10|6|
|vidstg_recheck|GlobalSemanticQuery|23.8149|35.5330|6/7/19|0|
|vidstg_recheck|DirectGlobal|23.6888|35.0400|9/16/7|7|
|vidstg_recheck|NoUpdate|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|Source|15.6772|34.0110|—|—|

vidstg_recheck: primary−G6 t-0.2247pp CI[-0.48930167784388157, 0.015994762048133562]; v-0.1809pp CI[-0.3923170407641088, -0.00015189856425192234]. Primary−Source t+1.4196/v+8.0995pp.
Source short+0.6393/long+4.2066tpp; fixed screen=False.
Primary−DirectTarget: t-0.9082pp CI[-4.863932990201454, 1.9754455024558844].
Primary−ReflectedQuery: t-0.6162pp CI[-1.2589298506289373, -0.10882576332965412].
Primary−GlobalSemanticQuery: t-0.1023pp CI[-0.3357892205210128, 0.11467837527751115].

Not qualified. Do not scan query steps/lr/radius or promote a control; loss decrease alone is not task improvement.
Actualcalls{'time_decoder_calls': 8704, 'optimizer_steps': 3840, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_ASA_head': 0, 'new_expert': 0, 'new_observations': 0, 'new_network_training': 0}; worker122.12s includes loads/research recording/3optimizerarms, not deploymentlatency. Added query optimization is counted, not described as zero computation. Official/densechecks2304maxerror1.67e-15; differentiableposterior maxerror1.97e-16.

[All signed results](signed.tsv), [all severe declines](severe_declines.tsv), [protocol](protocol.json), [same-information teacher study](../native_event_target_semantic_v1/ANALYSIS.md).

CPU scoring engineering correction: original scorer carried stale comparison names from the previous study and failed after the valid sealed inference. Original logs/code/partial score progress preserved privately. The independent [score adapter](../../research/target_query_score_adapter.py) changes ONLY that literal qualification comparison list to the exact ReflectedQuery/GlobalSemanticQuery required by the frozen plan; AST of all remaining code, metric functions, data and inference unchanged. All128 prediction/trace hashes verified before and after; no model or inference rerun. Use this adapter for scoring the immutable study. [Adapter proof](SCORING_ADAPTER.json).
