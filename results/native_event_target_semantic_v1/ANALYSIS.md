# Read native event-semantic coordinates on propagated target support

Original global verb-class projection improvedHC64 but failed fixedHC32/Vid32. This study tests a distinct input interface: condition only the extra native ASA word-class head on fixed fulltime complete-box target context. Original fullcaption, fusedfeatures, q0, native query/decoder, Source/ERef, all65 nodes and final spatial DP stay unchanged. Learned word-class logits are observed before frame averaging, not new actionness/foreground feedback.

Existing1.5context and equalarea180reflected support; wrongtime uses reversed65-node auxiliarytrajectory positions without reversing motionfeatures. Original3D still contains clipcontext and fusedfeatures contain global context, so this is not a pure local causal intervention. Controls may still cover the true target; overlaps and empty-ROI wholevalid fallbacks recorded, no query rejected. No context/strength/iteration scan. SourceTimeDecoder feature/mask ordering audit finds allzero paddingmask in all128, so that code-order concern is excluded here without a model experiment.

Original ASA BertSelfAttention clones unmasked att_map before dropout. A scoped dropout prehook restricts softmax KEY probabilities and renormalizes before value aggregation; the returned prehook att_map clone is UNMASKED and unused as event feedback. No provider file/weight edit. AllKeysNoOp returns original tensor object; all128 global semantic p and64 originalu restore bitwise, all128 AllKeysNoOp classp identical. Hooks finally removed. Parent original Bernoulli2-round crossfit/q0 anchored loop is reused EXACTLY, no OAS/gain/step change. Direct uses SAMEsecond-round evidence; random coordinates preserve exactperclass target-p histogram; reflected/wrongtime andparentglobal/sharpen controls.

Five CPU tests: masked-softmax equivalence; allkeys identity; fixedcontext/reflection/empty fallback; wrongtime indexing; scopedhook cleanup on exception. All128 predictions seal before independent CPU fixed officiallabels; exposeddevelopment panels.

|Panel|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---|---:|---:|---:|---:|
|HC64|NativeG6|29.6952|55.9613|—|—|
|HC64|TargetSemanticLoop2|31.5936|60.2765|36/14/14|2|
|HC64|OneStepTarget|31.4161|59.6997|35/15/14|1|
|HC64|DirectTargetEvidence|28.7151|55.6174|28/36/0|23|
|HC64|ReflectedSemanticLoop2|30.4808|58.7320|33/19/12|6|
|HC64|WrongTimeSemanticLoop2|30.7815|58.3553|28/23/13|6|
|HC64|RandomTargetSemanticLoop2|29.2556|55.5720|22/25/17|12|
|HC64|ParentGlobalSemanticLoop2|30.9658|58.8883|31/17/16|6|
|HC64|SharpenOriginal|28.7258|53.7518|16/38/10|20|
|HC64|AllKeysNoOp|29.6952|55.9613|0/0/64|0|
|HC64|Source|26.8938|54.8230|—|—|

HC64: primary−G6 t+4.3152pp CI[2.2408790456130925, 6.661729194257984]; v+1.8984pp CI[0.8137197209048007, 3.1319413696912437]. Primary−Source t+5.4535/v+4.6998pp.
Source short+2.2310/long+6.2749tpp; fixed screen=True.
Primary−ParentGlobalSemanticLoop2 t+1.3882pp CI[0.13451187299998707, 2.8141974012960818].
Primary−ReflectedSemanticLoop2 t+1.5445pp CI[-0.2642251582335199, 3.5090847544196007].
Primary−WrongTimeSemanticLoop2 t+1.9212pp CI[0.3930795858833003, 3.6149946523307572].
Primary−RandomTargetSemanticLoop2 t+4.7045pp CI[2.4702478813053723, 7.259318878185991].
|hc2_recheck|NativeG6|27.8907|51.6010|—|—|
|hc2_recheck|TargetSemanticLoop2|27.8940|52.0803|14/11/7|5|
|hc2_recheck|OneStepTarget|28.2058|52.2307|16/10/6|5|
|hc2_recheck|DirectTargetEvidence|26.0025|46.5307|14/17/1|16|
|hc2_recheck|ReflectedSemanticLoop2|28.1065|52.7487|15/9/8|7|
|hc2_recheck|WrongTimeSemanticLoop2|27.3459|50.7011|14/12/6|7|
|hc2_recheck|RandomTargetSemanticLoop2|28.9568|52.4794|12/10/10|3|
|hc2_recheck|ParentGlobalSemanticLoop2|27.3123|51.9587|15/12/5|5|
|hc2_recheck|SharpenOriginal|27.5315|50.1412|8/19/5|5|
|hc2_recheck|AllKeysNoOp|27.8907|51.6010|0/0/32|0|
|hc2_recheck|Source|23.4655|50.4639|—|—|

hc2_recheck: primary−G6 t+0.4793pp CI[-2.02933643259865, 3.001041406571229]; v+0.0033pp CI[-1.5652525821194962, 1.4250160337100535]. Primary−Source t+1.6164/v+4.4285pp.
Source short+0.3358/long+2.2872tpp; fixed screen=False.
Primary−ParentGlobalSemanticLoop2 t+0.1216pp CI[-1.514945843409547, 1.4958829840743013].
Primary−ReflectedSemanticLoop2 t-0.6684pp CI[-2.8365225311223443, 1.2625380851205859].
Primary−WrongTimeSemanticLoop2 t+1.3792pp CI[-1.0075339656288376, 5.035303821812306].
Primary−RandomTargetSemanticLoop2 t-0.3991pp CI[-3.348762325458736, 2.4620425210798667].
|vidstg_recheck|NativeG6|23.9576|35.6554|—|—|
|vidstg_recheck|TargetSemanticLoop2|25.3553|36.3389|9/16/7|6|
|vidstg_recheck|OneStepTarget|25.1420|36.5028|10/13/9|5|
|vidstg_recheck|DirectTargetEvidence|22.9150|33.1622|11/19/2|13|
|vidstg_recheck|ReflectedSemanticLoop2|25.2854|36.3672|11/11/10|6|
|vidstg_recheck|WrongTimeSemanticLoop2|23.5428|34.0109|9/16/7|6|
|vidstg_recheck|RandomTargetSemanticLoop2|23.1706|34.1383|9/12/11|5|
|vidstg_recheck|ParentGlobalSemanticLoop2|23.6888|35.0400|9/16/7|7|
|vidstg_recheck|SharpenOriginal|23.4813|35.3245|14/11/7|1|
|vidstg_recheck|AllKeysNoOp|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|Source|15.6772|34.0110|—|—|

vidstg_recheck: primary−G6 t+0.6835pp CI[-2.308653907793297, 4.70051731851963]; v+1.3978pp CI[-0.9044742678731547, 4.8133948434212614]. Primary−Source t+2.3278/v+9.6781pp.
Source short-0.6685/long+13.0288tpp; fixed screen=False.
Primary−ParentGlobalSemanticLoop2 t+1.2988pp CI[-1.0627267549055293, 4.011682458974264].
Primary−ReflectedSemanticLoop2 t-0.0283pp CI[-2.570512907844257, 2.520994969487515].
Primary−WrongTimeSemanticLoop2 t+2.3280pp CI[-0.4227128074979705, 6.227025156210302].
Primary−RandomTargetSemanticLoop2 t+2.2006pp CI[-1.0728622437019644, 6.404163762672821].

Not qualified. No context/softmask/strength/iteration scan, control promotion or full/default upgrade.
Actualcalls{'ASA_head_forwards': 1280, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_time_decoder': 0, 'new_expert': 0, 'new_observations': 0, 'new_optimizer': 0}; GPU studyworker31.93s includes loading,5reads/query andrecording, NOTdeploymenttiming. Additional sourceASA headnetwork computations counted; no newRGB/visual/text/fusion/time-decoder/expert/observations/optimizer. Official/densechecks2816maxerror1.89e-15.

[All signed results](signed.tsv), [all severe declines](severe_declines.tsv), [protocol](protocol.json), [native cross-attention code](https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/bert_model/bert_module.py).
