# Dense semantic feedback through existing native temporal features

Reuse old ridge1 interface and normalized six-layer native hidden concat; replace old sparse8ReferDINO verb/noun margins with dense original learnedSource-word TargetContext feedback. Every round fits opposite-parity dense LLR then restores original units. Original prior q0, inclusive event evidence SUM, physical MBR and complete-box spatial DP fixed. Exactly two rounds, no SGD/query updates/network weight training/new observations or expert. Native temporal features were trained for boundaries but may retain the same wrong hypotheses; not proof of extra event information.

Controls isolate same-info directsecondrawfeedback, parentDirectTarget, one round, reflected/random/reversedsemantic placement, original learnedposition-only features, halfcycle temporalhidden permutation, sharpen and constant/nativealiases. Crossfit pseudo-targets are not independent validation. All128 seal before official CPU labels; all panels previously exposed. Five tests cover exactoldridge, constantnative, heldoutlabelisolation, feature rotation/scale and synthetic linear positive control.

|Panel|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---|---:|---:|---:|---:|
|HC64|NativeG6|29.6952|55.9613|—|—|
|HC64|DenseTemporalProjectionLoop2|31.4450|59.8943|34/18/12|3|
|HC64|OneStepProjection|31.0899|58.9761|35/18/11|3|
|HC64|DirectSecondRoundFeedback|31.6185|60.3707|36/14/14|2|
|HC64|ParentDirectTarget|31.5936|60.2765|36/14/14|2|
|HC64|ReflectedProjectionLoop2|30.3113|58.3031|32/20/12|7|
|HC64|RandomProjectionLoop2|29.8872|56.4799|21/26/17|7|
|HC64|ReversedSemanticProjectionLoop2|29.1158|54.4702|20/27/17|15|
|HC64|NativePositionProjectionLoop2|31.2695|59.5203|34/19/11|3|
|HC64|ShuffledTemporalProjectionLoop2|30.3959|57.9253|31/20/13|7|
|HC64|SharpenOriginal|28.7258|53.7518|16/38/10|20|
|HC64|ConstantFeedbackNoOp|29.6952|55.9613|0/0/64|0|
|HC64|Source|26.8938|54.8230|—|—|

HC64: primary−G6 t+3.9330pp CI[2.0754257869887516, 6.053591069743009], v+1.7498pp CI[0.7661353353480445, 2.8155542942897065]. Primary−Source t+5.0713/v+4.5512pp.
Source short+1.7559/long+5.9164tpp; fixed screen=True.
Primary−ReflectedProjectionLoop2: t+1.5912pp CI[-0.014789488288577556, 3.3144381218641255].
Primary−RandomProjectionLoop2: t+3.4144pp CI[1.757719842007476, 5.205215071950952].
Primary−ReversedSemanticProjectionLoop2: t+5.4242pp CI[2.910506482391619, 8.316180692199204].
Primary−NativePositionProjectionLoop2: t+0.3740pp CI[-0.3252724557832043, 1.0850153960194913].
Primary−ShuffledTemporalProjectionLoop2: t+1.9690pp CI[0.5372748559012779, 3.6401765812258957].
Primary−SharpenOriginal: t+6.1425pp CI[3.5773248881965736, 8.880545509007051].
Primary−DirectSecondRoundFeedback: t-0.4764pp CI[-1.4080487054490967, 0.3700814835100632].
Primary−ParentDirectTarget: t-0.3822pp CI[-1.3442622565555076, 0.5281486022288907].
Primary−OneStepProjection: t+0.9182pp CI[0.17395050219794467, 1.6501026909760237].
|hc2_recheck|NativeG6|27.8907|51.6010|—|—|
|hc2_recheck|DenseTemporalProjectionLoop2|27.5347|51.2975|14/12/6|4|
|hc2_recheck|OneStepProjection|27.7743|51.3278|15/11/6|4|
|hc2_recheck|DirectSecondRoundFeedback|28.0633|52.2176|15/11/6|5|
|hc2_recheck|ParentDirectTarget|27.8940|52.0803|14/11/7|5|
|hc2_recheck|ReflectedProjectionLoop2|27.4762|50.9725|13/12/7|7|
|hc2_recheck|RandomProjectionLoop2|27.4833|50.8417|12/11/9|5|
|hc2_recheck|ReversedSemanticProjectionLoop2|27.2388|51.1086|12/14/6|4|
|hc2_recheck|NativePositionProjectionLoop2|27.7125|51.2971|12/12/8|5|
|hc2_recheck|ShuffledTemporalProjectionLoop2|27.5673|51.1027|12/13/7|5|
|hc2_recheck|SharpenOriginal|27.5315|50.1412|8/19/5|5|
|hc2_recheck|ConstantFeedbackNoOp|27.8907|51.6010|0/0/32|0|
|hc2_recheck|Source|23.4655|50.4639|—|—|

hc2_recheck: primary−G6 t-0.3035pp CI[-2.8052148755165818, 2.067006155319353], v-0.3560pp CI[-1.8871299947248639, 0.85989002137594]. Primary−Source t+0.8336/v+4.0692pp.
Source short-0.2444/long+1.3982tpp; fixed screen=False.
Primary−ReflectedProjectionLoop2: t+0.3250pp CI[-0.6812474978428833, 1.3906106798334135].
Primary−RandomProjectionLoop2: t+0.4558pp CI[-1.5667175541874059, 2.799738272009552].
Primary−ReversedSemanticProjectionLoop2: t+0.1889pp CI[-2.7459939993673474, 3.262442743125427].
Primary−NativePositionProjectionLoop2: t+0.0004pp CI[-0.8199274122929459, 0.7780124668002372].
Primary−ShuffledTemporalProjectionLoop2: t+0.1948pp CI[-1.6821275648819807, 2.247730176000679].
Primary−SharpenOriginal: t+1.1563pp CI[-2.5040300341600346, 4.81827899074429].
Primary−DirectSecondRoundFeedback: t-0.9201pp CI[-1.9435140312309176, 0.058275779146358404].
Primary−ParentDirectTarget: t-0.7828pp CI[-1.9255470740991696, 0.2820231646900527].
Primary−OneStepProjection: t-0.0304pp CI[-0.5787473178539483, 0.5374734659346696].
|vidstg_recheck|NativeG6|23.9576|35.6554|—|—|
|vidstg_recheck|DenseTemporalProjectionLoop2|24.5365|35.5757|7/17/8|5|
|vidstg_recheck|OneStepProjection|24.0732|35.3711|7/16/9|3|
|vidstg_recheck|DirectSecondRoundFeedback|25.3116|36.6090|10/14/8|6|
|vidstg_recheck|ParentDirectTarget|25.3553|36.3389|9/16/7|6|
|vidstg_recheck|ReflectedProjectionLoop2|24.9283|36.2193|10/12/10|5|
|vidstg_recheck|RandomProjectionLoop2|22.8233|34.2571|9/13/10|3|
|vidstg_recheck|ReversedSemanticProjectionLoop2|23.0206|34.0219|11/14/7|6|
|vidstg_recheck|NativePositionProjectionLoop2|25.2445|36.5603|7/15/10|4|
|vidstg_recheck|ShuffledTemporalProjectionLoop2|25.8763|37.2042|8/13/11|4|
|vidstg_recheck|SharpenOriginal|23.4813|35.3245|14/11/7|1|
|vidstg_recheck|ConstantFeedbackNoOp|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|Source|15.6772|34.0110|—|—|

vidstg_recheck: primary−G6 t-0.0796pp CI[-2.8150257968524817, 3.2705314922714153], v+0.5790pp CI[-1.4390442451884564, 3.2676684931069797]. Primary−Source t+1.5647/v+8.8593pp.
Source short-0.6481/long+9.4675tpp; fixed screen=False.
Primary−ReflectedProjectionLoop2: t-0.6435pp CI[-2.503177762468979, 1.0730420664383629].
Primary−RandomProjectionLoop2: t+1.3186pp CI[-1.55819091809745, 4.807487333695369].
Primary−ReversedSemanticProjectionLoop2: t+1.5538pp CI[-3.111906120788758, 7.291193153850895].
Primary−NativePositionProjectionLoop2: t-0.9846pp CI[-2.555655105588499, 0.20778398912564505].
Primary−ShuffledTemporalProjectionLoop2: t-1.6285pp CI[-3.9840612337638306, 0.22914019433328703].
Primary−SharpenOriginal: t+0.2512pp CI[-3.4379446804825773, 4.4221960993615115].
Primary−DirectSecondRoundFeedback: t-1.0333pp CI[-2.6334730709700644, 0.1997729082788917].
Primary−ParentDirectTarget: t-0.7631pp CI[-2.3092649706675674, 0.4033951588362977].
Primary−OneStepProjection: t+0.2046pp CI[-1.5867510020054574, 2.573782840113132].

Not qualified. No ridge/layer/round/gain scan or control promotion.
CPU worker9.75s includes all controls/DP/I/O, not deployment latency. Actual counts{'new_model_calls': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_ASA_head': 0, 'new_expert': 0, 'new_observations': 0, 'new_optimizer_steps': 0, 'ridge_fits_current_worker': 896, 'ridge_fits_including_exact96': 3584, 'migrated_exact_predictions': 96}. Official/dense checks3328, maxerror1.78e-15.

[All signed outcomes](signed.tsv), [severe declines](severe_declines.tsv), [protocol](protocol.json).
