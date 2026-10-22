# Native trained time-decoder dropout read

Test conditional temporal stability of the existing trained source. Actual original7 TimeDecoder kwargs/strides/fullcaption/all original physical nodes/space/evidence8 unchanged. Enable ONLY original nn.Dropout and nn.MultiheadAttention inside TimeDecoder with original probabilities, four fixed draws per query, scoped input-SHA/seed20260920 RNG. Restore flags/rates finally; all source weights frozen. Six native temporal layers and two-offset old hull per draw, probability mean, fixed half mix with NativeG6, original physical MBR and complete-box spatial DP. No new semantic/expert signals or optimizer. This is a feasibility test, not a new Bayesian method or proof of probability calibration/private checkpoint training history.

Native128 and actual same-flags zero-probability replay restore hidden/logits/q bitwise. NoDropoutMean/ZeroDropout/Sharpen preserve parent q/interval/finalboxes. Five CPU tests cover exception restoration, zero/no-op and deterministic seed, empirical Gaussian moment matching, wrong-time histogram/roundtrip and legal probability normalization. All128 sealed before independent CPU official labels. All panels are previously exposed development/fixed rechecks.

Same-information controls: DirectMCMean, fixed FirstDraw, meanlogits before posterior, independent-coordinate Gaussian deltas with exact actual4draw empirical mean/std up to FP32 rounding, within-offset halfcycle delta shift, sharpen and aliases. Gaussian needs actual MC statistics and is not a zero-model deployment alternative. Wrong-time delta is a placement control, not a known negative event.

|Panel|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---|---:|---:|---:|---:|
|HC64|NativeG6|29.6952|55.9613|—|—|
|HC64|NativeDropoutFusion|29.8502|56.1564|9/6/49|0|
|HC64|DirectMCMean|29.9488|56.2315|16/13/35|3|
|HC64|FirstDrawFusion|29.8698|56.1645|11/9/44|2|
|HC64|MeanLogitsFusion|29.7950|56.0314|8/7/49|1|
|HC64|MomentMatchedGaussianFusion|29.8502|56.1564|9/6/49|0|
|HC64|WrongTimeDeltaFusion|29.7632|56.0283|11/10/43|0|
|HC64|NoDropoutMean|29.6952|55.9613|0/0/64|0|
|HC64|SharpenOriginal|28.7258|53.7518|16/38/10|20|
|HC64|ZeroDropoutNoOp|29.6952|55.9613|0/0/64|0|
|HC64|Source|26.8938|54.8230|—|—|

HC64: primary−G6 t+0.1950pp CI[-0.12435345041816007, 0.5594866740217558], v+0.1550pp CI[-0.014219463553088197, 0.33062303916919544]. Primary−Source t+1.3334/v+2.9564pp.
Source short+0.0095/long+1.6708tpp; fixed screen=False.
Primary−MomentMatchedGaussianFusion: t+0.0000pp CI[0.0, 0.0].
Primary−WrongTimeDeltaFusion: t+0.1280pp CI[-0.40346836466513814, 0.6679654371403746].
Primary−SharpenOriginal: t+2.4045pp CI[0.8615791108750572, 3.937795489955109].
Primary−FirstDrawFusion: t-0.0082pp CI[-0.42650275194559617, 0.392867666653073].
Primary−MeanLogitsFusion: t+0.1249pp CI[-0.01628959989902817, 0.36379027974504236].
|hc2_recheck|NativeG6|27.8907|51.6010|—|—|
|hc2_recheck|NativeDropoutFusion|28.2276|52.1057|6/6/20|1|
|hc2_recheck|DirectMCMean|27.7785|51.6457|12/11/9|1|
|hc2_recheck|FirstDrawFusion|27.5959|51.1985|4/7/21|2|
|hc2_recheck|MeanLogitsFusion|28.2255|51.9469|6/6/20|1|
|hc2_recheck|MomentMatchedGaussianFusion|28.2276|52.1057|6/6/20|1|
|hc2_recheck|WrongTimeDeltaFusion|28.0810|51.8246|6/5/21|0|
|hc2_recheck|NoDropoutMean|27.8907|51.6010|0/0/32|0|
|hc2_recheck|SharpenOriginal|27.5315|50.1412|8/19/5|5|
|hc2_recheck|ZeroDropoutNoOp|27.8907|51.6010|0/0/32|0|
|hc2_recheck|Source|23.4655|50.4639|—|—|

hc2_recheck: primary−G6 t+0.5047pp CI[-0.5296925550133019, 1.7057709129663896], v+0.3369pp CI[-0.40563834860990744, 1.170141483420748]. Primary−Source t+1.6418/v+4.7622pp.
Source short+1.1296/long+1.9101tpp; fixed screen=False.
Primary−MomentMatchedGaussianFusion: t+0.0000pp CI[0.0, 0.0].
Primary−WrongTimeDeltaFusion: t+0.2811pp CI[-0.7687510636420617, 1.52604607602842].
Primary−SharpenOriginal: t+1.9646pp CI[-0.7590274425533724, 5.242080434602972].
Primary−FirstDrawFusion: t+0.9072pp CI[0.12088377561267274, 1.8316305814512959].
Primary−MeanLogitsFusion: t+0.1588pp CI[-0.14728246495636055, 0.5582616127770321].
|vidstg_recheck|NativeG6|23.9576|35.6554|—|—|
|vidstg_recheck|NativeDropoutFusion|24.0881|35.8364|4/7/21|0|
|vidstg_recheck|DirectMCMean|23.8602|35.6014|7/13/12|1|
|vidstg_recheck|FirstDrawFusion|23.6478|35.3339|3/11/18|1|
|vidstg_recheck|MeanLogitsFusion|24.0773|35.7572|5/5/22|0|
|vidstg_recheck|MomentMatchedGaussianFusion|24.0881|35.8364|4/7/21|0|
|vidstg_recheck|WrongTimeDeltaFusion|23.8096|35.5045|2/7/23|0|
|vidstg_recheck|NoDropoutMean|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|SharpenOriginal|23.4813|35.3245|14/11/7|1|
|vidstg_recheck|ZeroDropoutNoOp|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|Source|15.6772|34.0110|—|—|

vidstg_recheck: primary−G6 t+0.1810pp CI[-0.18548484526885725, 0.6536557353111099], v+0.1306pp CI[-0.12377612991865738, 0.461186222362651]. Primary−Source t+1.8253/v+8.4109pp.
Source short+1.1156/long+4.3600tpp; fixed screen=False.
Primary−MomentMatchedGaussianFusion: t+0.0000pp CI[0.0, 0.0].
Primary−WrongTimeDeltaFusion: t+0.3319pp CI[-0.07849478875500888, 0.8129393227752586].
Primary−SharpenOriginal: t+0.5118pp CI[-1.6244043101223122, 3.4843258924448777].
Primary−FirstDrawFusion: t+0.5024pp CI[-0.011837630289828534, 1.2077224064458243].
Primary−MeanLogitsFusion: t+0.0791pp CI[-0.17968086940648598, 0.4559830566187687].

Not qualified; no p/K/seed/fusion/query budget scan, no control promotion.
Actual worker122.96s includes loading/all controls/DP/I/O; {'time_decoder_calls': 1536, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_ASA_head': 0, 'new_expert': 0, 'new_observations': 0, 'new_optimizer': 0}. Extra deployment calls would be8TimeDecoder reads, but deployment E2E latency is unmeasured. Gaussian empirical moment FP32 max error5.61e-08. Official/dense checks2816, maxerror1.78e-15.

[Complete signed outcomes](signed.tsv), [severe declines](severe_declines.tsv), [protocol](protocol.json). [Gal and Ghahramani, ICML2016](https://proceedings.mlr.press/v48/gal16.html) motivates dropout uncertainty, not STVG efficacy.

Protocol wording correction: HC grids are65nodes. Fixed Vid32 original source grids vary16–200; actual dropout code always used original saved shapes and offsets, never resampled them. Original locked plan wording is retained as evidence; this correction does not change code/predictions or the128 bitwise no-op checks.
