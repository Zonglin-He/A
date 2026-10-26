# vidstg: standalone retained192 gamma review control

Exactly192queries, 192 parents. This pre-existing retained-raw cohort includes previously exposed development and confirmation sources; it is not the full1902/10303 domain, an unseen test, or a prefix accuracy estimate. All192 prediction configurations sealed before this independent official scoring. Separate actual ROI repooling/cold gamma chain, prediction/input/parent/receipt SHA and all-mean audits passed.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|29.926298|12.612844|14.062500|3.125000|
|Gamma0|34.050322|14.775759|18.229167|6.770833|
|Gamma025|33.507463|14.637668|18.750000|7.291667|
|Gamma05|33.508715|14.879680|19.791667|6.770833|
|Gamma075|33.554295|15.150721|20.312500|8.333333|
|Gamma1|33.734719|15.350508|20.312500|8.854167|

Parent-paired10000-bootstrap pointwise95% CIs, no familywise significance claim:

- Gamma0-Source: t+4.124024pp CI[2.200951611646878, 6.208201076820147]; v+2.162915pp CI[0.8184962596378428, 3.585966926778717]; severe t/v/union28/18/35, rescues78.
- Gamma025-Source: t+3.581165pp CI[1.657927328767272, 5.712936994451079]; v+2.024824pp CI[0.5556440111150974, 3.5557540034587607]; severe t/v/union29/22/41, rescues70.
- Gamma05-Source: t+3.582417pp CI[1.6776398050517167, 5.65607475621899]; v+2.266836pp CI[0.7234930223053088, 3.875646740754599]; severe t/v/union29/24/43, rescues75.
- Gamma075-Source: t+3.627997pp CI[1.7366280732763562, 5.706473430401296]; v+2.537876pp CI[0.8719789036103788, 4.238977667356815]; severe t/v/union30/25/43, rescues82.
- Gamma1-Source: t+3.808420pp CI[1.9292801496811613, 5.876613509238303]; v+2.737663pp CI[0.9896791622984895, 4.513635268269065]; severe t/v/union28/25/43, rescues88.
- Gamma0-Gamma1: t+0.315604pp CI[-0.24456558833327124, 0.8345035565445305]; v-0.574749pp CI[-1.4665660319839806, 0.40058400405347117]; severe t/v/union7/28/34, rescues27.
- Gamma025-Gamma1: t-0.227255pp CI[-0.7999805985884738, 0.3011395923930443]; v-0.712840pp CI[-1.4163012549143168, 0.02452202201167155]; severe t/v/union12/20/32, rescues17.
- Gamma05-Gamma1: t-0.226003pp CI[-0.6508515761961469, 0.17599911691749118]; v-0.470828pp CI[-0.9593199905968256, 0.02393585780270061]; severe t/v/union9/9/18, rescues15.
- Gamma075-Gamma1: t-0.180424pp CI[-0.5160092106175197, 0.14276937069015916]; v-0.199787pp CI[-0.4722717490176036, 0.04769146285419765]; severe t/v/union9/4/13, rescues6.

Short<=5s/long ground-truth-duration strata are offline only:

- short_le5s, n118, Gamma0-Source: v+0.391241pp CI[-0.8021227995315161, 1.5915778840966286]; t+0.509633 CI[-1.1452393747431306, 2.170519816683574].
- short_le5s, n118, Gamma025-Source: v+0.176913pp CI[-1.0942775195394454, 1.4581626751462022]; t+0.059056 CI[-1.5434283254481167, 1.709605834531622].
- short_le5s, n118, Gamma05-Source: v+0.520143pp CI[-0.814332002319646, 1.8857647231256562]; t+0.200429 CI[-1.424335138489092, 1.826913857129979].
- short_le5s, n118, Gamma075-Source: v+0.784245pp CI[-0.5918738649067988, 2.2133183989842906]; t+0.049186 CI[-1.5187644234023117, 1.6174337215144499].
- short_le5s, n118, Gamma1-Source: v+0.959518pp CI[-0.4248373343966502, 2.3936609644564544]; t+0.433597 CI[-1.1546238299874418, 2.0199162033691636].
- short_le5s, n118, Gamma0-Gamma1: v-0.568277pp CI[-1.4408968188976254, 0.3580252067475804]; t+0.076036 CI[-0.6291903893874667, 0.7244885870028063].
- short_le5s, n118, Gamma025-Gamma1: v-0.782605pp CI[-1.3254670208528632, -0.2497748443727949]; t-0.374541 CI[-1.1083728192906792, 0.26823250788028424].
- short_le5s, n118, Gamma05-Gamma1: v-0.439376pp CI[-0.8448059733828825, -0.012382922180762248]; t-0.233168 CI[-0.7674366605504235, 0.27196684191170595].
- short_le5s, n118, Gamma075-Gamma1: v-0.175274pp CI[-0.377242236874369, 0.020689459305905225]; t-0.384411 CI[-0.8659445628310964, 0.04385003512978228].
- long_gt5s, n74, Gamma0-Source: v+4.988015pp CI[2.249898521917487, 8.12667528032344]; t+9.887512 CI[5.909591737494351, 14.290516224635562].
- long_gt5s, n74, Gamma025-Source: v+4.971493pp CI[1.940916293938199, 8.305276862646764]; t+9.197501 CI[5.099372732515901, 13.806832258918128].
- long_gt5s, n74, Gamma05-Source: v+5.052103pp CI[1.789036872435582, 8.550935251545962]; t+8.975317 CI[4.940209882053137, 13.406608782441667].
- long_gt5s, n74, Gamma075-Source: v+5.334208pp CI[1.750481397036863, 9.11617352480265]; t+9.334749 CI[5.3214036180739, 13.819435171251506].
- long_gt5s, n74, Gamma1-Source: v+5.573084pp CI[1.7631192478600393, 9.620162332528484]; t+9.189896 CI[5.1297578391782475, 13.686027981386646].
- long_gt5s, n74, Gamma0-Gamma1: v-0.585069pp CI[-2.416434195540979, 1.4515526071623122]; t+0.697617 CI[-0.15614785330484696, 1.5513222681602818].
- long_gt5s, n74, Gamma025-Gamma1: v-0.601591pp CI[-2.2065825229056704, 1.0803156847500215]; t+0.007606 CI[-0.8745032331808967, 0.8997510031857673].
- long_gt5s, n74, Gamma05-Gamma1: v-0.520981pp CI[-1.6026114227367667, 0.5636666206133067]; t-0.214579 CI[-0.9131406917456205, 0.4720034753262504].
- long_gt5s, n74, Gamma075-Gamma1: v-0.238876pp CI[-0.8564884879638585, 0.30278761021036893]; t+0.144853 CI[-0.3206015157345812, 0.6140140583452758].

Every gamma point reconstructs E, bidirectional geometry, actual native-grid E ROI features, prototype/posterior, own legal jointMAP interval and finalDP. Both spatial and temporal downstream effects change, so this is not an isolated appearance parameter. Current historical defaults remain HC0.5/Vid1; the complete five-point curve is descriptive sensitivity on exposed data and does not establish source-only calibration. No default selection, new fullGamma acquisition, old-owner restoration or main-method change.

All signed positive/negative rows, severe tails/rescues and short/long CIs are retained. Conditional replay elapsed clocks include file SHA, resource checks and actual original-grid pooling; not process CPU or complete deployment latency. New model/Source/expert/optimizer calls are zero. Original acquired Source/eight-expert contexts and old gamma preprocessing are inherited nonzero costs, not free observations. Shared original Vid producer/TTA costs must not be added twice.
