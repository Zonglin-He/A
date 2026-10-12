# P9: teacher geometry versus update actionability

One finite CPU audit of the exact P7 roster:128 historical nondevelopment parents per target,2 original orders,512 cells and1024 cached Omega0/Omega1 updates.512 complete new readouts and every test-visible score were sealed before offline GT labels. Zero new model forward/GPU fit/DINO/optimizer steps. Shared Before boxes/parameter states and original source/pixel/frame/expert bindings were checked. Original complete fit mathematics and GPU repeat qualification are reused; no new model-Jacobian replay.

Primary consequence is spatial IoU after-minus-before on GT-event real decoder frames excluding both observation sets. Requested own-U excludes only the training allocation and is also reported in full. Interpolated/unsupported/missing GT frames never enter these labels. Held-out admitted anchors are chosen strictly from other allocation minus actual training-observed positions, without GT filtering. Both sets come from the same frozen DINO and cannot certify referent identity or independent-teacher correctness.

Parent means average two orders first. Single orders, previous32, added96 and all128 are separately saved. Binary AUROC and tie-group average precision (AP) condition on positive/negative nonneutral consequences and defined scores; unknown/neutral cases remain in all-parent tables.10000 paired parent bootstrap, seed20261012; confidence intervals are exploratory and unadjusted for multiple diagnostics. Undefined classless resamples are counted, not imputed. Matched AUROC differences use the identical eligible intersection. Teacher GT quality is offline oracle information, unavailable in deployment.

|Target|Training scope|Subset|All parents|Positive|Negative|Neutral|Undefined U|Held-out eligible +/-|AUROC [95% CI]|AP [95% CI]|Eligible positive prevalence|
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|
|hc2|Omega0|all128|128|65|56|7|0|110 (58/52)|0.5633 [0.4527, 0.6715]|0.5487 [0.4388, 0.6923]|0.5273|
|hc2|Omega0|previous_P6_32|32|16|15|1|0|30 (15/15)|0.6000 [0.3750, 0.8170]|0.5420 [0.3616, 0.8230]|0.5000|
|hc2|Omega0|added96|96|49|41|6|0|80 (43/37)|0.5732 [0.4403, 0.6989]|0.5790 [0.4520, 0.7503]|0.5375|
|hc2|Omega1|all128|128|61|59|8|0|110 (55/55)|0.5749 [0.4602, 0.6845]|0.5146 [0.4120, 0.6542]|0.5000|
|hc2|Omega1|previous_P6_32|32|12|19|1|0|31 (12/19)|0.5439 [0.3333, 0.7542]|0.4068 [0.2460, 0.6820]|0.3871|
|hc2|Omega1|added96|96|49|40|7|0|79 (43/36)|0.5898 [0.4513, 0.7227]|0.5862 [0.4584, 0.7481]|0.5443|
|vidstg|Omega0|all128|128|61|35|32|0|77 (51/26)|0.6169 [0.4879, 0.7453]|0.7805 [0.6643, 0.8866]|0.6623|
|vidstg|Omega0|previous_P6_32|32|17|7|8|0|19 (13/6)|0.5769 [0.2833, 0.8571]|0.7914 [0.5733, 0.9618]|0.6842|
|vidstg|Omega0|added96|96|44|28|24|0|58 (38/20)|0.6276 [0.4764, 0.7722]|0.7854 [0.6507, 0.9056]|0.6552|
|vidstg|Omega1|all128|128|61|36|31|0|80 (51/29)|0.5950 [0.4665, 0.7177]|0.7425 [0.6174, 0.8615]|0.6375|
|vidstg|Omega1|previous_P6_32|32|17|7|8|0|19 (13/6)|0.4872 [0.1795, 0.7976]|0.6861 [0.4721, 0.9466]|0.6842|
|vidstg|Omega1|added96|96|44|29|23|0|61 (38/23)|0.6236 [0.4760, 0.7607]|0.7839 [0.6612, 0.8885]|0.6230|

|Target|Scope|Mean labeled Q>=0.5, harmful parent|Fully-labeled all anchors>=0.5, harmful parent|Former, harmful in both orders|Strict, harmful in both orders|Opposite order signs|
|---|---|---:|---:|---:|---:|---:|
|hc2|Omega0|22|5|16|3|25|
|hc2|Omega1|23|4|16|2|18|
|vidstg|Omega0|15|5|10|3|14|
|vidstg|Omega1|15|4|11|3|14|

Mean labeled Q>=0.5 is a prespecified descriptive geometric criterion, not an online admission threshold. Partial annotation does not label every other teacher anchor correct. Strict counts require every admitted anchor to have an event annotation and IoU>=0.5; empty evidence is not correct by vacuous truth. These are parent/scope diagnostics, not a universal independent-video failure rate. Accurate geometry co-occurring with harm does not isolate referent confusion, an optimizer variable or a Jacobian as its cause.

|Target|Scope|Subset|Held-out minus baseline on identical parents|Eligible parents|AUROC difference [95% CI]|
|---|---|---|---|---:|---|
|hc2|Omega0|all128|detector_score|110|-0.0836 [-0.2360, 0.0679]|
|hc2|Omega0|all128|before_training_agreement|110|-0.0507 [-0.2148, 0.1127]|
|hc2|Omega0|all128|training_agreement_gain|110|0.1015 [-0.0415, 0.2376]|
|hc2|Omega0|all128|teacher_geometric_consistency|107|0.0577 [-0.0886, 0.2078]|
|hc2|Omega0|all128|negative_output_displacement|110|-0.0070 [-0.1766, 0.1582]|
|hc2|Omega0|added96|detector_score|80|-0.0566 [-0.2257, 0.1175]|
|hc2|Omega0|added96|before_training_agreement|80|0.0377 [-0.1535, 0.2273]|
|hc2|Omega0|added96|training_agreement_gain|80|0.1062 [-0.0589, 0.2726]|
|hc2|Omega0|added96|teacher_geometric_consistency|79|0.1208 [-0.0643, 0.3011]|
|hc2|Omega0|added96|negative_output_displacement|80|-0.0082 [-0.2007, 0.1848]|
|hc2|Omega1|all128|detector_score|110|0.0003 [-0.1625, 0.1567]|
|hc2|Omega1|all128|before_training_agreement|110|0.0251 [-0.1452, 0.1919]|
|hc2|Omega1|all128|training_agreement_gain|110|0.1058 [-0.0251, 0.2371]|
|hc2|Omega1|all128|teacher_geometric_consistency|110|0.1131 [-0.0290, 0.2555]|
|hc2|Omega1|all128|negative_output_displacement|110|0.0324 [-0.1379, 0.2001]|
|hc2|Omega1|added96|detector_score|79|0.0052 [-0.1849, 0.1922]|
|hc2|Omega1|added96|before_training_agreement|79|0.0568 [-0.1477, 0.2601]|
|hc2|Omega1|added96|training_agreement_gain|79|0.1783 [0.0182, 0.3386]|
|hc2|Omega1|added96|teacher_geometric_consistency|79|0.1331 [-0.0348, 0.3007]|
|hc2|Omega1|added96|negative_output_displacement|79|0.0239 [-0.1840, 0.2240]|
|vidstg|Omega0|all128|detector_score|77|0.0136 [-0.1720, 0.2076]|
|vidstg|Omega0|all128|before_training_agreement|77|0.1946 [-0.0216, 0.4106]|
|vidstg|Omega0|all128|training_agreement_gain|77|0.0271 [-0.0816, 0.1385]|
|vidstg|Omega0|all128|teacher_geometric_consistency|76|0.1738 [-0.0024, 0.3500]|
|vidstg|Omega0|all128|negative_output_displacement|77|0.1674 [-0.0548, 0.3846]|
|vidstg|Omega0|added96|detector_score|58|0.0053 [-0.2162, 0.2294]|
|vidstg|Omega0|added96|before_training_agreement|58|0.2303 [-0.0194, 0.4743]|
|vidstg|Omega0|added96|training_agreement_gain|58|0.0276 [-0.1120, 0.1763]|
|vidstg|Omega0|added96|teacher_geometric_consistency|57|0.1459 [-0.0790, 0.3706]|
|vidstg|Omega0|added96|negative_output_displacement|58|0.1763 [-0.0744, 0.4222]|
|vidstg|Omega1|all128|detector_score|80|0.0460 [-0.1457, 0.2311]|
|vidstg|Omega1|all128|before_training_agreement|80|0.1406 [-0.0684, 0.3448]|
|vidstg|Omega1|all128|training_agreement_gain|80|0.0629 [-0.0742, 0.2012]|
|vidstg|Omega1|all128|teacher_geometric_consistency|77|0.2296 [0.0809, 0.3815]|
|vidstg|Omega1|all128|negative_output_displacement|80|0.0663 [-0.1536, 0.2798]|
|vidstg|Omega1|added96|detector_score|61|0.0709 [-0.1472, 0.2813]|
|vidstg|Omega1|added96|before_training_agreement|61|0.1854 [-0.0443, 0.4111]|
|vidstg|Omega1|added96|training_agreement_gain|61|0.1041 [-0.0571, 0.2658]|
|vidstg|Omega1|added96|teacher_geometric_consistency|58|0.2870 [0.0958, 0.4667]|
|vidstg|Omega1|added96|negative_output_displacement|61|0.0664 [-0.1845, 0.3133]|

Predeclared information criterion supports considering one bounded counterfactual selection test: **False**. No learned gate, threshold search, dynamic alpha, additional teacher, query-space strategy or production change was implemented. Failure of this conjunction does not imply every test-visible signal is information-free or every future actionability mechanism impossible.

After-the-update held-out agreement requires a candidate update and additional cached teacher observations; this audit is not a zero-cost pre-update predictor. Its zero new DINO count means existing evidence reuse, not zero deployment expert cost. Any future controlled selection test needs an explicit matched budget and state lifecycle, and cannot use these GT labels online.

All1024 anonymous rows,512 parent/scope rows, both outcomes, six fixed test-visible scores, oracle geometry, full36 target/scope/subset/order blocks, positive prevalence, neutral/undefined coverage and every negative case are retained. No target-GT polarity flip, fitted threshold, selected positive-only subset or paper manuscript. C expanded-state GPU execution continues with its original scientific runtime; cancelled formal baselines are not restored.
