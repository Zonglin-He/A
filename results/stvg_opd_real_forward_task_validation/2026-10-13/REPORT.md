# Real-Forward: complete committed-state task validation

Actual fixed64 historical queries/Inherited-order1/Omega0, four complete arms:64 original-A control replays and192 new SEL/HALF/LN fits,6400 saved rounds. Qualification separately32 complete fits/800 rounds/eight fresh captures. All256 formal predictions were sealed before GT scoring. Every A control exactly reproduces its old complete fit. No new DINO, teacher, hyperparameter search, main run or production promotion.

Every step recomputes OPD gradient and Adam from the actual state committed by that arm. SEL uses the unchanged A/LN/Query/Half, rho=.9, fixed tie order and original A for nonpositive G(A). HALF/LN are proposal masks with all1792 Adam moments retained, not optimizer-block removal. Admitted observation evidence and actual noise are common; sampled actions diverge after states diverge.

Primary contrast SEL-A complete mvIoU; source-equal32 parents per target, paired10000 bootstrap seed20261013. All intervals are conditional on these historically exposed states; secondary intervals are exploratory and unadjusted. This is episodic receiving-state adaptation, not a new prequential stream.

|Target|Arm|Full mvIoU gain versus entering (pp,95%CI)|Observed GT learning|Unvisited correct-frame IoU gain|Unvisited wrong-frame IoU gain|
|---|---|---|---|---|---|
|hc2|A|-1.6944 [-6.2764,+1.1561]; n=32|-0.7985 [-8.7494,+6.1501]; n=26|-8.3708 [-17.1905,-1.0339]; n=26|+4.7403 [+1.0883,+9.4377]; n=22|
|hc2|SEL|-1.9931 [-6.6011,+0.8421]; n=32|-1.0806 [-8.9684,+5.8399]; n=26|-8.1754 [-16.6805,-1.1271]; n=26|+4.1055 [+0.6323,+8.2942]; n=22|
|hc2|HALF|-1.2120 [-5.3814,+1.3175]; n=32|-0.4308 [-7.2655,+4.7811]; n=26|-6.0438 [-13.7723,+0.2965]; n=26|+5.1569 [+1.7187,+9.3271]; n=22|
|hc2|LN|-1.9401 [-6.4789,+0.8533]; n=32|-1.0110 [-8.8677,+5.7910]; n=26|-8.1243 [-16.6988,-0.9531]; n=26|+2.0175 [-0.8330,+5.3028]; n=22|
|vidstg|A|+2.3475 [-1.4658,+5.9260]; n=32|+11.0210 [+1.4961,+20.4945]; n=23|+1.8190 [-6.1231,+8.8930]; n=23|+16.8703 [+9.5573,+24.7893]; n=23|
|vidstg|SEL|+2.2945 [-1.5235,+5.8603]; n=32|+10.8318 [+1.3533,+20.3041]; n=23|+1.4623 [-6.4129,+8.6079]; n=23|+16.2744 [+9.3519,+23.5963]; n=23|
|vidstg|HALF|+3.0217 [+0.3747,+5.7936]; n=32|+12.7205 [+5.3816,+20.6237]; n=23|+4.2274 [-1.6589,+9.8635]; n=23|+16.2902 [+8.9400,+24.4405]; n=23|
|vidstg|LN|+1.7898 [-1.6069,+5.0680]; n=32|+8.5841 [+0.3255,+16.7530]; n=23|+1.4120 [-5.7832,+7.9182]; n=23|+11.4997 [+5.8836,+17.6480]; n=23|

|Target|Contrast|Full mvIoU (pp,95%CI)|Positive/negative/neutral/unknown parents|>5/>20pp harms|
|---|---|---|---|---|
|hc2|SEL_minus_A|-0.2986 [-0.6154,-0.0651]; n=32|8/17/7/0|0/0|
|hc2|SEL_minus_HALF|-0.7811 [-1.4843,-0.1997]; n=32|7/18/7/0|2/0|
|hc2|SEL_minus_LN|-0.0529 [-0.5158,+0.3829]; n=32|9/15/8/0|0/0|
|hc2|HALF_minus_A|+0.4825 [-0.0742,+1.1626]; n=32|17/8/7/0|0/0|
|hc2|LN_minus_A|-0.2457 [-0.7688,+0.2458]; n=32|12/13/7/0|0/0|
|vidstg|SEL_minus_A|-0.0530 [-0.2999,+0.1806]; n=32|8/12/12/0|0/0|
|vidstg|SEL_minus_HALF|-0.7272 [-2.2616,+0.6600]; n=32|9/12/11/0|2/0|
|vidstg|SEL_minus_LN|+0.5048 [-0.4138,+1.8827]; n=32|13/7/12/0|1/0|
|vidstg|HALF_minus_A|+0.6742 [-0.7636,+2.2284]; n=32|12/9/11/0|1/0|
|vidstg|LN_minus_A|-0.5578 [-1.9279,+0.3412]; n=32|7/14/11/0|1/0|

|Target|Arm|Retention of initially correct unvisited frames (%)|Correction of initially wrong unvisited frames (%)|Expert agreement gain pp|Mean instrumented fit seconds|
|---|---|---|---|---|---|
|hc2|A|+76.6300 [+61.3773,+90.2244]; n=26|+16.0227 [+7.1970,+26.3475]; n=22|+24.1663 [+16.4688,+33.1230]; n=32|2.285256|
|hc2|SEL|+76.8205 [+61.7867,+89.9862]; n=26|+15.6439 [+7.0452,+25.7687]; n=22|+22.9449 [+16.0082,+30.8557]; n=32|4.873083|
|hc2|HALF|+73.4066 [+57.6694,+87.5231]; n=26|+17.1212 [+7.9004,+27.6948]; n=22|+19.3401 [+13.7727,+25.8653]; n=32|2.351777|
|hc2|LN|+77.8205 [+62.5946,+91.2821]; n=26|+14.0530 [+5.5464,+24.2808]; n=22|+21.7909 [+15.3921,+29.1274]; n=32|2.444905|
|vidstg|A|+81.0425 [+64.8309,+95.2657]; n=23|+39.5217 [+23.9491,+55.3047]; n=23|+24.0263 [+18.3289,+30.0509]; n=25|0.569295|
|vidstg|SEL|+81.2713 [+64.8309,+95.2657]; n=23|+38.3701 [+22.5097,+54.4606]; n=23|+23.2744 [+17.9092,+28.8206]; n=25|1.086595|
|vidstg|HALF|+87.3809 [+73.9130,+98.4190]; n=23|+35.6211 [+20.3675,+51.3820]; n=23|+20.6289 [+15.9320,+25.7601]; n=25|0.570424|
|vidstg|LN|+83.4159 [+68.3397,+95.4278]; n=23|+33.6522 [+18.3986,+49.6598]; n=23|+20.8068 [+16.4207,+25.3900]; n=25|0.570238|

## Evidence and decision boundaries

Initially correct means entering real-event-frame IoU>=.5; initially wrong means <.5. Masks use the shared entering state and are defined only offline after the global seal. Real sampled GT frames exclude visited Omega0 for retention/correction; unsupported/interpolated frames are not imputed into these partitions. The full native interval is fixed across arms, so complete tIoU differences are zero. Unknown frame partitions remain unknown.

Protection, correction and task quality must be assessed together. A smaller drift or higher expert agreement does not prove correctness. Positive historical effects can justify specifying an independent replication; they do not certify a deployed method. Equivalence to HALF/LN requires a separate predeclared equivalence margin, not a confidence interval crossing zero. Simple-control comparisons are retained without a requirement to beat Capped Direct before this test.

Actual cost is instrumented complete-fit time including original native likelihood/action/gradient recording and independent arithmetic; it excludes frozen backbone capture, old expert generation and serialization. SEL additionally performs four actual complete-grid candidate decoder forwards per informative step. This is not an end-to-end deployment benchmark and does not imply zero fresh expert cost. Every parent timing, selected/fallback round and negative case is published.

The old local screen remains 4/9 below its locked50% gate. This is a separately authorized full-task test, not a retrospective screen pass. High-dimensional P/F numerical failures, ordinary R without a task test, negative-evidence scale confounds, Q0 candidate/mean and P9 weak-predictor uncertainties remain in EVIDENCE_SCOPE_CORRECTION.md. No other experiment was reopened.

CURRENT unchanged; cancelled-baseline controller/worker remain their original T standby identities, with accepted prefix and optimizer custody preserved. The user-deleted monitor stays deleted. No automatic candidate expansion, parameter search, order2, new teacher, paper or full-main launch.

## Actual bounded-task judgment

SEL minus A complete mvIoU is **HC -0.2986pp,95%CI[-0.6154,-0.0651]**, **Vid -0.0530pp,95%CI[-0.2999,+0.1806]**. HC SEL is also worse than HALF: -0.7811pp[-1.4843,-0.1997]. The current selector does not establish complete-task benefit; this is measured fixed-recipe negative evidence, not failure merely to clear the old local resource gate.

On initially correct unvisited real-event frames, HC IoU change is A -8.3708pp,SEL -8.1754pp,HALF -6.0438pp; each is a conditional mean among26 valid parents. On initially wrong frames, A correction is +4.7403pp,SEL +4.1055pp,HALF +5.1569pp among22 valid parents. The selector's small descriptive protection difference does not offset its reduced correction or task loss. On Vid, SEL weakens both correct-frame change (+1.4623 versus A +1.8190pp) and wrong-frame correction (+16.2744 versus A +16.8703pp), with23 valid parents per partition. All confidence intervals and direct paired contrasts are retained above and in SUMMARY.json; means alone do not establish these secondary mechanisms.

HALF minus A complete mvIoU is HC +0.4825pp[-0.0742,+1.1626],Vid +0.6742pp[-0.7636,+2.2284]. Both intervals include0. This suggests keeping step strength separate from selection complexity; it does not prove HALF's general superiority or equivalence to SEL/LN. No HALF promotion or additional experiment is automatic.

Across the original informative formal rounds, selected counts A/LN/Query/Half are698/484/14/44 on HC and139/87/5/19 on Vid. All440 no-information formal rounds, including280 rounds from28 empty fits and160 equal-weight rounds, are retained. Actual complete-fit instrumented time averages are A/SEL 2.2853/4.8731s on HC and0.5693/1.0866s on Vid. SEL adds cost while its primary task benefit is unestablished; these are measured instrumented fit times, not complete deployment latency.

EVIDENCE_SCOPE_CORRECTION.md records the assessment before this experiment. This result now moves this particular Real-Forward recipe from task utility unevaluated to fixed-recipe task benefit unestablished, with a harmful HC contrast. It does not retroactively change the old4/9 screen, invalidate ordinary R without a full task test, or launch Q0/P9/negative-evidence/P7 experiments.
