# Fixed-budget coarse-to-fine OPD

HC32/VidSTG32 already exposed development parents, deterministic parent-hash selection. Homogeneous caption-fixed Native-clock B sparse Teacher references, identical existing T4 final intervals. No new Teacher or Temporal fit. No diffusion training, new confirmation, CURRENT promotion or automatic expansion. Primary matched contrasts are small and all corresponding vIoU source-paired t95 intervals include zero; this experiment does not establish an advantage for descending order. Historical HC original OPD also has two extra possible steps, so its higher mean is not an isolated noise-order result.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@0.3 % | R@0.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|P0_F|28.686631|53.902845|51.365522|43.750000|15.625000|
|hc2|P0_O|29.529468|55.509946|51.365522|43.750000|15.625000|
|hc2|constant|29.254481|54.979861|51.365522|43.750000|15.625000|
|hc2|coarse_to_fine|29.257587|54.978077|51.365522|43.750000|15.625000|
|hc2|reverse|29.269224|54.983221|51.365522|43.750000|15.625000|
|vidstg|P0_F|11.043237|33.120715|34.579557|9.375000|0.000000|
|vidstg|P0_O|11.076720|32.389757|34.579557|9.375000|3.125000|
|vidstg|constant|11.116193|32.465341|34.579557|9.375000|3.125000|
|vidstg|coarse_to_fine|11.231233|32.788690|34.579557|9.375000|3.125000|
|vidstg|reverse|11.173997|32.706444|34.579557|9.375000|3.125000|

| Domain | Contrast | Metric | Delta pp | Source-paired t95 pp | + / - / neutral / unknown |
|---|---|---|---:|---|---|
|hc2|P0_O_minus_P0_F|m_vIoU|0.8428370543120115|[0.357612431996798, 1.328061676627225]|21/5/6/0|
|hc2|P0_O_minus_P0_F|m_sIoU_dense_GT|1.6071010764084557|[0.7387982538437525, 2.475403898973159]|20/6/6/0|
|hc2|P0_O_minus_P0_F|m_sIoU_anchor_GT|1.7599760711029024|[0.5589744153466367, 2.960977726859168]|16/6/4/6|
|hc2|P0_O_minus_P0_F|m_sIoU_native_nonanchor_GT|1.5757316395780472|[0.6895849383312375, 2.461878340824857]|20/6/6/0|
|hc2|P0_O_minus_P0_F|m_sIoU_interpolated_GT|1.6087814698996115|[0.7420410241458175, 2.475521915653405]|20/6/6/0|
|hc2|constant_minus_P0_F|m_vIoU|0.5678508264075106|[0.19897135708622937, 0.9367302957287919]|19/6/7/0|
|hc2|constant_minus_P0_F|m_sIoU_dense_GT|1.0770159606514573|[0.485225806370022, 1.6688061149328925]|19/8/5/0|
|hc2|constant_minus_P0_F|m_sIoU_anchor_GT|1.2494804240162969|[0.44064859401868495, 2.058312254013909]|16/6/4/6|
|hc2|constant_minus_P0_F|m_sIoU_native_nonanchor_GT|1.0509489595470585|[0.4471763892940125, 1.6547215298001046]|19/8/5/0|
|hc2|constant_minus_P0_F|m_sIoU_interpolated_GT|1.078848973541298|[0.4876733236684011, 1.670024623414195]|19/8/5/0|
|hc2|constant_minus_P0_O|m_vIoU|-0.2749862279045009|[-0.4292530249799943, -0.12071943082900749]|5/21/6/0|
|hc2|constant_minus_P0_O|m_sIoU_dense_GT|-0.5300851157569982|[-0.8429583827841071, -0.2172118487298894]|5/22/5/0|
|hc2|constant_minus_P0_O|m_sIoU_anchor_GT|-0.5104956470866054|[-0.9565708199636274, -0.06442047420958352]|6/16/4/6|
|hc2|constant_minus_P0_O|m_sIoU_native_nonanchor_GT|-0.5247826800309886|[-0.8399327647926966, -0.20963259526928057]|6/21/5/0|
|hc2|constant_minus_P0_O|m_sIoU_interpolated_GT|-0.5299324963583133|[-0.8425256226269611, -0.21733937008966547]|5/22/5/0|
|hc2|coarse_to_fine_minus_P0_F|m_vIoU|0.5709567213344238|[0.20267845645864685, 0.9392349862102007]|19/6/7/0|
|hc2|coarse_to_fine_minus_P0_F|m_sIoU_dense_GT|1.0752326782418111|[0.48319298359297513, 1.6672723728906471]|19/8/5/0|
|hc2|coarse_to_fine_minus_P0_F|m_sIoU_anchor_GT|1.2537401227023999|[0.44425693136770117, 2.0632233140370984]|16/6/4/6|
|hc2|coarse_to_fine_minus_P0_F|m_sIoU_native_nonanchor_GT|1.0498289109874233|[0.4450041260630818, 1.6546536959117648]|18/9/5/0|
|hc2|coarse_to_fine_minus_P0_F|m_sIoU_interpolated_GT|1.076963732992265|[0.4856596067933313, 1.6682678591911988]|19/8/5/0|
|hc2|coarse_to_fine_minus_P0_O|m_vIoU|-0.2718803329775877|[-0.42568673663494017, -0.11807392932023517]|5/21/6/0|
|hc2|coarse_to_fine_minus_P0_O|m_sIoU_dense_GT|-0.5318683981666446|[-0.846022159151363, -0.21771463718192607]|5/22/5/0|
|hc2|coarse_to_fine_minus_P0_O|m_sIoU_anchor_GT|-0.5062359484005026|[-0.9527248037714735, -0.05974709302953163]|6/16/4/6|
|hc2|coarse_to_fine_minus_P0_O|m_sIoU_native_nonanchor_GT|-0.5259027285906239|[-0.8421698027385977, -0.2096356544426502]|6/21/5/0|
|hc2|coarse_to_fine_minus_P0_O|m_sIoU_interpolated_GT|-0.5318177369073464|[-0.8456949828800232, -0.2179404909346695]|5/22/5/0|
|hc2|coarse_to_fine_minus_constant|m_vIoU|0.0031058949269132307|[-0.003942574880869677, 0.010154364734696138]|10/15/7/0|
|hc2|coarse_to_fine_minus_constant|m_sIoU_dense_GT|-0.0017832824096463556|[-0.01531964981752189, 0.011753084998229179]|14/13/5/0|
|hc2|coarse_to_fine_minus_constant|m_sIoU_anchor_GT|0.004259698686102931|[-0.00935660240172843, 0.01787599977393429]|12/10/4/6|
|hc2|coarse_to_fine_minus_constant|m_sIoU_native_nonanchor_GT|-0.0011200485596352897|[-0.013957097991466558, 0.011717000872195979]|13/14/5/0|
|hc2|coarse_to_fine_minus_constant|m_sIoU_interpolated_GT|-0.0018852405490331174|[-0.015545173289398936, 0.011774692191332701]|13/14/5/0|
|hc2|reverse_minus_P0_F|m_vIoU|0.5825931819568251|[0.23922874427536972, 0.9259576196382804]|21/4/7/0|
|hc2|reverse_minus_P0_F|m_sIoU_dense_GT|1.0803763740040837|[0.48961281258729383, 1.6711399354208736]|22/5/5/0|
|hc2|reverse_minus_P0_F|m_sIoU_anchor_GT|1.2396445462290193|[0.42109852544038007, 2.0581905670176583]|16/6/4/6|
|hc2|reverse_minus_P0_F|m_sIoU_native_nonanchor_GT|1.057907676720578|[0.44678019029803107, 1.6690351631431248]|21/6/5/0|
|hc2|reverse_minus_P0_F|m_sIoU_interpolated_GT|1.0823385159889911|[0.4937424693469613, 1.6709345626310208]|21/6/5/0|
|hc2|reverse_minus_P0_O|m_vIoU|-0.2602438723551864|[-0.4221696723190466, -0.09831807239132623]|6/20/6/0|
|hc2|reverse_minus_P0_O|m_sIoU_dense_GT|-0.5267247024043721|[-0.8309224614963348, -0.2225269433124094]|6/21/5/0|
|hc2|reverse_minus_P0_O|m_sIoU_anchor_GT|-0.5203315248738829|[-0.9520404251310829, -0.08862262461668291]|6/16/4/6|
|hc2|reverse_minus_P0_O|m_sIoU_native_nonanchor_GT|-0.5178239628574692|[-0.8237718002883736, -0.21187612542656492]|7/20/5/0|
|hc2|reverse_minus_P0_O|m_sIoU_interpolated_GT|-0.5264429539106203|[-0.8311024171253911, -0.22178349069584957]|6/21/5/0|
|hc2|reverse_minus_constant|m_vIoU|0.01474235554931448|[-0.057410776058687746, 0.08689548715731671]|15/10/7/0|
|hc2|reverse_minus_constant|m_sIoU_dense_GT|0.0033604133526261674|[-0.10184293369739744, 0.10856376040264978]|17/10/5/0|
|hc2|reverse_minus_constant|m_sIoU_anchor_GT|-0.009835877787277451|[-0.19998262080221763, 0.1803108652276627]|14/8/4/6|
|hc2|reverse_minus_constant|m_sIoU_native_nonanchor_GT|0.006958717173519385|[-0.09162342547601535, 0.10554085982305411]|15/12/5/0|
|hc2|reverse_minus_constant|m_sIoU_interpolated_GT|0.0034895424476929365|[-0.10269799493715558, 0.10967707983254145]|17/10/5/0|
|hc2|reverse_minus_coarse_to_fine|m_vIoU|0.01163646062240125|[-0.06060952552024169, 0.0838824467650442]|14/11/7/0|
|hc2|reverse_minus_coarse_to_fine|m_sIoU_dense_GT|0.005143695762272523|[-0.09999003987526665, 0.1102774313998117]|18/9/5/0|
|hc2|reverse_minus_coarse_to_fine|m_sIoU_anchor_GT|-0.014095576473380382|[-0.20247275229196401, 0.17428159934520324]|12/10/4/6|
|hc2|reverse_minus_coarse_to_fine|m_sIoU_native_nonanchor_GT|0.008078765733154674|[-0.090178626154378, 0.10633615762068735]|17/10/5/0|
|hc2|reverse_minus_coarse_to_fine|m_sIoU_interpolated_GT|0.005374782996726054|[-0.10081739515613167, 0.11156696114958378]|17/10/5/0|
|vidstg|P0_O_minus_P0_F|m_vIoU|0.0334829827805342|[-1.2414613804740353, 1.308427346035104]|9/7/16/0|
|vidstg|P0_O_minus_P0_F|m_sIoU_dense_GT|-0.7309581742745456|[-4.160079192654255, 2.6981628441051635]|13/9/10/0|
|vidstg|P0_O_minus_P0_F|m_sIoU_anchor_GT|-0.6894899406164868|[-4.75119198526207, 3.3722121040290958]|8/7/7/10|
|vidstg|P0_O_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.5757651317208368|[-3.9691470323978324, 2.8176167689561584]|12/10/10/0|
|vidstg|P0_O_minus_P0_F|m_sIoU_interpolated_GT|-0.7171214627114932|[-4.147078048426337, 2.7128351230033507]|13/8/11/0|
|vidstg|constant_minus_P0_F|m_vIoU|0.07295607565152301|[-1.2387611614308214, 1.3846733127338675]|10/7/15/0|
|vidstg|constant_minus_P0_F|m_sIoU_dense_GT|-0.6553743931767323|[-4.134690626273132, 2.823941839919667]|13/10/9/0|
|vidstg|constant_minus_P0_F|m_sIoU_anchor_GT|-0.6416811896757612|[-4.719288250782313, 3.4359258714307903]|8/8/6/10|
|vidstg|constant_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.5220952452435009|[-3.983429096941022, 2.9392386064540204]|13/10/9/0|
|vidstg|constant_minus_P0_F|m_sIoU_interpolated_GT|-0.6409159762368908|[-4.119866753051575, 2.838034800577793]|13/9/10/0|
|vidstg|constant_minus_P0_O|m_vIoU|0.03947309287098873|[-0.05932798928494888, 0.13827417502692635]|9/8/15/0|
|vidstg|constant_minus_P0_O|m_sIoU_dense_GT|0.07558378109781344|[-0.12090656504845954, 0.2720741272440864]|11/11/10/0|
|vidstg|constant_minus_P0_O|m_sIoU_anchor_GT|0.047808750940725855|[-0.3791457660189187, 0.4747632679003704]|9/7/6/10|
|vidstg|constant_minus_P0_O|m_sIoU_native_nonanchor_GT|0.05366988647733555|[-0.1397697877446048, 0.24710956069927592]|10/12/10/0|
|vidstg|constant_minus_P0_O|m_sIoU_interpolated_GT|0.0762054864746022|[-0.11857304934472036, 0.27098402229392476]|11/11/10/0|
|vidstg|coarse_to_fine_minus_P0_F|m_vIoU|0.1879963542386984|[-1.1259315144072741, 1.501924222884671]|10/7/15/0|
|vidstg|coarse_to_fine_minus_P0_F|m_sIoU_dense_GT|-0.33202564637000614|[-3.8067659063876054, 3.142714613647593]|14/9/9/0|
|vidstg|coarse_to_fine_minus_P0_F|m_sIoU_anchor_GT|-0.18235391350762456|[-4.15549851247211, 3.7907906854568605]|9/7/6/10|
|vidstg|coarse_to_fine_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.22294229102798468|[-3.668075394083725, 3.222190812027756]|13/10/9/0|
|vidstg|coarse_to_fine_minus_P0_F|m_sIoU_interpolated_GT|-0.3129591400661995|[-3.7906005989544846, 3.164682318822086]|13/8/11/0|
|vidstg|coarse_to_fine_minus_P0_O|m_vIoU|0.15451337145816413|[-0.06411001965128021, 0.37313676256760847]|11/5/16/0|
|vidstg|coarse_to_fine_minus_P0_O|m_sIoU_dense_GT|0.39893252790453965|[-0.08671195288251193, 0.8845770086915912]|12/7/13/0|
|vidstg|coarse_to_fine_minus_P0_O|m_sIoU_anchor_GT|0.5071360271088623|[-0.1644755192011722, 1.1787475734188968]|11/4/7/10|
|vidstg|coarse_to_fine_minus_P0_O|m_sIoU_native_nonanchor_GT|0.35282284069285197|[-0.11247484728280976, 0.8181205286685137]|12/7/13/0|
|vidstg|coarse_to_fine_minus_P0_O|m_sIoU_interpolated_GT|0.4041623226452938|[-0.08478737760889798, 0.8931120228994855]|11/8/13/0|
|vidstg|coarse_to_fine_minus_constant|m_vIoU|0.11504027858717539|[-0.08348855720229847, 0.31356911437664925]|8/9/15/0|
|vidstg|coarse_to_fine_minus_constant|m_sIoU_dense_GT|0.3233487468067262|[-0.12593545348115998, 0.7726329470946124]|9/13/10/0|
|vidstg|coarse_to_fine_minus_constant|m_sIoU_anchor_GT|0.45932727616813646|[-0.3487439461602954, 1.2673984984965683]|7/9/6/10|
|vidstg|coarse_to_fine_minus_constant|m_sIoU_native_nonanchor_GT|0.2991529542155164|[-0.11754684682610483, 0.7158527552571377]|11/11/10/0|
|vidstg|coarse_to_fine_minus_constant|m_sIoU_interpolated_GT|0.3279568361706916|[-0.12710459881082597, 0.7830182711522091]|9/13/10/0|
|vidstg|reverse_minus_P0_F|m_vIoU|0.1307596991899176|[-1.1495497721915668, 1.411069170571402]|11/7/14/0|
|vidstg|reverse_minus_P0_F|m_sIoU_dense_GT|-0.41427167052607106|[-3.969370230215391, 3.140826889163249]|14/10/8/0|
|vidstg|reverse_minus_P0_F|m_sIoU_anchor_GT|-0.40043774062145215|[-4.747863289885857, 3.9469878086429526]|8/8/6/10|
|vidstg|reverse_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.42127458756272657|[-3.973460886675861, 3.130911711550408]|13/11/8/0|
|vidstg|reverse_minus_P0_F|m_sIoU_interpolated_GT|-0.3825286604010372|[-3.9389828833901923, 3.173925562588118]|14/9/9/0|
|vidstg|reverse_minus_P0_O|m_vIoU|0.09727671640938332|[-0.22424289484910673, 0.41879632766787334]|12/6/14/0|
|vidstg|reverse_minus_P0_O|m_sIoU_dense_GT|0.3166865037484745|[-0.5421838269084519, 1.175556834405401]|12/11/9/0|
|vidstg|reverse_minus_P0_O|m_sIoU_anchor_GT|0.28905219999503456|[-0.9305794413507089, 1.5086838413407782]|9/7/6/10|
|vidstg|reverse_minus_P0_O|m_sIoU_native_nonanchor_GT|0.15449054415811|[-0.6008466162193162, 0.9098277045355363]|13/10/9/0|
|vidstg|reverse_minus_P0_O|m_sIoU_interpolated_GT|0.334592802310456|[-0.5445660364421487, 1.2137516410630607]|12/11/9/0|
|vidstg|reverse_minus_constant|m_vIoU|0.05780362353839461|[-0.23079145536302995, 0.3463987024398192]|11/7/14/0|
|vidstg|reverse_minus_constant|m_sIoU_dense_GT|0.2411027226506611|[-0.5498034068091948, 1.032008852110517]|13/10/9/0|
|vidstg|reverse_minus_constant|m_sIoU_anchor_GT|0.24124344905430864|[-0.8806910338015607, 1.363177931910178]|8/8/6/10|
|vidstg|reverse_minus_constant|m_sIoU_native_nonanchor_GT|0.10082065768077446|[-0.5674572905766946, 0.7690986059382434]|13/10/9/0|
|vidstg|reverse_minus_constant|m_sIoU_interpolated_GT|0.2583873158358537|[-0.55710333886765, 1.0738779705393573]|13/10/9/0|
|vidstg|reverse_minus_coarse_to_fine|m_vIoU|-0.05723665504878078|[-0.3276528221478916, 0.21317951205033003]|11/7/14/0|
|vidstg|reverse_minus_coarse_to_fine|m_sIoU_dense_GT|-0.08224602415606512|[-0.7598966458643314, 0.5954045975522012]|12/11/9/0|
|vidstg|reverse_minus_coarse_to_fine|m_sIoU_anchor_GT|-0.2180838271138278|[-1.312659475707103, 0.8764918214794476]|8/8/6/10|
|vidstg|reverse_minus_coarse_to_fine|m_sIoU_native_nonanchor_GT|-0.19833229653474194|[-0.815338619126517, 0.4186740260570331]|10/13/9/0|
|vidstg|reverse_minus_coarse_to_fine|m_sIoU_interpolated_GT|-0.06956952033483785|[-0.7597817639204704, 0.6206427232507946]|12/11/9/0|

Primary comparison is descending against ascending and constant, all with three attempts and identical reject-rollback-continue semantics. Descending and ascending share the same three scales and standard-noise tensors per scale. Constant matches mean squared sigma. Original HC5/Vid3 early-stop OPD is historical only; differences from it combine retry/step/scale changes. Every stage recenters on actual Student output, and stage diagnostics report actual logit-anchor motion, protected-grid box motion and high-initial-pseudoreference anchor degradation. Smaller noise does not guarantee smaller actual movement or true-object correctness. No accepted bad step is allowed through the original sparse reference and protection gates.

All64 predictions and full-prefix decoder readouts are sealed and independently audited before GT scoring. GT supports are identical between arms; anchor, other native-grid and interpolated known-GT strata retain unknowns. Metrics do not establish semantic identity or human review. Pairwise t95 intervals are descriptive development uncertainty, not a new selection confirmation. Source/feature reuse costs and actual new stage/solver costs are separated; peak memory/fresh end-to-end speedup remain unknown. No private media, GT coordinates, query/Caption text, raw Teacher/tensors/weights, credentials or personal conversations are exported.

Actual settings are in ACTUAL_CONFIGURATION.json; fixed design and execution limitations are in the accompanying protocol.
