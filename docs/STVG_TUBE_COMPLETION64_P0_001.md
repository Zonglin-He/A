# Sparse Teacher reference to complete Tube to Student: P0 feasibility

HC32/VidSTG32 already exposed development parents, deterministic parent-hash selection. Homogeneous caption-fixed Native-clock B sparse Teacher references, identical existing T4 final intervals. No new Teacher or Temporal fit. No diffusion training, new confirmation, CURRENT promotion or automatic expansion. The fixed deterministic completion does not improve on original sparse OPD: Student writeback minus original sparse OPD is -0.652194pp on HC and -0.162072pp on VidSTG (paired t95 [-1.067719,-0.236669] and [-1.000444,0.676300]). The external proposal is also below same-time Source on both means. This does not support replacing the sparse method or automatically starting denoiser training, and does not establish that all learned completion methods are ineffective.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@0.3 % | R@0.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|P0_F|28.686631|53.902845|51.365522|43.750000|15.625000|
|hc2|P0_O|29.529468|55.509946|51.365522|43.750000|15.625000|
|hc2|P0_R|25.332264|49.498952|51.365522|40.625000|9.375000|
|hc2|P0_R_to_S|28.877274|54.139496|51.365522|43.750000|15.625000|
|vidstg|P0_F|11.043237|33.120715|34.579557|9.375000|0.000000|
|vidstg|P0_O|11.076720|32.389757|34.579557|9.375000|3.125000|
|vidstg|P0_R|10.467268|32.043357|34.579557|12.500000|0.000000|
|vidstg|P0_R_to_S|10.914648|32.202999|34.579557|9.375000|0.000000|

| Domain | Contrast | Metric | Delta pp | Source-paired t95 pp | + / - / neutral / unknown |
|---|---|---|---:|---|---|
|hc2|P0_O_minus_P0_F|m_vIoU|0.8428370543120115|[0.357612431996798, 1.328061676627225]|21/5/6/0|
|hc2|P0_O_minus_P0_F|m_sIoU_dense_GT|1.6071010764084557|[0.7387982538437525, 2.475403898973159]|20/6/6/0|
|hc2|P0_O_minus_P0_F|m_sIoU_anchor_GT|1.7599760711029024|[0.5589744153466367, 2.960977726859168]|16/6/4/6|
|hc2|P0_O_minus_P0_F|m_sIoU_native_nonanchor_GT|1.5757316395780472|[0.6895849383312375, 2.461878340824857]|20/6/6/0|
|hc2|P0_O_minus_P0_F|m_sIoU_interpolated_GT|1.6087814698996115|[0.7420410241458175, 2.475521915653405]|20/6/6/0|
|hc2|P0_R_minus_P0_F|m_vIoU|-3.3543666967988894|[-6.623182216522247, -0.08555117707553217]|6/15/11/0|
|hc2|P0_R_minus_P0_F|m_sIoU_dense_GT|-4.403893086609607|[-8.288593369982554, -0.5191928032366602]|6/16/10/0|
|hc2|P0_R_minus_P0_F|m_sIoU_anchor_GT|-15.456678580384876|[-24.569729038925725, -6.343628121844027]|3/16/7/6|
|hc2|P0_R_minus_P0_F|m_sIoU_native_nonanchor_GT|-3.6251163384071785|[-7.138865626308899, -0.11136705050545759]|6/16/10/0|
|hc2|P0_R_minus_P0_F|m_sIoU_interpolated_GT|-4.386944777990455|[-8.280929173053607, -0.49296038292730193]|7/15/10/0|
|hc2|P0_R_minus_P0_O|m_vIoU|-4.197203751110901|[-7.666083947296381, -0.7283235549254199]|4/24/4/0|
|hc2|P0_R_minus_P0_O|m_sIoU_dense_GT|-6.010994163018062|[-10.031961810380047, -1.990026515656079]|4/25/3/0|
|hc2|P0_R_minus_P0_O|m_sIoU_anchor_GT|-17.21665465148778|[-26.314744448348208, -8.118564854627353]|4/21/1/6|
|hc2|P0_R_minus_P0_O|m_sIoU_native_nonanchor_GT|-5.200847977985226|[-8.908477717318172, -1.4932182386522794]|3/26/3/0|
|hc2|P0_R_minus_P0_O|m_sIoU_interpolated_GT|-5.995726247890067|[-10.024136272008242, -1.9673162237718929]|4/25/3/0|
|hc2|P0_R_to_S_minus_P0_F|m_vIoU|0.19064296019812238|[0.03302328604637125, 0.3482626343498735]|13/6/13/0|
|hc2|P0_R_to_S_minus_P0_F|m_sIoU_dense_GT|0.2366516293706017|[-0.058052086742920506, 0.5313553454841239]|13/7/12/0|
|hc2|P0_R_to_S_minus_P0_F|m_sIoU_anchor_GT|0.1636964145824783|[-0.30651486741384076, 0.6339076965787973]|8/8/10/6|
|hc2|P0_R_to_S_minus_P0_F|m_sIoU_native_nonanchor_GT|0.23490088271470172|[-0.06655145055019179, 0.5363532159795952]|13/7/12/0|
|hc2|P0_R_to_S_minus_P0_F|m_sIoU_interpolated_GT|0.2369679194477584|[-0.05801072112922728, 0.5319465600247442]|14/6/12/0|
|hc2|P0_R_to_S_minus_P0_O|m_vIoU|-0.6521940941138891|[-1.0677188473874566, -0.2366693408403216]|5/23/4/0|
|hc2|P0_R_to_S_minus_P0_O|m_sIoU_dense_GT|-1.3704494470378539|[-2.1933297348403524, -0.5475691592353551]|5/24/3/0|
|hc2|P0_R_to_S_minus_P0_O|m_sIoU_anchor_GT|-1.596279656520424|[-2.69405795324881, -0.498501359792038]|4/19/3/6|
|hc2|P0_R_to_S_minus_P0_O|m_sIoU_native_nonanchor_GT|-1.3408307568633453|[-2.171598641716919, -0.5100628720097713]|5/24/3/0|
|hc2|P0_R_to_S_minus_P0_O|m_sIoU_interpolated_GT|-1.371813550451853|[-2.194395728664845, -0.5492313722388608]|6/23/3/0|
|hc2|P0_R_to_S_minus_P0_R|m_vIoU|3.5450096569970118|[0.24251621662248724, 6.847503097371536]|16/5/11/0|
|hc2|P0_R_to_S_minus_P0_R|m_sIoU_dense_GT|4.640544715980208|[0.7853270168471829, 8.495762415113234]|16/6/10/0|
|hc2|P0_R_to_S_minus_P0_R|m_sIoU_anchor_GT|15.620374994967355|[6.695739298837941, 24.54501069109677]|16/3/7/6|
|hc2|P0_R_to_S_minus_P0_R|m_sIoU_native_nonanchor_GT|3.8600172211218804|[0.3515885783300554, 7.368445863913705]|17/5/10/0|
|hc2|P0_R_to_S_minus_P0_R|m_sIoU_interpolated_GT|4.623912697438214|[0.7623748432111173, 8.48545055166531]|16/6/10/0|
|vidstg|P0_O_minus_P0_F|m_vIoU|0.0334829827805342|[-1.2414613804740353, 1.308427346035104]|9/7/16/0|
|vidstg|P0_O_minus_P0_F|m_sIoU_dense_GT|-0.7309581742745456|[-4.160079192654255, 2.6981628441051635]|13/9/10/0|
|vidstg|P0_O_minus_P0_F|m_sIoU_anchor_GT|-0.6894899406164868|[-4.75119198526207, 3.3722121040290958]|8/7/7/10|
|vidstg|P0_O_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.5757651317208368|[-3.9691470323978324, 2.8176167689561584]|12/10/10/0|
|vidstg|P0_O_minus_P0_F|m_sIoU_interpolated_GT|-0.7171214627114932|[-4.147078048426337, 2.7128351230033507]|13/8/11/0|
|vidstg|P0_R_minus_P0_F|m_vIoU|-0.5759687292888918|[-2.252542038250194, 1.1006045796724109]|8/8/16/0|
|vidstg|P0_R_minus_P0_F|m_sIoU_dense_GT|-1.0773581933853673|[-3.676465654817271, 1.5217492680465365]|9/9/14/0|
|vidstg|P0_R_minus_P0_F|m_sIoU_anchor_GT|-4.78212185841444|[-13.428243126976469, 3.8639994101475885]|7/9/6/10|
|vidstg|P0_R_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.5576505740054386|[-2.5844208830968016, 1.4691197350859246]|9/9/14/0|
|vidstg|P0_R_minus_P0_F|m_sIoU_interpolated_GT|-1.0758133865678108|[-3.6899962920750013, 1.5383695189393798]|9/8/15/0|
|vidstg|P0_R_minus_P0_O|m_vIoU|-0.609451712069426|[-2.5220792482142054, 1.3031758240753535]|8/12/12/0|
|vidstg|P0_R_minus_P0_O|m_sIoU_dense_GT|-0.34640001911082174|[-4.468278358720583, 3.7754783204989395]|10/16/6/0|
|vidstg|P0_R_minus_P0_O|m_sIoU_anchor_GT|-4.092631917797953|[-11.824081704181616, 3.6388178685857087]|7/12/3/10|
|vidstg|P0_R_minus_P0_O|m_sIoU_native_nonanchor_GT|0.018114557715397876|[-3.9118585672182147, 3.9480876826490108]|13/13/6/0|
|vidstg|P0_R_minus_P0_O|m_sIoU_interpolated_GT|-0.3586919238563175|[-4.49195668631906, 3.7745728386064257]|9/16/7/0|
|vidstg|P0_R_to_S_minus_P0_F|m_vIoU|-0.12858873603197965|[-1.1177055216340945, 0.8605280495701352]|6/4/22/0|
|vidstg|P0_R_to_S_minus_P0_F|m_sIoU_dense_GT|-0.917716421494855|[-4.263448947400066, 2.4280161044103568]|8/6/18/0|
|vidstg|P0_R_to_S_minus_P0_F|m_sIoU_anchor_GT|0.00040756977936757204|[-3.644167236976216, 3.6449823765349514]|6/4/12/10|
|vidstg|P0_R_to_S_minus_P0_F|m_sIoU_native_nonanchor_GT|-0.725782565480379|[-3.8667113531784545, 2.4151462222176963]|10/4/18/0|
|vidstg|P0_R_to_S_minus_P0_F|m_sIoU_interpolated_GT|-0.9419854474595051|[-4.318808496141478, 2.4348376012224673]|8/5/19/0|
|vidstg|P0_R_to_S_minus_P0_O|m_vIoU|-0.16207171881251392|[-1.0004438925459043, 0.6763004549208764]|7/9/16/0|
|vidstg|P0_R_to_S_minus_P0_O|m_sIoU_dense_GT|-0.18675824722030931|[-1.7721566518255831, 1.3986401573849645]|7/15/10/0|
|vidstg|P0_R_to_S_minus_P0_O|m_sIoU_anchor_GT|0.6898975103958545|[-1.869382696417772, 3.2491777172094807]|6/9/7/10|
|vidstg|P0_R_to_S_minus_P0_O|m_sIoU_native_nonanchor_GT|-0.1500174337595425|[-1.7187451706771415, 1.4187103031580564]|9/13/10/0|
|vidstg|P0_R_to_S_minus_P0_O|m_sIoU_interpolated_GT|-0.22486398474801186|[-1.8061002658292158, 1.3563722963331921]|7/15/10/0|
|vidstg|P0_R_to_S_minus_P0_R|m_vIoU|0.44737999325691213|[-1.510091401915152, 2.4048513884289764]|9/7/16/0|
|vidstg|P0_R_to_S_minus_P0_R|m_sIoU_dense_GT|0.15964177189051235|[-4.182940156424228, 4.502223700205252]|10/10/12/0|
|vidstg|P0_R_to_S_minus_P0_R|m_sIoU_anchor_GT|4.782529428193808|[-3.8527886955990436, 13.41784755198666]|9/7/6/10|
|vidstg|P0_R_to_S_minus_P0_R|m_sIoU_native_nonanchor_GT|-0.16813199147494057|[-4.016236270583209, 3.679972287633328]|12/8/12/0|
|vidstg|P0_R_to_S_minus_P0_R|m_sIoU_interpolated_GT|0.13382793910830568|[-4.243175231509743, 4.510831109726355]|10/9/13/0|

P0_R is an external deterministic regularized residual proposal, not parameter TTA. P0_R_to_S is a genuine 1792-parameter Student fit through original Gaussian OPD/Adam/M2 with fixed sigma and original early stop. Original sparse Teacher acceptance and full-grid-minus-original-sparse protection are retained even though dense reward targets cover the full native grid. Missing references or invalid logit chart produce exact Source whole-query no-ops, retained in the denominator. Leave-one-reference checks use pseudoreferences and are not GT evidence. Dense reward/solver costs differ from sparse OPD; no equal-budget speed claim.

All64 predictions and full-prefix decoder readouts are sealed and independently audited before GT scoring. GT supports are identical between arms; anchor, other native-grid and interpolated known-GT strata retain unknowns. Metrics do not establish semantic identity or human review. Pairwise t95 intervals are descriptive development uncertainty, not a new selection confirmation. Source/feature reuse costs and actual new stage/solver costs are separated; peak memory/fresh end-to-end speedup remain unknown. No private media, GT coordinates, query/Caption text, raw Teacher/tensors/weights, credentials or personal conversations are exported.

Actual settings are in ACTUAL_CONFIGURATION.json; fixed design and execution limitations are in the accompanying protocol.
