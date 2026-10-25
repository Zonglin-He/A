# Current no-shrinkage own-K budget: vidstg

Complete128query/128parent historical fixed panel,8 independent configurations/query, whole seal→1152officialdensechecksmax1.2212453270876722e-15→256predreceiptSHA checks plus parent/teacher/rawgrid hashes and all arrays rederived exactly/all means. These are1024budgetconfigurations/domain, not the full12205-query budget study. Current kappa0 choice followed exposed full results; no independent-unseen confirmation claim.

|Schedule|K|tIoU %|vIoU %|R.3 %|R.5 %|Original expert forward s/query|Native-source severe t/v/union|
|---|---:|---:|---:|---:|---:|---:|---|
|Hybrid|2|35.238570|13.048670|17.187500|7.031250|0.056671|18/22/34|
|Uniform|2|36.023010|12.239226|16.406250|7.812500|0.056889|16/30/36|
|Hybrid|4|35.542752|14.422615|19.531250|10.156250|0.098214|15/24/32|
|Uniform|4|35.062631|13.642303|18.750000|10.156250|0.097979|22/25/37|
|Hybrid|8|34.677709|15.427777|20.312500|10.156250|0.181962|20/22/34|
|Uniform|8|35.367545|14.879433|19.531250|9.375000|0.181769|16/23/33|
|Hybrid|16|35.063852|15.640462|19.531250|10.156250|0.360705|15/22/29|
|Uniform|16|35.417928|15.459760|18.750000|10.156250|0.360741|18/22/31|

EveryK uses its own original joint teacher boxes/masks/context and original raw grids; rebuild E/geometry/auxDP, genuinely repool, kappa0 posterior/MAP/newintervalDP. Hidden old kappa64 per-K consistency checks remain bitwise prerequisites; Hybrid8 is exactly current full NoShrink on the same queries. No K8 subset pretending to be smallK. Restore original Torch4 only during derive, otherstages1thread, sameCPU14-15. DefaultK8 is not changed from these curves. Absolute values must not be compared to full-target means.

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|Hybrid2-Source|+3.117035 [0.887592, 5.568787]|-0.274232 [-2.292305, 1.763247]|18/22/34|46|
|Uniform2-Source|+3.901475 [1.450308, 6.547648]|-1.083676 [-3.436027, 1.171968]|16/30/36|54|
|Hybrid4-Source|+3.421217 [1.122879, 5.950584]|+1.099713 [-0.999005, 3.258164]|15/24/32|55|
|Uniform4-Source|+2.941096 [0.633198, 5.487709]|+0.319400 [-1.937263, 2.620548]|22/25/37|55|
|Hybrid8-Source|+2.556174 [0.254528, 5.113013]|+2.104875 [-0.108344, 4.344650]|20/22/34|52|
|Uniform8-Source|+3.246010 [0.986527, 5.674450]|+1.556530 [-0.804191, 3.993085]|16/23/33|55|
|Hybrid16-Source|+2.942317 [0.712841, 5.418892]|+2.317560 [0.009504, 4.755043]|15/22/29|57|
|Uniform16-Source|+3.296394 [1.024642, 5.782874]|+2.136858 [-0.036935, 4.409790]|18/22/31|58|
|Hybrid2-Hybrid8|+0.560861 [-0.348982, 1.443303]|-2.379107 [-3.864290, -1.131364]|9/25/28|20|
|Hybrid4-Hybrid8|+0.865043 [-0.144120, 1.920646]|-1.005162 [-2.069981, 0.024652]|9/15/20|21|
|Hybrid16-Hybrid8|+0.386143 [-0.586147, 1.342962]|+0.212685 [-0.738294, 1.141282]|9/7/14|20|
|Uniform2-Hybrid2|+0.784440 [-0.453311, 2.099178]|-0.809444 [-2.405331, 0.575594]|12/17/23|27|
|Uniform4-Hybrid4|-0.480120 [-1.681658, 0.621553]|-0.780312 [-2.318584, 0.501777]|15/15/26|22|
|Uniform8-Hybrid8|+0.689836 [-0.181528, 1.527071]|-0.548345 [-2.079864, 0.703867]|8/11/17|21|
|Uniform16-Hybrid16|+0.354077 [-0.356763, 1.143402]|-0.180702 [-0.820570, 0.397441]|6/8/11|12|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / Hybrid2-Source|78|-0.515832 [-2.398081, 1.292086]|-1.144308 [-2.612709, 0.248396]|
|short_le5s / Uniform2-Source|78|-0.149584 [-2.298263, 1.975571]|-1.532109 [-3.303293, 0.168058]|
|short_le5s / Hybrid4-Source|78|-0.391353 [-2.350441, 1.453010]|-0.139195 [-1.833492, 1.555978]|
|short_le5s / Uniform4-Source|78|-0.910006 [-2.772049, 0.937837]|-0.636886 [-2.739123, 1.476110]|
|short_le5s / Hybrid8-Source|78|-1.235087 [-3.237156, 0.713222]|+0.223262 [-1.634495, 2.151168]|
|short_le5s / Uniform8-Source|78|-0.509659 [-2.505746, 1.414177]|+0.447686 [-1.649181, 2.778579]|
|short_le5s / Hybrid16-Source|78|-0.408249 [-2.460447, 1.558787]|+0.907931 [-1.212219, 3.241156]|
|short_le5s / Uniform16-Source|78|-0.580120 [-2.474627, 1.285344]|+0.478484 [-1.393887, 2.400220]|
|short_le5s / Hybrid2-Hybrid8|78|+0.719254 [-0.218637, 1.730853]|-1.367570 [-2.483470, -0.362603]|
|short_le5s / Hybrid4-Hybrid8|78|+0.843733 [-0.257495, 1.979293]|-0.362457 [-1.480212, 0.737116]|
|short_le5s / Hybrid16-Hybrid8|78|+0.826837 [-0.187585, 1.959899]|+0.684669 [-0.356035, 1.865153]|
|short_le5s / Uniform2-Hybrid2|78|+0.366249 [-0.965032, 1.734767]|-0.387801 [-1.555551, 0.780307]|
|short_le5s / Uniform4-Hybrid4|78|-0.518653 [-1.482308, 0.440448]|-0.497691 [-1.721936, 0.642465]|
|short_le5s / Uniform8-Hybrid8|78|+0.725428 [-0.040201, 1.540468]|+0.224424 [-0.692546, 1.382536]|
|short_le5s / Uniform16-Hybrid16|78|-0.171871 [-0.882183, 0.526483]|-0.429447 [-1.266532, 0.254116]|
|long_gt5s / Hybrid2-Source|50|+8.784309 [4.260999, 13.996220]|+1.083085 [-3.670443, 5.784376]|
|long_gt5s / Uniform2-Source|50|+10.221126 [5.430588, 15.605829]|-0.384122 [-5.659687, 4.854871]|
|long_gt5s / Hybrid4-Source|50|+9.368826 [4.656075, 14.743097]|+3.032408 [-1.716381, 7.796966]|
|long_gt5s / Uniform4-Source|50|+8.948816 [4.356852, 14.242019]|+1.811207 [-3.014924, 6.783048]|
|long_gt5s / Hybrid8-Source|50|+8.470540 [3.889644, 13.774641]|+5.040190 [0.204868, 9.857605]|
|long_gt5s / Uniform8-Source|50|+9.104854 [4.680091, 14.115570]|+3.286327 [-1.756543, 8.254948]|
|long_gt5s / Hybrid16-Source|50|+8.169200 [3.679076, 13.370091]|+4.516580 [-0.400428, 9.610712]|
|long_gt5s / Uniform16-Source|50|+9.343755 [4.749459, 14.515836]|+4.723921 [-0.113965, 9.710826]|
|long_gt5s / Hybrid2-Hybrid8|50|+0.313769 [-1.520838, 1.868747]|-3.957105 [-7.358444, -1.176807]|
|long_gt5s / Hybrid4-Hybrid8|50|+0.898286 [-1.176980, 2.903446]|-2.007782 [-4.102680, -0.082443]|
|long_gt5s / Hybrid16-Hybrid8|50|-0.301340 [-2.231349, 1.462172]|-0.523611 [-2.278796, 0.983263]|
|long_gt5s / Uniform2-Hybrid2|50|+1.436818 [-0.914159, 4.073705]|-1.467207 [-5.220493, 1.551727]|
|long_gt5s / Uniform4-Hybrid4|50|-0.420009 [-3.040580, 1.946307]|-1.221201 [-4.540924, 1.485427]|
|long_gt5s / Uniform8-Hybrid8|50|+0.634313 [-1.302354, 2.434060]|-1.753864 [-5.275391, 0.921729]|
|long_gt5s / Uniform16-Hybrid16|50|+1.174555 [-0.242441, 2.860173]|+0.207341 [-0.752090, 1.238354]|

Pointwise parent-cluster CIs, no familywise correction. Allsigned positive/negative losses/rescues are retained. Source is native per-offset MAP+hull. Not every metric/schedule must increase with K; all exceptions/CIs crossing0 remain in tables. Originalexpert forward is only one nested component of deployment cost; source acquisition/requests/transport/startup are not zero or inferable from cachedCPUwall. No newSource/expert/GTscoring in this root export. Source/teacher costs inherited; current worker costs in cost_segments.json. PrivatePNG/PDF must be viewed by root before curve publication.
