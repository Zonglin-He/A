# Current no-shrinkage own-K budget: hc2

Complete128query/128parent historical fixed panel,8 independent configurations/query, whole seal→1152officialdensechecksmax8.8817841970012523e-16→256predreceiptSHA checks plus parent/teacher/rawgrid hashes and all arrays rederived exactly/all means. These are1024budgetconfigurations/domain, not the full12205-query budget study. Current kappa0 choice followed exposed full results; no independent-unseen confirmation claim.

|Schedule|K|tIoU %|vIoU %|R.3 %|R.5 %|Original expert forward s/query|Native-source severe t/v/union|
|---|---:|---:|---:|---:|---:|---:|---|
|Hybrid|2|53.607280|27.973843|42.968750|20.312500|0.064057|22/22/35|
|Uniform|2|54.507904|28.956413|46.093750|21.875000|0.059465|22/22/35|
|Hybrid|4|53.453941|29.482507|47.656250|22.656250|0.105522|21/19/32|
|Uniform|4|53.277307|27.572667|45.312500|19.531250|0.102314|25/23/37|
|Hybrid|8|54.377109|31.502183|48.437500|23.437500|0.193621|23/19/34|
|Uniform|8|53.330792|30.258326|49.218750|21.875000|0.190714|23/16/30|
|Hybrid|16|54.245473|32.709588|51.562500|23.437500|0.378321|19/11/24|
|Uniform|16|54.553919|31.240555|46.875000|21.093750|0.375483|22/18/32|

EveryK uses its own original joint teacher boxes/masks/context and original raw grids; rebuild E/geometry/auxDP, genuinely repool, kappa0 posterior/MAP/newintervalDP. Hidden old kappa64 per-K consistency checks remain bitwise prerequisites; Hybrid8 is exactly current full NoShrink on the same queries. No K8 subset pretending to be smallK. Restore original Torch4 only during derive, otherstages1thread, sameCPU14-15. DefaultK8 is not changed from these curves. Absolute values must not be compared to full-target means.

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|Hybrid2-Source|+7.097803 [3.930618, 10.292686]|+2.849590 [-0.164002, 5.788533]|22/22/35|69|
|Uniform2-Source|+7.998426 [5.011547, 11.026670]|+3.832160 [0.968300, 6.651417]|22/22/35|69|
|Hybrid4-Source|+6.944463 [3.842018, 10.070420]|+4.358254 [1.803619, 6.956678]|21/19/32|72|
|Uniform4-Source|+6.767829 [3.650879, 9.855118]|+2.448414 [-0.810154, 5.542451]|25/23/37|73|
|Hybrid8-Source|+7.867631 [4.711738, 11.028059]|+6.377931 [3.509068, 9.211525]|23/19/34|74|
|Uniform8-Source|+6.821314 [3.747020, 9.939320]|+5.134073 [2.446703, 7.717014]|23/16/30|72|
|Hybrid16-Source|+7.735996 [4.720919, 10.832901]|+7.585335 [5.059731, 10.190960]|19/11/24|74|
|Uniform16-Source|+8.044441 [5.123274, 11.061265]|+6.116302 [3.541136, 8.703595]|22/18/32|75|
|Hybrid2-Hybrid8|-0.769829 [-2.078892, 0.428283]|-3.528341 [-5.717788, -1.470432]|18/36/41|26|
|Hybrid4-Hybrid8|-0.923168 [-2.128273, 0.239401]|-2.019676 [-3.929944, -0.237098]|21/32/39|19|
|Hybrid16-Hybrid8|-0.131636 [-1.196437, 1.155860]|+1.207405 [0.059583, 2.499472]|14/12/19|26|
|Uniform2-Hybrid2|+0.900623 [-0.573509, 2.582492]|+0.982571 [-1.702005, 3.707137]|17/25/36|42|
|Uniform4-Hybrid4|-0.176635 [-1.546213, 1.208761]|-1.909840 [-4.436659, 0.505810]|16/30/37|31|
|Uniform8-Hybrid8|-1.046317 [-2.154604, -0.066371]|-1.243858 [-2.376951, -0.197802]|18/21/31|19|
|Uniform16-Hybrid16|+0.308445 [-0.515176, 1.190812]|-1.469033 [-2.564945, -0.428350]|9/20/25|17|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / Hybrid2-Source|42|+2.751119 [-2.326396, 7.723860]|+1.723377 [-1.858272, 5.284640]|
|short_le5s / Uniform2-Source|42|+3.209695 [-1.159882, 7.575799]|+0.833427 [-2.749231, 4.387159]|
|short_le5s / Hybrid4-Source|42|+2.193058 [-2.537200, 6.834402]|+1.590068 [-2.118812, 5.316477]|
|short_le5s / Uniform4-Source|42|+1.805253 [-3.569081, 6.922760]|+0.144962 [-4.524846, 4.530553]|
|short_le5s / Hybrid8-Source|42|+2.716859 [-1.895399, 7.274408]|+3.438325 [-0.116410, 7.148047]|
|short_le5s / Uniform8-Source|42|+2.443386 [-2.505781, 7.245623]|+2.577715 [-0.790936, 6.112491]|
|short_le5s / Hybrid16-Source|42|+2.726955 [-2.223984, 7.650616]|+5.067078 [1.632486, 8.923797]|
|short_le5s / Uniform16-Source|42|+3.022581 [-1.893978, 7.986194]|+2.369881 [-1.458458, 6.290286]|
|short_le5s / Hybrid2-Hybrid8|42|+0.034260 [-1.696875, 1.753636]|-1.714948 [-4.353176, 0.756064]|
|short_le5s / Hybrid4-Hybrid8|42|-0.523801 [-1.540586, 0.576865]|-1.848257 [-3.730950, 0.117471]|
|short_le5s / Hybrid16-Hybrid8|42|+0.010096 [-1.460469, 1.456181]|+1.628753 [0.237849, 3.272892]|
|short_le5s / Uniform2-Hybrid2|42|+0.458576 [-1.384681, 2.431273]|-0.889950 [-4.276079, 2.621656]|
|short_le5s / Uniform4-Hybrid4|42|-0.387804 [-2.321100, 1.623223]|-1.445106 [-5.050037, 1.700850]|
|short_le5s / Uniform8-Hybrid8|42|-0.273473 [-1.497748, 1.105735]|-0.860610 [-2.335471, 0.517277]|
|short_le5s / Uniform16-Hybrid16|42|+0.295626 [-1.389268, 2.435810]|-2.697197 [-5.047195, -0.540361]|
|long_gt5s / Hybrid2-Source|86|+9.220602 [5.129875, 13.190299]|+3.399601 [-0.691324, 7.337829]|
|long_gt5s / Uniform2-Source|86|+10.337109 [6.429205, 14.230019]|+5.296658 [1.404491, 9.157307]|
|long_gt5s / Hybrid4-Source|86|+9.264917 [5.219625, 13.267179]|+5.710159 [2.340824, 9.123368]|
|long_gt5s / Uniform4-Source|86|+9.191412 [5.354313, 12.934776]|+3.573356 [-0.614298, 7.552639]|
|long_gt5s / Hybrid8-Source|86|+10.383125 [6.253841, 14.458518]|+7.813552 [3.966076, 11.621903]|
|long_gt5s / Uniform8-Source|86|+8.959372 [4.983597, 12.833541]|+6.382527 [2.795954, 9.861992]|
|long_gt5s / Hybrid16-Source|86|+10.182271 [6.352341, 14.123015]|+8.815182 [5.493429, 12.206497]|
|long_gt5s / Uniform16-Source|86|+10.496977 [6.840607, 14.217624]|+7.945950 [4.715773, 11.260752]|
|long_gt5s / Hybrid2-Hybrid8|86|-1.162523 [-2.921652, 0.400123]|-4.413951 [-7.353978, -1.649937]|
|long_gt5s / Hybrid4-Hybrid8|86|-1.118208 [-2.850619, 0.563707]|-2.103393 [-4.760884, 0.454579]|
|long_gt5s / Hybrid16-Hybrid8|86|-0.200854 [-1.590856, 1.608890]|+1.001630 [-0.506597, 2.749277]|
|long_gt5s / Uniform2-Hybrid2|86|+1.116507 [-0.938132, 3.429831]|+1.897058 [-1.710302, 5.660185]|
|long_gt5s / Uniform4-Hybrid4|86|-0.073505 [-1.827635, 1.692979]|-2.136803 [-5.442569, 1.034546]|
|long_gt5s / Uniform8-Hybrid8|86|-1.423753 [-2.868385, -0.077266]|-1.431025 [-2.925598, 0.014049]|
|long_gt5s / Uniform16-Hybrid16|86|+0.314706 [-0.592896, 1.217107]|-0.869232 [-1.972487, 0.263721]|

Pointwise parent-cluster CIs, no familywise correction. Allsigned positive/negative losses/rescues are retained. Source is native per-offset MAP+hull. Not every metric/schedule must increase with K; all exceptions/CIs crossing0 remain in tables. Originalexpert forward is only one nested component of deployment cost; source acquisition/requests/transport/startup are not zero or inferable from cachedCPUwall. No newSource/expert/GTscoring in this root export. Source/teacher costs inherited; current worker costs in cost_segments.json. PrivatePNG/PDF must be viewed by root before curve publication.
