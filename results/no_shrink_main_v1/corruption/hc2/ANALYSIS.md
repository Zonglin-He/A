# Current no-shrinkage controlled views: hc2

Complete 128-query/128-parent panel × three views, sealed and independently officially scored (768 dense checks, max error 1.4432899320127035e-15), then 768 new prediction/receipt SHA checks plus bound original Source/compact/teacher hashes, all-array exact rederivation and all means verified. This root export reads already scored scalar rows only; zero new model, Source, expert or GT scoring. Kappa0 was selected by the user after exposed full results, not independent unseen confirmation.

|View|Native Source t/v %|FullMAP t/v %|Full−Source Δv pp [95% CI]|Full−CleanFull Δv pp [95% CI]|Native-source severe t/v/union|
|---|---:|---:|---:|---:|---|
|Blur|44.460067/24.345615|53.240135/31.177157|+6.831543 [4.210286, 9.423293]|-0.325026 [-1.511524, 0.840396]|23/16/32|
|JPEG|46.202452/24.940263|53.393422/30.826088|+5.885825 [3.371124, 8.320824]|-0.676095 [-2.188772, 0.806733]|18/20/31|
|Dim|45.093437/24.143950|54.560168/30.732645|+6.588695 [3.913813, 9.251969]|-0.769538 [-2.350178, 0.739197]|23/21/35|

## Every signed comparison and duration group

|View / comparison / group|Queries|Δt pp [95% CI]|Δv pp [95% CI]|
|---|---:|---:|---:|
|Blur / FullMAP-Source / all|128|+8.780068 [5.674596, 11.929879]|+6.831543 [4.210286, 9.423293]|
|Blur / FullMAP-CleanFullMAP / all|128|-1.136974 [-2.879593, 0.478887]|-0.325026 [-1.511524, 0.840396]|
|Blur / Source-CleanSource / all|128|-2.049410 [-3.925942, -0.265132]|-0.778638 [-1.807969, 0.237091]|
|Blur / FullMAP-Source / short_le5s|42|+4.774861 [0.264456, 9.404494]|+4.062176 [0.409807, 8.010618]|
|Blur / FullMAP-CleanFullMAP / short_le5s|42|-0.052849 [-2.459244, 2.556864]|+0.228609 [-1.475736, 2.039596]|
|Blur / Source-CleanSource / short_le5s|42|-2.110851 [-5.504680, 0.975063]|-0.395242 [-2.114743, 1.233249]|
|Blur / FullMAP-Source / long_gt5s|86|+10.736100 [6.739061, 14.704005]|+8.184024 [4.850946, 11.519863]|
|Blur / FullMAP-CleanFullMAP / long_gt5s|86|-1.666430 [-3.926366, 0.413696]|-0.595406 [-2.169638, 0.926540]|
|Blur / Source-CleanSource / long_gt5s|86|-2.019405 [-4.244473, 0.150341]|-0.965878 [-2.213275, 0.294708]|
|JPEG / FullMAP-Source / all|128|+7.190970 [4.130150, 10.249227]|+5.885825 [3.371124, 8.320824]|
|JPEG / FullMAP-CleanFullMAP / all|128|-0.983687 [-2.475942, 0.454437]|-0.676095 [-2.188772, 0.806733]|
|JPEG / Source-CleanSource / all|128|-0.307025 [-2.206023, 1.397029]|-0.183990 [-1.624374, 1.067896]|
|JPEG / FullMAP-Source / short_le5s|42|+1.276476 [-3.445055, 6.156434]|+1.944213 [-1.618669, 5.810219]|
|JPEG / FullMAP-CleanFullMAP / short_le5s|42|-0.214760 [-2.494435, 2.064113]|-0.293664 [-2.235404, 1.759544]|
|JPEG / Source-CleanSource / short_le5s|42|+1.225623 [-0.714957, 3.266074]|+1.200449 [0.049487, 2.371855]|
|JPEG / FullMAP-Source / long_gt5s|86|+10.079443 [6.261891, 13.948413]|+7.810799 [4.675022, 10.896039]|
|JPEG / FullMAP-CleanFullMAP / long_gt5s|86|-1.359210 [-3.325432, 0.471612]|-0.862864 [-2.876887, 1.120963]|
|JPEG / Source-CleanSource / long_gt5s|86|-1.055528 [-3.615630, 1.251580]|-0.860111 [-2.873913, 0.882192]|
|Dim / FullMAP-Source / all|128|+9.466731 [6.161960, 12.974569]|+6.588695 [3.913813, 9.251969]|
|Dim / FullMAP-CleanFullMAP / all|128|+0.183059 [-2.071895, 2.362035]|-0.769538 [-2.350178, 0.739197]|
|Dim / Source-CleanSource / all|128|-1.416040 [-4.183293, 1.192243]|-0.980303 [-2.748976, 0.680160]|
|Dim / FullMAP-Source / short_le5s|42|+2.984551 [-1.814734, 8.171558]|+3.756157 [-0.566169, 8.606956]|
|Dim / FullMAP-CleanFullMAP / short_le5s|42|+0.912955 [-1.938310, 3.924293]|+0.140710 [-1.924315, 2.204343]|
|Dim / Source-CleanSource / short_le5s|42|+0.645263 [-3.459923, 4.623686]|-0.177123 [-2.279469, 1.889896]|
|Dim / FullMAP-Source / long_gt5s|86|+12.632447 [8.350831, 17.123526]|+7.972028 [4.738659, 11.314283]|
|Dim / FullMAP-CleanFullMAP / long_gt5s|86|-0.173402 [-3.245075, 2.744041]|-1.214078 [-3.321391, 0.772516]|
|Dim / Source-CleanSource / long_gt5s|86|-2.422723 [-5.935211, 0.859916]|-1.372554 [-3.843980, 0.823073]|

Every signed positive/negative scalar, severe loss (>5pp, t/v union once), rescue, observed/unobserved effect and short (≤5s)/long interval is retained in each view’s matched_clean report/TSVs. CIs are pointwise parent-cluster bootstrap, not familywise. The same 128 parents recur across views; 384 views are not 384 independent parents. No blanket robustness or guaranteed non-degradation claim.

Blur=5×5 Gaussian sigma1.5; JPEG quality30/subsampling2; Dim uint8×0.5. Original acquisition used actual transformed-RGB Source/grid and exact-context teacher payloads, rebuilt geometry/real ROI pooling. Current kappa0 replay uses its own posterior MAP interval and actually re-solves final DP, while original kappa64 array equality remains an exact prerequisite. CleanFullMAP is the audited NoShrink output on these same queries; Source remains native per-offset MAP+hull. No MBR performance comparison.

Costs retain actual Source/teacher provenance and original per-PID/start query counts; current cached CPU rederivation wall is additional research work. Source-forward/request/finish/query durations overlap and must not be added. Old teacher hits do not erase original expert acquisition. Loading/decode/outerloop are accounted only by whole original worker wall, not individually instrumented. None of these cached times substitutes for current uncached deployment measurement. Media/captions/coordinates remain private.
