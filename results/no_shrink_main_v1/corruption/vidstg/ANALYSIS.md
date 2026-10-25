# Current no-shrinkage controlled views: vidstg

Complete 128-query/128-parent panel × three views, sealed and independently officially scored (768 dense checks, max error 1.4432899320127035e-15), then 768 new prediction/receipt SHA checks plus bound original Source/compact/teacher hashes, all-array exact rederivation and all means verified. This root export reads already scored scalar rows only; zero new model, Source, expert or GT scoring. Kappa0 was selected by the user after exposed full results, not independent unseen confirmation.

|View|Native Source t/v %|FullMAP t/v %|Full−Source Δv pp [95% CI]|Full−CleanFull Δv pp [95% CI]|Native-source severe t/v/union|
|---|---:|---:|---:|---:|---|
|Blur|31.889488/12.297238|35.770594/14.996405|+2.699167 [0.239962, 5.298345]|-0.431372 [-1.754676, 1.002564]|17/23/32|
|JPEG|31.908497/13.369802|35.071119/14.504310|+1.134508 [-0.952232, 3.311222]|-0.923468 [-1.854762, -0.078195]|24/24/37|
|Dim|31.104485/13.433187|34.784342/14.982930|+1.549743 [-0.830497, 3.951142]|-0.444847 [-1.368462, 0.379715]|17/25/31|

## Every signed comparison and duration group

|View / comparison / group|Queries|Δt pp [95% CI]|Δv pp [95% CI]|
|---|---:|---:|---:|
|Blur / FullMAP-Source / all|128|+3.881106 [1.667078, 6.232405]|+2.699167 [0.239962, 5.298345]|
|Blur / FullMAP-CleanFullMAP / all|128|+1.092885 [-0.417673, 2.561618]|-0.431372 [-1.754676, 1.002564]|
|Blur / Source-CleanSource / all|128|-0.232047 [-2.070639, 1.638662]|-1.025665 [-2.090074, 0.010000]|
|Blur / FullMAP-Source / short_le5s|78|+0.043737 [-1.821068, 1.816563]|+2.364877 [-0.073604, 5.147835]|
|Blur / FullMAP-CleanFullMAP / short_le5s|78|+0.826837 [-1.079220, 2.504989]|+0.744759 [-0.758648, 2.619808]|
|Blur / Source-CleanSource / short_le5s|78|-0.451987 [-2.477467, 1.284296]|-1.396855 [-2.710217, -0.289852]|
|Blur / FullMAP-Source / long_gt5s|50|+9.867401 [5.420603, 14.721341]|+3.220661 [-1.786675, 8.326521]|
|Blur / FullMAP-CleanFullMAP / long_gt5s|50|+1.507921 [-1.248992, 4.284371]|-2.266137 [-4.477131, -0.267646]|
|Blur / Source-CleanSource / long_gt5s|50|+0.111060 [-3.381831, 4.032508]|-0.446608 [-2.201973, 1.449950]|
|JPEG / FullMAP-Source / all|128|+3.162622 [0.840986, 5.752774]|+1.134508 [-0.952232, 3.311222]|
|JPEG / FullMAP-CleanFullMAP / all|128|+0.393410 [-1.041324, 1.796543]|-0.923468 [-1.854762, -0.078195]|
|JPEG / Source-CleanSource / all|128|-0.213038 [-1.657598, 1.171046]|+0.046899 [-0.474031, 0.600034]|
|JPEG / FullMAP-Source / short_le5s|78|-0.864771 [-2.837894, 1.067769]|-0.206198 [-1.964959, 1.629497]|
|JPEG / FullMAP-CleanFullMAP / short_le5s|78|+0.332612 [-0.846399, 1.566646]|-0.722057 [-1.648401, 0.086835]|
|JPEG / Source-CleanSource / short_le5s|78|-0.037703 [-1.254352, 1.198721]|-0.292596 [-0.915021, 0.256242]|
|JPEG / FullMAP-Source / long_gt5s|50|+9.445356 [4.814794, 14.692344]|+3.226010 [-1.425780, 7.971550]|
|JPEG / FullMAP-CleanFullMAP / long_gt5s|50|+0.488256 [-2.753833, 3.427466]|-1.237669 [-3.158001, 0.487201]|
|JPEG / Source-CleanSource / long_gt5s|50|-0.486560 [-3.784135, 2.463878]|+0.576512 [-0.305686, 1.671723]|
|Dim / FullMAP-Source / all|128|+3.679857 [1.208405, 6.246784]|+1.549743 [-0.830497, 3.951142]|
|Dim / FullMAP-CleanFullMAP / all|128|+0.106634 [-1.796562, 1.909702]|-0.444847 [-1.368462, 0.379715]|
|Dim / Source-CleanSource / all|128|-1.017050 [-3.294666, 1.160052]|+0.110284 [-0.783611, 1.028377]|
|Dim / FullMAP-Source / short_le5s|78|-0.041563 [-2.322655, 2.185416]|-0.494882 [-2.695331, 1.587131]|
|Dim / FullMAP-CleanFullMAP / short_le5s|78|-0.804081 [-2.954313, 1.157843]|-0.739481 [-1.928699, 0.280350]|
|Dim / Source-CleanSource / short_le5s|78|-1.997605 [-4.236907, -0.025197]|-0.021337 [-1.184671, 1.160969]|
|Dim / FullMAP-Source / long_gt5s|50|+9.485272 [4.693084, 14.654708]|+4.739359 [-0.356769, 9.855730]|
|Dim / FullMAP-CleanFullMAP / long_gt5s|50|+1.527349 [-2.182312, 4.974432]|+0.014781 [-1.367720, 1.359613]|
|Dim / Source-CleanSource / long_gt5s|50|+0.512618 [-4.305251, 5.100196]|+0.315612 [-1.047139, 1.627624]|

Every signed positive/negative scalar, severe loss (>5pp, t/v union once), rescue, observed/unobserved effect and short (≤5s)/long interval is retained in each view’s matched_clean report/TSVs. CIs are pointwise parent-cluster bootstrap, not familywise. The same 128 parents recur across views; 384 views are not 384 independent parents. No blanket robustness or guaranteed non-degradation claim.

Blur=5×5 Gaussian sigma1.5; JPEG quality30/subsampling2; Dim uint8×0.5. Original acquisition used actual transformed-RGB Source/grid and exact-context teacher payloads, rebuilt geometry/real ROI pooling. Current kappa0 replay uses its own posterior MAP interval and actually re-solves final DP, while original kappa64 array equality remains an exact prerequisite. CleanFullMAP is the audited NoShrink output on these same queries; Source remains native per-offset MAP+hull. No MBR performance comparison.

Costs retain actual Source/teacher provenance and original per-PID/start query counts; current cached CPU rederivation wall is additional research work. Source-forward/request/finish/query durations overlap and must not be added. Old teacher hits do not erase original expert acquisition. Loading/decode/outerloop are accounted only by whole original worker wall, not individually instrumented. None of these cached times substitutes for current uncached deployment measurement. Media/captions/coordinates remain private.
