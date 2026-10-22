# Covariance-aware semantic prototype feedback

Source-only class correlation/effective-rank audit motivates testing shared covariance, not a proved failure cause. Keep exact original learned semantic p, native prior, opposite-offset positive/negative soft means and two rounds. Fit OAS covariance ONCE per heldout parity from ALL opposite-parity rows, centered unweighted; covariance is independent of pseudo-membership. Gaussian shared-covariance logdensity ratio replaces independent-class Bernoulli pseudolikelihood. Each round anchors to original q0; inclusive interval sum, original MBR/eight observations/complete-box DP unchanged. No new model calls, query learning, gain/temperature/iteration/shrinkage/dimension/class-count scan or confidence veto. Gaussian/iid covariance assumptions are approximations for bounded correlated temporal features.

Implementation follows sklearn1.9.0 modified OAS formula, tested against actual installed _oas including one-feature special case. Shared Gaussian covariance cancels logdet. Stable solve with no manual ridge/cutoff. Exact flat covariance gives zero evidence/native exactly; undefined seed class uses parity global mean. Five CPU tests cover reference covariance, direct Gaussian densities, constant exact no-op, coordinate permutation/rotation/crossfit isolation, synthetic correlated-event positive control.

Same-info controls: diagonal SAME OAS covariance, trace-matched isotropic, exact-histogram random coordinates, reversed semantic placement, one round, direct SAME second-round evidence with uniform prior, exact parentBernoulli and Sharpen aliases, constant no-op. All128 HC64/HC32/Vid32 decisions seal before independent CPU labels. These are exposed development panels.

|Panel|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---|---:|---:|---:|---:|
|HC64|NativeG6|29.6952|55.9613|—|—|
|HC64|CovariancePrototypeLoop2|29.3254|55.6925|28/34/2|24|
|HC64|OneStepCovariance|29.5031|55.7854|30/30/4|16|
|HC64|DirectCovariance|29.1922|55.6204|28/34/2|23|
|HC64|DiagonalCovarianceLoop2|30.0330|57.5007|33/30/1|21|
|HC64|IsotropicCovarianceLoop2|26.2434|51.9245|27/36/1|33|
|HC64|ReversedCovarianceLoop2|26.2035|49.4389|25/36/3|27|
|HC64|RandomCovarianceLoop2|28.4767|54.3221|31/30/3|23|
|HC64|ParentBernoulli|30.9658|58.8883|31/17/16|6|
|HC64|SharpenOriginal|28.7258|53.7518|16/38/10|20|
|HC64|ConstantProjectionNoOp|29.6952|55.9613|0/0/64|0|
|HC64|Source|26.8938|54.8230|—|—|

HC64: primary−G6 t-0.2688pp CI[-3.5873676625726447, 3.0464949274227178], v-0.3697pp CI[-2.1935038695952187, 1.4792343462375164]. Primary−Source t+0.8695/v+2.4317pp.
Source short+0.3073/long+1.0128tpp; fixed screen=False.
Primary−DiagonalCovarianceLoop2: t-1.8082pp CI[-5.321351812575048, 1.684068001177262].
Primary−IsotropicCovarianceLoop2: t+3.7679pp CI[-0.7849934461633779, 8.329112195877888].
Primary−RandomCovarianceLoop2: t+1.3703pp CI[-1.827981226204584, 4.846921978875801].
Primary−ReversedCovarianceLoop2: t+6.2536pp CI[2.258683463272513, 10.4432210525756].
Primary−ParentBernoulli: t-3.1958pp CI[-6.348029494315608, -0.20070590205508776].
|hc2_recheck|NativeG6|27.8907|51.6010|—|—|
|hc2_recheck|CovariancePrototypeLoop2|28.1938|52.1937|13/17/2|13|
|hc2_recheck|OneStepCovariance|27.5510|51.4375|16/15/1|9|
|hc2_recheck|DirectCovariance|27.5102|51.2296|13/18/1|14|
|hc2_recheck|DiagonalCovarianceLoop2|26.3268|50.3831|15/17/0|13|
|hc2_recheck|IsotropicCovarianceLoop2|22.4332|41.1008|9/23/0|19|
|hc2_recheck|ReversedCovarianceLoop2|27.9786|51.1487|16/13/3|10|
|hc2_recheck|RandomCovarianceLoop2|29.1384|53.7511|19/10/3|8|
|hc2_recheck|ParentBernoulli|27.3123|51.9587|15/12/5|5|
|hc2_recheck|SharpenOriginal|27.5315|50.1412|8/19/5|5|
|hc2_recheck|ConstantProjectionNoOp|27.8907|51.6010|0/0/32|0|
|hc2_recheck|Source|23.4655|50.4639|—|—|

hc2_recheck: primary−G6 t+0.5927pp CI[-3.656708291026863, 4.95471622292507], v+0.3031pp CI[-2.417640528036194, 2.9919664876544503]. Primary−Source t+1.7297/v+4.7283pp.
Source short-0.7906/long+3.0500tpp; fixed screen=False.
Primary−DiagonalCovarianceLoop2: t+1.8105pp CI[-3.0302092930501723, 7.411223577905961].
Primary−IsotropicCovarianceLoop2: t+11.0929pp CI[3.318777349393113, 19.73448280885262].
Primary−RandomCovarianceLoop2: t-1.5574pp CI[-6.0370239899439495, 3.0598591977859093].
Primary−ReversedCovarianceLoop2: t+1.0449pp CI[-4.278899800857063, 6.619428605742359].
Primary−ParentBernoulli: t+0.2349pp CI[-3.577869454176638, 4.426830822405682].
|vidstg_recheck|NativeG6|23.9576|35.6554|—|—|
|vidstg_recheck|CovariancePrototypeLoop2|22.0942|32.4982|5/19/8|8|
|vidstg_recheck|OneStepCovariance|22.3308|32.8192|7/18/7|9|
|vidstg_recheck|DirectCovariance|22.1208|32.3367|5/22/5|8|
|vidstg_recheck|DiagonalCovarianceLoop2|19.6813|29.3159|6/22/4|15|
|vidstg_recheck|IsotropicCovarianceLoop2|18.8936|28.2995|6/23/3|16|
|vidstg_recheck|ReversedCovarianceLoop2|22.1360|33.3520|7/21/4|12|
|vidstg_recheck|RandomCovarianceLoop2|23.4069|34.6500|11/15/6|7|
|vidstg_recheck|ParentBernoulli|23.6888|35.0400|9/16/7|7|
|vidstg_recheck|SharpenOriginal|23.4813|35.3245|14/11/7|1|
|vidstg_recheck|ConstantProjectionNoOp|23.9576|35.6554|0/0/32|0|
|vidstg_recheck|Source|15.6772|34.0110|—|—|

vidstg_recheck: primary−G6 t-3.1571pp CI[-8.134041622556884, 2.1392893761544394], v-1.8633pp CI[-5.493947817222999, 2.3211045627408136]. Primary−Source t-1.5128/v+6.4170pp.
Source short-2.9748/long+3.7085tpp; fixed screen=False.
Primary−DiagonalCovarianceLoop2: t+3.1823pp CI[-0.32585957793430786, 6.807559183290791].
Primary−IsotropicCovarianceLoop2: t+4.1988pp CI[-0.3354661515892738, 8.742148357671619].
Primary−RandomCovarianceLoop2: t-2.1518pp CI[-6.0981403541392645, 1.3668192936935404].
Primary−ReversedCovarianceLoop2: t-0.8538pp CI[-6.067382576095998, 4.723579794541957].
Primary−ParentBernoulli: t-2.5418pp CI[-5.609623367935899, -0.05004583037372133].

Not qualified. Retain all positive and negative controls; no covariance/shrinkage/gain/capacity scan or posthoc control promotion.
Actual new model calls{'new_model_calls': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_fusion': 0, 'new_expert': 0, 'new_observations': 0, 'new_optimizer': 0}; CPU worker13.58s includes128 queries/all controls/DP/I/O and is not deployment latency. Official/dense checks3072, maxerror1.89e-15.

[All signed results](signed.tsv), [all severe declines](severe_declines.tsv), [protocol](protocol.json). [Original OAS paper](https://arxiv.org/abs/0907.4698) concerns covariance estimation for Gaussian samples; it does not validate STVG event feedback. [Official OAS implementation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html).
