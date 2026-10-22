# What event information is accessible to a low-capacity readout?

This is an explicitly **GT-supervised offline diagnostic**, not an unsupervised method, final STVG prediction or qualifying experiment. All128 method inputs/predictions were already sealed and independently scored. For each heldout query, exclude its entire parent from supervised fitting; fit only other parents within the same previously exposed panel. Fixed weighted ridge1, equalquery weights and equalframes withinquery, training-only centering/RMS and intercept. No hyperparameter/layer selection. No MBR, DP or final interval/box is produced. Scores are unbounded ridge outputs, not event probabilities.

Question: after supplying true feedback on other parents, do existing features offer a stable linear improvement beyond their own native temporal posterior? This separates a limited representation/readout hypothesis from unsupervised label quality; it is not an information-theoretic upper bound. Native last timehidden uses original last supervised boundary layer; targetASAu is the actual target-support wordhead input; Source34/60 coordinates are not assigned guessed word names.

|Panel|Feature/readout|heldout query-mean frame AUC|
|---|---|---:|
|HC64|NativeMembership|0.84273|
|HC64|NativeLastHidden|0.81843|
|HC64|ShiftedHeldoutHidden|0.24032|
|HC64|TargetASAU|0.72498|
|HC64|TargetSemanticP|0.64564|
|HC64|RandomMatchedP|0.66384|
|HC64|PositionPrior|0.72798|
|HC64|MembershipAffine|0.84273|
|HC64|NativeHiddenPlusMembership|0.83991|

HC64: 64queries/64parents, AUC valid64, excluded emptyclass0.
NativeLastHidden-PositionPrior: ΔAUC+0.09046, parent95%CI[0.011815589713661758, 0.16995595803184424]; +/−/=36/28/0.
NativeLastHidden-NativeMembership: ΔAUC-0.02430, parent95%CI[-0.05286849035842441, 0.0014448963527932228]; +/−/=21/40/3.
TargetASAU-NativeMembership: ΔAUC-0.11775, parent95%CI[-0.16060023345987093, -0.0753160648350525]; +/−/=9/55/0.
NativeHiddenPlusMembership-MembershipAffine: ΔAUC-0.00282, parent95%CI[-0.017653130466347745, 0.009860154312478771]; +/−/=26/32/6.
|hc2_recheck|NativeMembership|0.82537|
|hc2_recheck|NativeLastHidden|0.81407|
|hc2_recheck|ShiftedHeldoutHidden|0.26502|
|hc2_recheck|TargetASAU|0.76075|
|hc2_recheck|TargetSemanticP|0.68357|
|hc2_recheck|RandomMatchedP|0.64436|
|hc2_recheck|PositionPrior|0.67633|
|hc2_recheck|MembershipAffine|0.82537|
|hc2_recheck|NativeHiddenPlusMembership|0.82075|

hc2_recheck: 32queries/32parents, AUC valid32, excluded emptyclass0.
NativeLastHidden-PositionPrior: ΔAUC+0.13774, parent95%CI[0.019492045552086792, 0.2636914023800669]; +/−/=21/11/0.
NativeLastHidden-NativeMembership: ΔAUC-0.01130, parent95%CI[-0.05850734034188011, 0.03434381230515831]; +/−/=10/22/0.
TargetASAU-NativeMembership: ΔAUC-0.06462, parent95%CI[-0.14514027496919013, 0.01889248376425744]; +/−/=9/23/0.
NativeHiddenPlusMembership-MembershipAffine: ΔAUC-0.00462, parent95%CI[-0.023745887650644355, 0.013442200597070922]; +/−/=11/19/2.
|vidstg_recheck|NativeMembership|0.74498|
|vidstg_recheck|NativeLastHidden|0.75175|
|vidstg_recheck|ShiftedHeldoutHidden|0.34045|
|vidstg_recheck|TargetASAU|0.46549|
|vidstg_recheck|TargetSemanticP|0.46871|
|vidstg_recheck|RandomMatchedP|0.45721|
|vidstg_recheck|PositionPrior|0.61667|
|vidstg_recheck|MembershipAffine|0.74498|
|vidstg_recheck|NativeHiddenPlusMembership|0.76359|

vidstg_recheck: 32queries/32parents, AUC valid32, excluded emptyclass0.
NativeLastHidden-PositionPrior: ΔAUC+0.13508, parent95%CI[0.01941780404698194, 0.24885653423355447]; +/−/=20/9/3.
NativeLastHidden-NativeMembership: ΔAUC+0.00677, parent95%CI[-0.06061010669560647, 0.07251934016318237]; +/−/=19/13/0.
TargetASAU-NativeMembership: ΔAUC-0.27949, parent95%CI[-0.3927933257264332, -0.16152321939891295]; +/−/=7/25/0.
NativeHiddenPlusMembership-MembershipAffine: ΔAUC+0.01861, parent95%CI[-0.020713929012968358, 0.055750362835211634]; +/−/=21/11/0.

The native temporal hidden carries information beyond the simple physical position prior: paired AUC gains are positive in all three panels with intervals abovezero. It does not establish reliable extra linear information beyond native posterior membership: last-hidden and hidden-plus-membership differences versus native/calibratednative all have intervals crossingzero. Native membership already obtains AUC.8427/.8254/.7450. TargetASAu is weaker, particularly on fixed Vid32(.4655); that doesnotprove absence of nonlinear or query-conditioned information. Coordinate-agnostic class probabilities may have query-dependent semantics; this pooled supervised readout is not their best possible decoder.

These results weaken the hypothesis that another low-capacity projection of the same features alone will solve feedback quality. They support preserving native temporal ability as a strong reference. They do not prove every query update, feature representation or unsupervised event method impossible, and cannot qualify any new method for full evaluation. Full remains paused until the user-authorized07 decision.
Actual CPU audit6.23s,0model/GPU/expert/newobservations/finalSTVGpredictions. Five CPU checks verify weightedreference, all-parent exclusion, heldoutlabel isolation, syntheticcrossquerypositivecontrol androtation/scale invariance. Source48/provider434 unchanged.

[Protocol](protocol.json), [full aggregate/paired/anonymous-fold results](report.json), [per-query signs](signed_auc.tsv). Private curves/labels/weights/source captions/coordinates are not published.
