# Fixed-support Gaussian-mean KL projection diagnostic32

Same exposed NH16/NV16 developer parents. Original frozenTA cross-domain EMA/E+finalMBR/four cachedDINO/FineROI paths unchanged. No new model, teacher, ROI, candidate, DINO, features, training or GPU calls. Five already saved originalCenter/tau_low/high/step_low/high Live paths, each of2 saved supports is independently projected. All32x16original/counterfactual outputs sealed before NEWGT; main comparison projection2 versus SAME original final2. It is not a new sequential on-policy path and no teacher is requeried at projected means.

Fix saved d, real valid and detachedq. F(mu)=KL(q||softmax(-||d-mu||²/2)); gradient mean_p[d]-mean_q[d], HessianCov_p(d). SVD of valid differences d-d0 determines affine rank (1e-12relative threshold), preserve saved eta_after null component. Newton/Armijo till projected gradient infinity<=1e-10/max120, no GT stopping/checkpoint selection. Active=allowed & originalinformative & any realvalid; all other nodes preserve original actualstate. Same .25Ewh mean readout, widths/planned/MBR unchanged.

Numerical recovery001: originalArmijo stalled atgradient2.654e-10/max120 before any prediction file was written. Preserved oldsolver/runtime/failure; added64FP64eps objective-scale rounding allowance ONLY in line search. Original objective, gradienttol1e-10/max120 and inputs unchanged; all final solves still meetKKT. Root independently verifies7650KKT/null/Pythagorean identities,320paths,2560metrics,60bootstraps; maxgrad9.995e-11/null7.77e-16/identity1.20e-10. This is a KKT tolerance numerical certificate, not a strict dual-gap bound.

Projected mean need not lie within candidate support; report candidate coordinate-axis bbox exceedance only. Not exceeding axis ranges does NOT prove inside convex hull. No newROI/reward at projected output, no visual-verification claim.

## Last support fit decomposition

KL means are conditional on active parents: HC15/Vid11 for originalCenter, not the ALL16parent primary metric denominator. Remaining KL refers ONLY to saved q/valid/fixedvariance mean family.

| Target | Original arm | Defined parents | KL at actual final mean | KL projected | Removable KL |
|---|---|---:|---:|---:|---:|
| hc2 | Center_live | 15 | 0.316680267 | 0.310331332 | 0.006348935 |
| hc2 | Tau_low | 15 | 0.681541592 | 0.665642825 | 0.015898767 |
| hc2 | Tau_high | 15 | 0.102215780 | 0.099953997 | 0.002261784 |
| hc2 | Step_low | 15 | 0.317548010 | 0.307996211 | 0.009551800 |
| hc2 | Step_high | 15 | 0.316153331 | 0.313420794 | 0.002732537 |
| vidstg | Center_live | 11 | 0.363616005 | 0.352353020 | 0.011262986 |
| vidstg | Tau_low | 11 | 0.794187301 | 0.767919163 | 0.026268138 |
| vidstg | Tau_high | 11 | 0.119379966 | 0.115551148 | 0.003828818 |
| vidstg | Step_low | 11 | 0.365985909 | 0.349299932 | 0.016685977 |
| vidstg | Step_high | 11 | 0.360456097 | 0.355869265 | 0.004586832 |

OriginalCenter HC/Vid KL .316680/.363616 falls to.310331/.352353: removable.006349/.011263nats, about2.00%/3.10% of conditional mean current KL. Much of this finite-target discrepancy is outside the fixedvariance mean family, but this does not prove an expanded family improves task utility. Means matched at optimum are not set to teacher coordinate expectation; the exponential family map must be solved.

## Projection2 minus original final2, parent pp

| Target | Original arm | Full vIoU [95%CI] | ALL known unobserved actual sIoU [95%CI] | v positive/negative/neutral |
|---|---|---|---|---|
| hc2 | Center_live | -0.084127 [-0.444452,+0.214195] | -0.240954 [-0.732160,+0.200786] | 8/7/1 |
| hc2 | Tau_low | -0.309524 [-0.911578,+0.138580] | -0.709722 [-1.508987,-0.039110] | 7/8/1 |
| hc2 | Tau_high | +0.061696 [-0.120087,+0.233490] | +0.049157 [-0.224763,+0.333190] | 9/6/1 |
| hc2 | Step_low | -0.042336 [-0.436559,+0.296624] | -0.178914 [-0.719122,+0.327248] | 7/8/1 |
| hc2 | Step_high | -0.087127 [-0.334239,+0.121776] | -0.214852 [-0.560298,+0.103325] | 7/8/1 |
| vidstg | Center_live | -0.008527 [-0.170879,+0.123228] | -0.078204 [-0.445351,+0.294354] | 4/2/10 |
| vidstg | Tau_low | -0.127535 [-0.477055,+0.103577] | -0.370957 [-1.022833,+0.173335] | 2/4/10 |
| vidstg | Tau_high | +0.075791 [+0.004712,+0.165026] | +0.084554 [-0.070307,+0.320641] | 4/2/10 |
| vidstg | Step_low | +0.028649 [-0.120403,+0.180654] | -0.001532 [-0.407642,+0.474997] | 4/2/10 |
| vidstg | Step_high | -0.031459 [-0.160406,+0.059095] | -0.107287 [-0.329428,+0.093990] | 2/4/10 |

OriginalCenter projection completev−.084127/−.008527pp and actuals−.240954/−.078204, all four intervals cross0. No established universal harm, but no evidence more complete current-q fitting yields common gains. HC NH013 completev−2.143635/actuals−2.698816 versusoriginalCenter; removing it turns fullv mean positive, whereasactuals leave-one-out means stay negative. All cases remain main results.

Tau_low projection actuals HC−.709722pp CI[-1.508987,-.039110], Vid−.370957CIcross0; NH013actuals−5.412068pp is a serious tail, not hidden by prior rounds no-severe-loss statements. Tau_high projection completev+.061696/+.075791, VidCI[+.004712,+.165026]positive, but actuals uncertain. Different finite q can have different outcomes; no selection of Tau_high from these offline results.

Step_high projection versusactual output fullv−.087127/−.031459, actuals−.214852/−.107287, allintervalscross0. More mean fit is not synonymous with better feedback or output. Does not revoke previous limited positive step sensitivity; new-source step confirmation remains independent and requires its own execution.

Primary metrics keep all parents/no-reference/unknown, original ALL-known-unobserved actual node denominator, temporal intervals identical, observed nodes unchanged. GT only offline after outputs sealed. Independent root audit reconstructs objective/moments/null, Pythagorean decomposition and official metrics; same existingGPT6.1Sol/max reviewedcode/results. This repeated panel and unadjusted intervals are diagnosis, not independent method confirmation.

All signed projection-versusE and versusoriginal metrics and every parent retained in RESULTS_PUBLIC.json; rank0/1/2, axisrange, iterations/KKT/conditional fitstats in FIT_DECOMPOSITION.json and private sealed certificates. No default/production/heartbeat change, no new variance/readout/step search.
