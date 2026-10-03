# Role-conditioned P/R readout audit

The population-only intervention fits A/W roles much better in-sample but
does not recover held-out absolute P/R quality. Every confirmation-corruption
role MAE rises in point estimate, with all paired MAE intervals crossing zero;
all eight role R2 values remain negative. HC's analytic decision loses helpful
accepts and admits more harmful corrections. Retain Vid's local severe-harm
rejection, but do not promote M-role or infer that latent information is absent.

This is explicitly target-search-GT-supervised CPU diagnosis on fixed cached
A8/L32 choices, not an unsupervised TTA method, new prediction stream or frozen
backbone performance comparison. No GPU, experts, candidates, replay or updates.

## Matched setting and provenance

Predecessor: d408f154fb753764fef4aac2ca510f9b3b8fd36a. Same source lists,
checkpoints Vid fbb1ed88 /HC ee72f0d9, sixth temporal hidden, Inside[512:768] ->
precision /Endpoint[0:512] -> recall, fixed source-selected ridge alpha Vid1/1
and HC1/.1. The original 32 search +16 confirm sources/dataset/one query/two
orders/six conditions/25% expert design yields 288 cached expert cells: Vid
16 search +8 confirm independent expert sources, HC14+7, each96+48 cells.
Corruption contains80 search +40 confirm cells/dataset; clean16+8. All have
historical exposure; source-disjoint confirmation is not fresh project testing.

M-all reuses its four sealed target-search fits trained on3072 candidate rows
per dataset. M-role fits four shared heads on192 ordered [A,W] rows/dataset;
retain identical rows if A=W, without labels filtering helpful/harmful. Both
arms have source-equal weights summing to1, unpenalized bias and identical
weighted normalization procedure. New mean/std/bias are learned only from the
role population, so normalizer, label prior and candidate diversity change
together; this is not a coefficients-only intervention under frozen statistics.
No role-ID feature, separate A/W heads, alpha search or decision threshold.
The quality readout never changes the original L32 winner or A's1792 spatial
parameter trajectory, VidK1/HCK8, two-offset pixels or expert positions.

Separate processes sealed role models, then all288 GT-free scalar readouts
before confirmation GT join. M-all predictions reproduce predecessor arrays.
No confirmation features/labels entered fitting. Hash integrity reads and
historical label exposure are disclosed. Private coefficients/normalizers,
latents, GT spans, media and original weights are excluded from publication.

## Absolute role readout: primary confirmation corruption

Raw predictions are not clipped for regression. Conditions -> orders -> equal
source aggregation and paired whole-source10000 bootstrap, seed20261003.
Intervals are exploratory, unadjusted; undefined draws remain undefined.

### MAE (lower is better)

| Dataset | Role | M-all | M-role | Paired role − all |
|---|---|---:|---:|---:|
| vidstg | P_A | 0.3619 [0.2121, 0.5315] | 0.3792 [0.2272, 0.5569] | 0.0173 [-0.0473, 0.0870] |
| vidstg | R_A | 0.2887 [0.1923, 0.3968] | 0.3155 [0.2125, 0.4285] | 0.0268 [-0.0490, 0.1157] |
| vidstg | P_W | 0.3335 [0.1796, 0.5133] | 0.3729 [0.2164, 0.5697] | 0.0394 [-0.0316, 0.1165] |
| vidstg | R_W | 0.2936 [0.1516, 0.4476] | 0.3140 [0.1911, 0.4389] | 0.0205 [-0.1042, 0.1492] |
| hc2 | P_A | 0.2753 [0.1733, 0.3624] | 0.2822 [0.1791, 0.3813] | 0.0069 [-0.0306, 0.0410] |
| hc2 | R_A | 0.2217 [0.1386, 0.3044] | 0.2579 [0.1244, 0.4058] | 0.0362 [-0.0466, 0.1159] |
| hc2 | P_W | 0.2287 [0.1423, 0.3071] | 0.2378 [0.1371, 0.3386] | 0.0091 [-0.0311, 0.0459] |
| hc2 | R_W | 0.1939 [0.1428, 0.2585] | 0.2551 [0.1379, 0.3962] | 0.0613 [-0.0121, 0.1424] |

### R2

| Dataset | Role | M-all | M-role | Paired role − all |
|---|---|---:|---:|---:|
| vidstg | P_A | -0.1461 [-2.3661, 0.5339] | -0.2552 [-2.7926, 0.5129] | -0.1091 [-0.7463, 0.2805] |
| vidstg | R_A | -2.1293 [-15.8048, -0.0829] | -2.9330 [-16.4492, -0.6939] | -0.8036 [-3.4547, 1.1429] |
| vidstg | P_W | -0.2059 [-3.1828, 0.6040] | -0.4717 [-4.1572, 0.5273] | -0.2658 [-1.2676, 0.2341] |
| vidstg | R_W | -7.1752 [-34.4518, -1.6742] | -7.4156 [-27.9244, -3.2664] | -0.2404 [-4.1194, 8.9589] |
| hc2 | P_A | -0.4989 [-19.4841, -0.0499] | -0.5595 [-17.5367, -0.2310] | -0.0607 [-0.4103, 2.2767] |
| hc2 | R_A | -0.7389 [-18.9618, -0.2321] | -1.3741 [-27.1090, -0.6470] | -0.6352 [-8.8659, 1.0635] |
| hc2 | P_W | -0.1877 [-8.0508, 0.1275] | -0.2889 [-7.3060, -0.0745] | -0.1012 [-0.4959, 1.2525] |
| hc2 | R_W | -0.4458 [-3.9261, -0.1824] | -1.2609 [-9.8347, -0.4443] | -0.8151 [-5.8248, 0.1565] |

### Pearson correlation

| Dataset | Role | M-all | M-role | Paired role − all |
|---|---|---:|---:|---:|
| vidstg | P_A | 0.2426 [-0.4761, 0.8041] | 0.2716 [-0.4363, 0.8573] | 0.0290 [-0.1305, 0.3078] |
| vidstg | R_A | 0.3961 [-0.3860, 0.7710] | 0.0225 [-0.5348, 0.4899] | -0.3735 [-0.7975, 0.4471] |
| vidstg | P_W | 0.2797 [-0.4384, 0.8310] | 0.2861 [-0.4496, 0.8269] | 0.0064 [-0.1418, 0.2069] |
| vidstg | R_W | -0.2098 [-0.7726, 0.4174] | 0.0600 [-0.6672, 0.5737] | 0.2698 [-0.2893, 0.8950] |
| hc2 | P_A | -0.2189 [-0.6574, 0.2761] | -0.5520 [-0.7325, -0.1214] | -0.3331 [-0.5945, -0.0076] |
| hc2 | R_A | -0.1909 [-0.7078, 0.4152] | -0.4617 [-0.7691, -0.1526] | -0.2709 [-0.6882, 0.1070] |
| hc2 | P_W | 0.1692 [-0.3878, 0.7103] | -0.2203 [-0.5755, 0.4412] | -0.3895 [-0.7928, -0.0472] |
| hc2 | R_W | -0.2652 [-0.5945, 0.4283] | -0.2604 [-0.6272, 0.2408] | 0.0049 [-0.5852, 0.3659] |

## Search fitting is successful in-sample

Search roles are the training population of M-role, not an independent
validation. Better fitting alone does not prove generalizable P/R encoding.

| Dataset | Role | M-all | M-role | Paired role − all |
|---|---|---:|---:|---:|
| vidstg | P_A | 0.8593 [0.6858, 0.9446] | 0.9481 [0.9070, 0.9677] | 0.0888 [0.0174, 0.2253] |
| vidstg | R_A | 0.4176 [-0.3194, 0.6835] | 0.9404 [0.8702, 0.9711] | 0.5228 [0.2619, 1.2020] |
| vidstg | P_W | 0.8358 [0.6932, 0.9177] | 0.9383 [0.8857, 0.9677] | 0.1025 [0.0402, 0.2131] |
| vidstg | R_W | 0.4981 [0.0368, 0.6339] | 0.9222 [0.8646, 0.9564] | 0.4240 [0.2845, 0.8530] |
| hc2 | P_A | 0.7241 [0.3927, 0.8213] | 0.9136 [0.7485, 0.9638] | 0.1895 [0.1147, 0.3999] |
| hc2 | R_A | 0.8823 [0.0935, 0.9538] | 0.9956 [0.9618, 0.9985] | 0.1133 [0.0435, 0.8770] |
| hc2 | P_W | 0.6935 [0.3820, 0.7879] | 0.9289 [0.8212, 0.9639] | 0.2354 [0.1717, 0.4511] |
| hc2 | R_W | 0.8499 [-0.0304, 0.9132] | 0.9910 [0.9530, 0.9977] | 0.1410 [0.0837, 1.0001] |

## Fixed analytic correction decision

Clip P/R only for F=PR/(P+R-PR), with F(0,0)=0. Accept the already fixed W only
when eligible and predicted F(W)-F(A)>0; no-op retains A. Binary truth is cached
true delta_t>+/-1e-12; neutral/no-op remain but are excluded from binary metrics.
AUROC is oriented without sign flipping and equal-source-weighted within each
class. Utility is the existing expert subset under a frozen readout simulation,
not full-flow benefit or newly executed online adaptation.

| Dataset/model | AUROC | Balanced accuracy | Helpful/harmful accepts | Severe v-harms | Expert ΔvIoU pp vs A |
|---|---:|---:|---:|---:|---:|
| vidstg/all | 0.7271 [0.3333, 1.0000] | 0.5786 [0.2339, 0.9167] | 19/7 | 1 | 0.3339 [-0.7417, 1.3142] |
| vidstg/role | 0.6652 [0.3283, 0.9750] | 0.5881 [0.3104, 0.9000] | 19/6 | 0 | 0.2522 [-0.9343, 1.4413] |
| hc2/all | 0.6796 [0.4584, 0.9048] | 0.7083 [0.5583, 0.8667] | 9/11 | 10 | -1.8620 [-4.9012, 0.7493] |
| hc2/role | 0.4417 [0.1500, 0.7405] | 0.5042 [0.3750, 0.6548] | 6/18 | 9 | -2.4015 [-5.2458, 0.1056] |

| Dataset | Paired AUROC | Paired BA | Paired expert ΔvIoU pp | Paired accepted precision |
|---|---:|---:|---:|---:|
| vidstg | -0.0619 [-0.2540, 0.1583] | 0.0095 [-0.1726, 0.2202] | -0.0817 [-1.1283, 1.1228] | 0.0292 [-0.1128, 0.1978] |
| hc2 | -0.2380 [-0.6361, 0.0476] | -0.2042 [-0.4067, 0.0333] | -0.5395 [-1.2868, -0.0357] | -0.1709 [-0.3622, -0.0067] |

HC helpful accepts9->6 and harmful11->18; severe10->9 does not offset those
errors. Paired expert delta_v=-.5395pp[-1.2868,-.0357], accepted precision
-.1709[-.3622,-.0067] and source-balanced accuracy-.2405[-.4429,-.0571]; AUROC
and BA deltas have intervals crossing zero. These are local exploratory
results from seven sources. The helpful-source TPR loss is identical in this
small class composition; its degenerate bootstrap interval is not universal
certainty or a new reliability guarantee.

Vid helpful accepts remain19, harmful7->6, severe1->0. It also switches which
helpful cases it accepts: expert utility+.3339pp->+.2522pp vs A, paired change
-.0817pp[-1.1283,1.1228]. Thus the local harm rejection is preserved without
declaring net improvement or role mapping recovery.

## Clean control

| Dataset/model | AUROC | Balanced accuracy | Helpful/harmful accepts | Severe v-harms | Expert ΔvIoU pp vs A |
|---|---:|---:|---:|---:|---:|
| vidstg/all | 0.7333 [0.2500, 1.0000] | 0.5667 [0.2500, 0.9286] | 4/2 | 0 | -0.1524 [-1.9149, 1.6101] |
| vidstg/role | 0.7333 [0.2500, 1.0000] | 0.5667 [0.2500, 0.9286] | 4/2 | 0 | -0.2797 [-1.9236, 1.2710] |
| hc2/all | 0.4167 [0.0000, 1.0000] | 0.4583 [0.0833, 0.8000] | 2/3 | 2 | -1.8446 [-5.5521, 1.5822] |
| hc2/role | 0.0000 [0.0000, 0.0000] | 0.3333 [0.0000, 0.5000] | 2/4 | 2 | -2.0295 [-5.7357, 1.3826] |

## Per-order confirmation corruption (point estimates only)

| Dataset/order | M-all AUROC/BA | M-role AUROC/BA |
|---|---:|---:|
| vidstg/order1 | 0.5868/0.4917 | 0.4410/0.3910 |
| vidstg/order2 | 1.0000/0.8333 | 1.0000/0.9667 |
| hc2/order1 | 0.7778/0.7500 | 0.4167/0.5417 |
| hc2/order2 | undefined/undefined | undefined/undefined |

HC order2 has no helpful class; its binary AUROC/BA are undefined, not chance. Delete-one-source outputs are influence of frozen predictions, not cross-validation or refits.

## Why pooled R2 versus role R2 is not sufficient root-cause proof

Changing selected-role population may matter, but selecting roles also changes
label variance and support. R2=1-MSE/Var(truth); a high pooled value alone
does not establish precise A/W estimates or a causal selection mismatch.
The following true variances and matched M-all pooled/role statistics make
that denominator change explicit; M-role's all32 application is extrapolation
outside its chosen training support, not its training objective.

| Dataset/population | True variance | M-all R2/MAE | M-role R2/MAE |
|---|---:|---:|---:|
| vidstg/P_pool | 0.1813 | -0.0866/0.3619 | -0.2830/0.3964 |
| vidstg/P_A | 0.1629 | -0.1461/0.3619 | -0.2552/0.3792 |
| vidstg/P_W | 0.1442 | -0.2059/0.3335 | -0.4717/0.3729 |
| vidstg/R_pool | 0.1530 | 0.4630/0.2333 | -2.7541/0.6332 |
| vidstg/R_A | 0.0361 | -2.1293/0.2887 | -2.9330/0.3155 |
| vidstg/R_W | 0.0165 | -7.1752/0.2936 | -7.4156/0.3140 |
| hc2/P_pool | 0.1533 | 0.4854/0.2193 | -1.0601/0.4598 |
| hc2/P_A | 0.0671 | -0.4989/0.2753 | -0.5595/0.2822 |
| hc2/P_W | 0.0622 | -0.1877/0.2287 | -0.2889/0.2378 |
| hc2/R_pool | 0.1543 | 0.7145/0.1620 | -1.3980/0.4751 |
| hc2/R_A | 0.0509 | -0.7389/0.2217 | -1.3741/0.2579 |
| hc2/R_W | 0.0495 | -0.4458/0.1939 | -1.2609/0.2551 |

## Preserved work and failure cases

The cases are selected by fixed delta_v extremes and largest changed-decision
gains/losses, not chosen to hide exceptions. Full288 anonymous rows remain.

* Vid source36/frame_drop/order1: true delta_t=-.07438, delta_v=-5.4593pp.
  M-all predicts+.12718 and accepts; M-role predicts-.06450 and rejects.
  This is a genuine local severe-harm rejection.
* Vid source36/motion_blur/order1: true delta_t=+.25219, delta_v=+17.3445pp.
  M-all predicts+.28857 and accepts; M-role predicts-.05934 and rejects.
  The same source's useful correction is lost; source-specific effects differ
  by observation condition, so a global protection claim is unsupported.
* HC source34/frame_drop/order1: true delta_t=-.05974, delta_v=-4.4188pp.
  M-all's-.01012 correctly rejects; M-role's+.05565 now accepts. This undoes
  the predecessor's local relative-decision repair.
* HC source33/motion_blur/order1: true delta_v=-6.0932pp. M-role rejects the
  previously accepted severe harm; retain this positive case despite negative
  aggregate results. HC source33/frame_freeze's+5.0708pp correction is also
  rejected by M-role, while M-all accepted it.

## Evidence and decision

The simple hypothesis that replacing generic candidate training by deployment
A/W roles suffices to recover held-out quality is weakened in this fixed
setting. Search fitting is strong, confirmation role measurement remains
poor and relative execution does not improve consistently. It does not
identify a unique deeper cause: few independent sources, correlated repeated
cells, fixed source-selected alpha, population/normalization changes, role
label range and linear-family generalization remain competing explanations.

Pause further rule/threshold construction on this final-layer linear P/R
readout in its current configuration. Preserve A and CURRENT; no supervised
weight promotion. No automatic MLP, layerwise, motion/appearance, new expert,
loss or GPU phase is started. The result does not prove that the latent lacks
information, that P/R structure is useless, or that nonlinear alternatives
would succeed. No additional oracle ladder was introduced.

## Verification and resources

10 meaningful CPU controls pass. Root audit independently verifies eight
ridge systems (four reused M-all plus four M-role) using SciPy SPD solve,
all immutable inputs, training memberships, physical labels, unlabelled
prediction parity, state/choice hashes and every metric/bootstrap:
**236030 scalar checks**. Public audit verifies all
anonymous regression/acceptance/statistics without private latents, weights
or GT spans: **139916 checks**.

Four new scientific ridge fits are distinct from eight verification refits.
Scientific CPU wall: fit0.497s,
readout0.682s,
diagnose3.951s;
root5.190s/public3.274s
verification are separate. All new GPU/backbone/expert/candidate/replay/
backward/online-stream/production-update counts are zero.

Figures: [search/confirmation role MAE](../results/tastvg_pr_role/2026-10-03/figures/role_fit_generalization.png),
[fixed decision](../results/tastvg_pr_role/2026-10-03/figures/role_confirmation_decision.png),
[paired confirmation MAE](../results/tastvg_pr_role/2026-10-03/figures/role_paired_confirmation_mae.png).
All have PDF equivalents and source-data bindings. Full scalar outputs:
[anonymous rows](../results/tastvg_pr_role/2026-10-03/ROWS.json),
[Vid summary](../results/tastvg_pr_role/2026-10-03/vidstg/SUMMARY.json),
[HC summary](../results/tastvg_pr_role/2026-10-03/hc2/SUMMARY.json),
[decision](../results/tastvg_pr_role/2026-10-03/DECISION.json),
[protocol](../protocols/tastvg_pr_role_v1.md).
