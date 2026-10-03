# Source-held-out P/R readout audit

Both populations lose substantial accuracy when the evaluated search source
is excluded from fitting: all sixteen paired role MAE increases have intervals
above zero. M-role's in-sample R2 is .914-.996 in HC and .922-.948 in Vid, but
all eight M-role LOSO role R2 are negative. M-all does not provide a reliable
escape: HC's four R2 are negative; Vid precision retains small positive R2
(.130/.043), while its recall R2 is negative. This establishes a source-held-out
generalization gap inside the search cohort, so a separate confirmation-cohort
shift is not needed to reproduce the failure. It supports source sensitivity /
overfitting in this fixed readout configuration, without uniquely assigning
it to high dimensionality, literal memorization or representation absence.

The count curve offers qualified positive evidence in Vid: increasing12 to15
training sources reduces precision MAE for both populations (paired intervals
below zero), though intermediate points are nonmonotone and recall is weaker.
HC shows no consistent improvement from4 to13 and all12-to13 paired MAE
intervals cross zero. This short, single-sequence curve does not establish an
asymptote, or prove that more independent sources cannot help HC. Do not
jump to layerwise/MLP or a new gate from this diagnostic.

## Fixed cohort and supervision boundary

Predecessor c1558266eaefee13b745202f332f7f52bffa1748. Each dataset has96 cached
search expert arrivals (80 corrupt /16 clean), Vid16 /HC14 independent sources.
Original design32 search sources, one query/source, two orders, six conditions,
25% expert schedule; no nonexpert features added. All historically exposed.
Holdout applies to quality-readout supervision; cached persistent A states
can have encountered those sources without quality labels. No new confirmation
fit or evaluation; prior confirmation findings are not another endpoint here.

TA same-domain checkpoints Vidfbb1ed88 /HCee72f0d9, sixth temporal hidden,
Inside256 ->precision /Endpoint512 ->recall, fixed alpha Vid1/1 /HC1/.1.
M-all trains all32 candidates/cell, M-role ordered[A,W] with no-op duplicates.
Shared heads and equal-source weighted MSE with total weight1, unpenalized
bias and per-fold training-only normalization unchanged. A1792 Uniform
Rank-RKL VidK1/HCK8, Paper48 original pixels/two offsets, Old8/Expanded32,
A8/L32 winner W, expert schedule and spatial state trajectory stay fixed.

## Source holdout and nested count protocol

All conditions/orders/roles of one source are excluded together. Remaining
IDs have a locked SHA order per(dataset,held source). Nested4/8/12/remaining
source prefixes are shared by both models and P/R. Maximum means15 Vid /
13 HC sources, never includes the held source. One sequence per fold, no
subset search/repeats, no alpha/layer/threshold/dimension tuning.

Thirty source holdouts x4 counts x2 populations x2 heads =480 scientific CPU
fits. Eight prior heads provide in-sample controls, without scientific refit.
All models seal, then192 GT-free held-source scalar readouts seal before label
join. Extraction creates source-specific supervised packs; each fit denies
opening the held-source labelled pack. Independent verification refits are
counted separately. This is supervised diagnosis, not unsupervised TTA gain.

## Primary: raw search-corruption role regression

All regimes use identical test labels; MAE/MSE are unbounded predictions.
Conditions ->orders ->equal source. Whole-source10000 bootstrap seed20261003,
conditional on fixed OOF predictions and overlapping training folds; no
bootstrap refits or multiple-comparison adjustment. Undefined draws remain.
R2 compares with a test-label mean, not a deployed train-mean predictor.

### MAE

| Dataset | Role | M-all in-sample | M-all LOSO | M-role in-sample | M-role LOSO |
|---|---|---:|---:|---:|---:|
| vidstg | P_A | 0.1115 [0.0698, 0.1655] | 0.3296 [0.2414, 0.4240] | 0.0764 [0.0569, 0.0991] | 0.3610 [0.2630, 0.4703] |
| vidstg | R_A | 0.2580 [0.1862, 0.3411] | 0.4505 [0.3239, 0.5991] | 0.0729 [0.0507, 0.0987] | 0.4113 [0.2834, 0.5475] |
| vidstg | P_W | 0.1393 [0.0948, 0.1885] | 0.3854 [0.3047, 0.4667] | 0.0787 [0.0524, 0.1059] | 0.4070 [0.3091, 0.5110] |
| vidstg | R_W | 0.2514 [0.1931, 0.3082] | 0.4409 [0.3279, 0.5647] | 0.0828 [0.0584, 0.1102] | 0.3919 [0.2700, 0.5189] |
| hc2 | P_A | 0.1111 [0.0846, 0.1381] | 0.2182 [0.1536, 0.2848] | 0.0602 [0.0460, 0.0759] | 0.2448 [0.1665, 0.3241] |
| hc2 | R_A | 0.0657 [0.0430, 0.0896] | 0.2330 [0.1558, 0.3156] | 0.0130 [0.0099, 0.0164] | 0.2711 [0.1601, 0.4000] |
| hc2 | P_W | 0.1225 [0.0844, 0.1618] | 0.2451 [0.1694, 0.3242] | 0.0558 [0.0389, 0.0736] | 0.2480 [0.1560, 0.3434] |
| hc2 | R_W | 0.0842 [0.0590, 0.1108] | 0.2676 [0.1878, 0.3666] | 0.0139 [0.0072, 0.0232] | 0.2766 [0.1616, 0.4265] |

### R2

| Dataset | Role | M-all in-sample | M-all LOSO | M-role in-sample | M-role LOSO |
|---|---|---:|---:|---:|---:|
| vidstg | P_A | 0.8593 [0.6858, 0.9446] | 0.1295 [-0.5277, 0.4712] | 0.9481 [0.9070, 0.9677] | -0.0621 [-0.9265, 0.3755] |
| vidstg | R_A | 0.4176 [-0.3194, 0.6835] | -0.7530 [-3.0509, 0.0761] | 0.9404 [0.8702, 0.9711] | -0.5722 [-2.1595, 0.0698] |
| vidstg | P_W | 0.8358 [0.6932, 0.9177] | 0.0434 [-0.5878, 0.3392] | 0.9383 [0.8857, 0.9677] | -0.1321 [-0.9534, 0.3050] |
| vidstg | R_W | 0.4981 [0.0368, 0.6339] | -0.6054 [-2.6784, 0.0116] | 0.9222 [0.8646, 0.9564] | -0.4401 [-1.8014, 0.1519] |
| hc2 | P_A | 0.7241 [0.3927, 0.8213] | -0.1111 [-1.3766, 0.2574] | 0.9136 [0.7485, 0.9638] | -0.4061 [-2.3318, 0.1395] |
| hc2 | R_A | 0.8823 [0.0935, 0.9538] | -0.2364 [-6.2342, 0.2540] | 0.9956 [0.9618, 0.9985] | -1.1132 [-8.1117, -0.4941] |
| hc2 | P_W | 0.6935 [0.3820, 0.7879] | -0.1527 [-1.2830, 0.1596] | 0.9289 [0.8212, 0.9639] | -0.3052 [-1.8986, 0.1287] |
| hc2 | R_W | 0.8499 [-0.0304, 0.9132] | -0.3848 [-8.5289, -0.0975] | 0.9910 [0.9530, 0.9977] | -0.8903 [-7.5909, -0.5825] |

### RHO

| Dataset | Role | M-all in-sample | M-all LOSO | M-role in-sample | M-role LOSO |
|---|---|---:|---:|---:|---:|
| vidstg | P_A | 0.9490 [0.8853, 0.9864] | 0.4008 [-0.0616, 0.7274] | 0.9820 [0.9695, 0.9903] | 0.2507 [-0.2781, 0.6613] |
| vidstg | R_A | 0.7035 [0.3955, 0.9008] | -0.0962 [-0.4998, 0.4009] | 0.9739 [0.9405, 0.9898] | 0.0097 [-0.4371, 0.4440] |
| vidstg | P_W | 0.9407 [0.8887, 0.9799] | 0.2773 [-0.1876, 0.6322] | 0.9766 [0.9571, 0.9889] | 0.1565 [-0.3479, 0.6097] |
| vidstg | R_W | 0.7594 [0.5478, 0.9031] | -0.0689 [-0.4703, 0.3371] | 0.9693 [0.9430, 0.9871] | 0.1460 [-0.2906, 0.5274] |
| hc2 | P_A | 0.8869 [0.6745, 0.9592] | 0.1834 [-0.2469, 0.5688] | 0.9576 [0.8733, 0.9842] | 0.0911 [-0.3068, 0.4639] |
| hc2 | R_A | 0.9443 [0.6558, 0.9786] | 0.1800 [-0.5034, 0.5587] | 0.9979 [0.9860, 0.9993] | -0.2969 [-0.5916, -0.0044] |
| hc2 | P_W | 0.8971 [0.6781, 0.9725] | 0.0399 [-0.3880, 0.5022] | 0.9773 [0.9291, 0.9930] | 0.0014 [-0.3345, 0.4332] |
| hc2 | R_W | 0.9471 [0.7330, 0.9782] | -0.1706 [-0.6341, 0.1148] | 0.9963 [0.9881, 0.9991] | -0.5016 [-0.7251, -0.0818] |

## Held-source count curve: MAE

| Dataset/population | Role | 4 sources | 8 sources | 12 sources | Remaining sources |
|---|---|---:|---:|---:|---:|
| vidstg/all | P_A | 0.4495 [0.2822, 0.6449] | 0.4621 [0.3349, 0.5909] | 0.4266 [0.3543, 0.4994] | 0.3296 [0.2414, 0.4240] |
| vidstg/all | R_A | 0.5289 [0.3893, 0.6848] | 0.4455 [0.3083, 0.6029] | 0.4695 [0.3341, 0.6270] | 0.4505 [0.3239, 0.5991] |
| vidstg/all | P_W | 0.5127 [0.3585, 0.6809] | 0.4874 [0.3549, 0.6190] | 0.4913 [0.4264, 0.5551] | 0.3854 [0.3047, 0.4667] |
| vidstg/all | R_W | 0.5627 [0.4547, 0.6780] | 0.4279 [0.3196, 0.5422] | 0.4811 [0.3644, 0.6052] | 0.4409 [0.3279, 0.5647] |
| vidstg/role | P_A | 0.4548 [0.2942, 0.6404] | 0.4983 [0.3902, 0.6087] | 0.4433 [0.3439, 0.5494] | 0.3610 [0.2630, 0.4703] |
| vidstg/role | R_A | 0.5494 [0.4016, 0.7018] | 0.4149 [0.2737, 0.5750] | 0.4072 [0.2789, 0.5462] | 0.4113 [0.2834, 0.5475] |
| vidstg/role | P_W | 0.5028 [0.3383, 0.6780] | 0.4976 [0.3804, 0.6181] | 0.4860 [0.3905, 0.5822] | 0.4070 [0.3091, 0.5110] |
| vidstg/role | R_W | 0.4178 [0.3021, 0.5414] | 0.3173 [0.2356, 0.4151] | 0.4040 [0.2923, 0.5284] | 0.3919 [0.2700, 0.5189] |
| hc2/all | P_A | 0.2403 [0.1515, 0.3355] | 0.2235 [0.1602, 0.2859] | 0.2237 [0.1553, 0.2929] | 0.2182 [0.1536, 0.2848] |
| hc2/all | R_A | 0.2538 [0.1690, 0.3457] | 0.2454 [0.1653, 0.3302] | 0.2322 [0.1582, 0.3116] | 0.2330 [0.1558, 0.3156] |
| hc2/all | P_W | 0.2500 [0.1547, 0.3508] | 0.2484 [0.1789, 0.3236] | 0.2478 [0.1676, 0.3296] | 0.2451 [0.1694, 0.3242] |
| hc2/all | R_W | 0.2736 [0.1704, 0.3856] | 0.2915 [0.2043, 0.3902] | 0.2623 [0.1802, 0.3603] | 0.2676 [0.1878, 0.3666] |
| hc2/role | P_A | 0.2460 [0.1526, 0.3455] | 0.2236 [0.1385, 0.3112] | 0.2529 [0.1707, 0.3345] | 0.2448 [0.1665, 0.3241] |
| hc2/role | R_A | 0.2852 [0.1546, 0.4304] | 0.2548 [0.1394, 0.3856] | 0.2822 [0.1661, 0.4153] | 0.2711 [0.1601, 0.4000] |
| hc2/role | P_W | 0.2486 [0.1510, 0.3517] | 0.2339 [0.1412, 0.3316] | 0.2502 [0.1597, 0.3428] | 0.2480 [0.1560, 0.3434] |
| hc2/role | R_W | 0.3045 [0.1755, 0.4541] | 0.2779 [0.1555, 0.4305] | 0.2771 [0.1645, 0.4211] | 0.2766 [0.1616, 0.4265] |

## Paired gap and last count increment

| Dataset/population | Role | LOSO − in-sample MAE | Max − 12 MAE |
|---|---|---:|---:|
| vidstg/all | P_A | 0.2181 [0.1588, 0.2833] | -0.0970 [-0.1525, -0.0493] |
| vidstg/all | R_A | 0.1925 [0.1238, 0.2905] | -0.0190 [-0.0550, 0.0192] |
| vidstg/all | P_W | 0.2461 [0.1903, 0.3053] | -0.1059 [-0.1652, -0.0513] |
| vidstg/all | R_W | 0.1895 [0.1133, 0.2934] | -0.0402 [-0.0788, -0.0028] |
| vidstg/role | P_A | 0.2846 [0.1976, 0.3769] | -0.0823 [-0.1333, -0.0336] |
| vidstg/role | R_A | 0.3384 [0.2219, 0.4580] | 0.0040 [-0.0571, 0.0704] |
| vidstg/role | P_W | 0.3282 [0.2426, 0.4167] | -0.0791 [-0.1350, -0.0237] |
| vidstg/role | R_W | 0.3091 [0.1872, 0.4355] | -0.0120 [-0.0655, 0.0484] |
| hc2/all | P_A | 0.1071 [0.0647, 0.1518] | -0.0056 [-0.0167, 0.0046] |
| hc2/all | R_A | 0.1673 [0.1055, 0.2354] | 0.0008 [-0.0155, 0.0172] |
| hc2/all | P_W | 0.1227 [0.0795, 0.1683] | -0.0027 [-0.0149, 0.0086] |
| hc2/all | R_W | 0.1834 [0.1220, 0.2630] | 0.0053 [-0.0093, 0.0187] |
| hc2/role | P_A | 0.1846 [0.1083, 0.2621] | -0.0081 [-0.0232, 0.0030] |
| hc2/role | R_A | 0.2581 [0.1470, 0.3873] | -0.0111 [-0.0250, 0.0027] |
| hc2/role | P_W | 0.1922 [0.1121, 0.2735] | -0.0023 [-0.0214, 0.0169] |
| hc2/role | R_W | 0.2628 [0.1498, 0.4110] | -0.0005 [-0.0145, 0.0137] |

## Secondary fixed analytic decision

Clip only for F=PR/(P+R-PR). Keep the frozen eligible W iff predicted deltaF>0.
No new winner, threshold or gate. Utility reuses cached official dense scores
for expert arrivals; this is not a new full-stream prediction or25%-scaled gain.
Helpful/harmful uses true delta tIoU, severe harm is accepted delta vIoU<-.05.

| Dataset/model | AUROC | BA | Helpful/harmful accepted | Severe harmful accepted | Expert delta vIoU vs A (pp) |
|---|---:|---:|---:|---:|---:|
| vidstg/all/in_sample | 0.7568 [0.6020, 0.9167] | 0.7070 [0.5572, 0.8551] | 22/18 | 5 | 0.5039 [-3.2569, 3.9448] |
| vidstg/all/nmax | 0.4566 [0.2439, 0.6550] | 0.4213 [0.2856, 0.5417] | 17/29 | 10 | -1.2205 [-4.9270, 2.3107] |
| vidstg/role/in_sample | 0.7538 [0.6197, 0.8958] | 0.6976 [0.5716, 0.8157] | 21/14 | 2 | 1.6317 [-0.2936, 4.3430] |
| vidstg/role/nmax | 0.4436 [0.2805, 0.6090] | 0.5291 [0.3750, 0.6648] | 16/24 | 8 | -0.4789 [-4.2006, 2.9833] |
| hc2/all/in_sample | 0.9228 [0.7884, 0.9970] | 0.8187 [0.6250, 0.9625] | 28/8 | 0 | 2.9823 [0.4532, 6.4449] |
| hc2/all/nmax | 0.6341 [0.3514, 0.8698] | 0.6260 [0.4062, 0.8194] | 25/18 | 6 | 1.7980 [-1.2910, 5.6585] |
| hc2/role/in_sample | 0.8254 [0.5902, 0.9753] | 0.7250 [0.4524, 0.9500] | 26/10 | 0 | 2.7877 [0.2148, 6.2982] |
| hc2/role/nmax | 0.5335 [0.2641, 0.8109] | 0.5077 [0.2810, 0.7458] | 22/13 | 3 | 2.1250 [-0.3491, 5.6607] |

## Controls, interpretation and deliverables

Clean, both orders, individual conditions, source moments, frozen-prediction
delete-source influence and undefined metrics are in SUMMARY.json. All192
anonymous scalar rows and original A/W identity hashes are in ROWS.json;
CASES.json retains smallest/largest held-source absolute errors and both
directions of correction-decision changes, without filtering training labels.
The learning curve changes source count in one nested sequence, not only an
IID row count. No small-source curve establishes an asymptote or a guarantee
that more data fixes the mapping. Population differences include candidate
diversity, label prior and fitted normalizers. Fixing source-selected alpha
tests transfer of this configuration, not optimal regularization at every n.

No nonlinear/layerwise/dimension/alpha sweep or new expert is initiated. A
and CURRENT remain unchanged; quality weights are private diagnostic models.
Protocol, CONFIG/FOLDS/FIT_SUMMARY, anonymous predictions, complete metrics,
negative cases, ROOT/PUBLIC_AUDIT, resources and figures accompany this review.
Root review records the measured interpretation after these tables are read.


## Concrete work and failure cases

Vid source6/exposure/order2: M-role's mean four-role absolute error rises
.1305 in-sample ->.8772 held-out. True anchor P/R=.9733/1, but OOF predicts
.0545/-.0827; a previously rejected correction with delta vIoU=-21.5919pp
is accepted after analytic clipping. Vid source11/freeze/order2: M-all true
anchor and winner recall both.9643 become negative OOF predictions, losing a
helpful +10.2685pp correction. These failures survive within search-source
holdout, rather than requiring a separate confirmation panel.

Preserve work cases: Vid source3/exposure/order2 M-role held-out four-role MAE
.0824 and still rejects a harmful -7.2796pp correction. HC source14/freeze/
order1 M-all OOF MAE.0793 and still accepts +7.0399pp; M-role OOF MAE is even
smaller .0450 but rejects the same helpful correction. Thus low absolute error
need not yield correct relative zero-threshold choice. HC source26/drop/order1
has small OOF error and no-op stays A. HC source4/blur/order2 has true P/R=0
for both fixed intervals but role OOF estimates recall>1 and precision>.5;
this is a neutral candidate pair, retained in regression and excluded from
helpful/harmful binary counts, not silently labelled harmful.

M-role has fewer severe accepted harms than M-all (Vid10->8, HC6->3), but
does not establish safe benefit over A: expert delta vIoU=-.4789pp in Vid and
+2.1250pp in HC, with both intervals spanning zero. Vid role-minus-all utility
is +.7416pp [.0893,1.7718], a conditional exploratory improvement over another
imperfect readout, not a TTA promotion. All success/failure and clean/order
values are preserved. Different orders give different A/W contexts and the
same source-level split excludes both. The primary result remains absolute
and decision generalization loss, not an oracle-selected subset.

## Decision scope

All four in-sample-to-LOSO AUROC gaps (two datasets x two populations) have
negative paired intervals. Poor LOSO in both populations weakens a pure
all32-versus-A/W mismatch fix. However the paired M-role-minus-M-all absolute
MAE intervals all span zero, so this audit does not prove that reducing
candidate supervision is the unique variance cause. Fixing alpha tests this
family/configuration; dimensionality, regularization, finite independent
sources, candidate/label diversity and source-conditioned latent-quality
mapping remain competing explanations. No new dimension/regularization/data/
layer/nonlinear experiment is launched automatically. Retain A and CURRENT.


## Actual resource accounting

```json
{
  "new_scientific_CPU_fits": 480,
  "reused_in_sample_heads": 8,
  "extraction_CPU_wall_seconds": 0.5106444358825684,
  "fit_CPU_wall_seconds": 15.421034097671509,
  "readout_CPU_wall_seconds": 0.6305716037750244,
  "diagnosis_CPU_wall_seconds": 5.260108232498169,
  "GPU_calls": 0,
  "backbone_expert_candidate_replay_backward_online_production_calls": 0,
  "folds": 120,
  "private_model_files": 240,
  "search_expert_cells": 192,
  "independent_test_sources": 30
}
```
