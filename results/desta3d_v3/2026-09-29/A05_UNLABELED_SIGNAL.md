# A0.5 R16 Unlabeled Direction Signal Audit — completed / independently audited

Neither predeclared signal passed all three direction gates. No finite native intervention was run; stop internal self-supervised objective tuning and make expert pseudo-gradient qualification the next proposed stage.

Frozen official PTD4B/B1, union256 and A0.3 Train-only B16; C0=0 in the shared R16 coefficient field. Original v2 mild RGB brightness1.05/contrast.95 for consistency, observed input for entropy. Detached teacher and both student branches use the same original B1 semantic reference/interval/anchors/native block schedule. Spatial loss includes all 152775 classes, temporal includes actual observed endpoint classes. No GT-validity filtering of unlabeled coordinates. 64 actual backwards, 0 optimizer; no offline predictor training.

| Signal | Mean cosine | Median cosine | Positive temporal GT descent | Positive spatial GT descent | Direction gate |
|---|---:|---:|---:|---:|---|
| U-Consistency | -0.018640428 | -0.045876424 | 8/15 | 7/15 | FAIL |
| U-Entropy | 0.022217620 | 0.032152648 | 9/15 | 8/15 | FAIL |

Gate was locked at median>=.10 and both supported-branch positive descent fractions>=.65. Strict positive dots; missing GT support only changes that branch denominator. All16 queries and both signals retained. Zero direction has undefined mathematical cosine and neutral0 only for the declared gate; no fabricated supervision.

| Anonymized query | Consistency cosine | C temporal dot | C spatial dot | Entropy cosine | E temporal dot | E spatial dot |
|---|---:|---:|---:|---:|---:|---:|
| Q01 | 0.166256434 | 0.470544942 | -0.00765100767 | -0.345211696 | -0.69279789 | -0.0709322905 |
| Q02 | -0.208361532 | -0.137456056 | -0.20102889 | -0.773071071 | -2.50973992 | -0.399307197 |
| Q03 | 0.0603979545 | 0.237008835 | -0.00559852638 | 0.0512072613 | 0.7777312 | -0.0543780928 |
| Q04 | -0.244677786 | -0.0567961254 | 0.000436505452 | 0.936105936 | 0.119964873 | 0.1083701 |
| Q05 | -0.138678366 | -0.197641505 | -0.0103237372 | 0.470931256 | 0.743447107 | 0.0147573727 |
| Q06 | 0.170416951 | 0.0493324147 | 0.105272503 | 0.433247079 | 0.299251827 | 0.14898108 |
| Q07 | -0.0555293621 | 0.104617853 | -0.0250391021 | -0.624378601 | -0.721785155 | -0.0856738917 |
| Q08 | 0.160880038 | 0.482314081 | -0.044095484 | -0.128313831 | 0.463703816 | -0.141786975 |
| Q09 | -0.0486389942 | -0.0419334693 | 0.0199813433 | 0.380826643 | 0.0959808351 | 0.0468911355 |
| Q10 | -0.0895154959 | -0.641407684 | missing | -0.612047173 | -4.38551727 | missing |
| Q11 | 0.0777249954 | 0.0580623291 | 0.00620726632 | 0.0130980354 | 0.395611824 | -0.0363089494 |
| Q12 | -0.0581462566 | -0.286626449 | 0.0160861213 | 0.16942922 | 0.256331234 | 0.00836325156 |
| Q13 | -0.233016003 | 0.000859921096 | -0.10939296 | 0.104283196 | 0.0143801304 | -0.198311103 |
| Q14 | 0.187188155 | 0.156171616 | 0.00878953646 | -0.0310161145 | -0.80639397 | 0.025993687 |
| Q15 | -0.00143373418 | -0.238939061 | 0.00374183669 | -0.0899763814 | -0.585973978 | 0.00345424638 |
| Q16 | -0.0431138543 | missing | -0.0134919256 | 0.400368157 | missing | 0.125290063 |

The normalization order is explicit: requested unlabeled branch gradients are normalized *after projecting into C*, whereas the historical analytic oracle normalized full-F gradients before projection. Both use equal-branch balancing but are not the same numeric construction. The saved Oracle-R16 direction is reused unchanged; no new GT oracle backward. GT branch gradients are projected only in a separate CPU reference/readout path and are absent from signal computation.

For consistency, the mild-view gradient is transported to the observed common THW/R16 coordinate system. A positive GT local dot at observed F is a first-order diagnostic, not measured finite tube improvement. Entropy confidence can reinforce an incorrect answer. This screen does not establish universal failure of all self-supervision or impossibility of label-free adaptation.

Independent CPU validation: scalar losses max error 7.0283836e-07; first real-query coefficient chain relative error 2.0567662e-07; raw norm/direction error 1.3322676e-15; original GT projection relative error 2.7493711e-16. Second Torch/standard-library aggregation checked 108 values with max error 5.7731597e-15. Observed replay logits exact at every available branch, physical input/context identity and frozen parameter scope verified. Complete mild student logits and all coefficient gradients are sealed locally.

Engineering failures retained: initial synthetic identity-KL gradient residual1.163e-10 exceeded an arbitrary1e-10 test bound; fixed before GPU to a dtype-aware bound. An auditor syntax typo prevented preflight receipt creation, and a subsequent prepare stopped with partial config/input/basis files; original code/partial preserved and fresh registration used. Neither was a GPU run or a scientific gate change. No GPU failure/replay.

Measured GPU worker/nonoverlap wrapper total 391.474111898s; cumulative 73197.77228018310s cap=null. CPU preparation/audits separately recorded. Free disk 128647352320bytes exceeds8GiB. All research workers exited, no artifact deletion.

Next proposal only: TVG/SVG expert pseudo-target gradient qualification on locked native support and the same R16 reference. It needs its own provider/provenance/token-support protocol before running. No additional entropy/augmentation/weight variants, iterative updates, full447/full618/fresh388/target or OPD were started.
