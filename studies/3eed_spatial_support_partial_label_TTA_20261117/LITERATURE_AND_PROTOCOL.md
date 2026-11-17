# R129: literature-to-grounding adaptation

The tested recipe is our spatial-support-conditioned partial-label adaptation,
not a reproduction of PASLE, Latte or DKD. The actual upstream source and
differences are recorded to make the scope testable.

- PASLE, ICLR 2025: [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/ac662d74829e4407ce1d126477f4a03a-Paper-Conference.pdf),
  [official repository](https://github.com/palm-ml/PASLE), inspected checkout
  `7ab3ceab72ca52143eebca9ba340d07a492e1f26`, `code/adapt_algorithm.py`.
  We transfer the candidate-label-set idea. Its implementation compares each
  probability plus a threshold to the largest probability. Its `thresh_gap=.1`
  is a progressive threshold-schedule range, NOT the margin used in R129.
  We do not transfer its fixed-class label cache, online progressive schedule,
  confident/unconfident buffer or theoretical guarantee to scene-local queries.
- Latte, ECCV 2024: [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04076.pdf),
  [official repository](https://github.com/AronCao49/Latte), reference checkout
  `15b7e69d646ba33153b634556bc82601993f10df`.
  We transfer local spatial qualification rather than treating an entire scene
  as an independent second observation. R129 does not have Latte's independent
  RGB/LiDAR segmentation branches, temporal voxel windows or class pseudo-labels.
- DKD, CVPR 2022: [paper landing page](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html).
  Decomposing distillation constraints is conceptual background only. R129
  implements no DKD label-conditioned loss and does not assume source Top1 is GT.
  The CVF endpoint returned access errors during this turn; no claim of a new
  complete DKD implementation/source audit is made.

## Exact grounding adaptation

We reuse R127's three epoch100 source-only checkpoints and R91 prepared states.
The deployed matching path includes frozen source-A RGB late features, original
normalization, token matching and source-A competition residual. Only point
Adapter Up weight/bias (4,896 scalars) change. All A256 boxes and six decoder
positions remain fixed; no new source training or external expert is introduced.
This tests front-end point adaptation under a fixed multimodal context, not
learning a new RGB/text/point identity binding network.
Here “source-A” means the initialized source-prepared network evaluated on the
current target input's A observation, not source-domain media reused as target
evidence. The context is frozen in value during that input's adaptation.

Candidate support comprises unique A seed coordinates inside its fixed box.
Valid B registrations count; fallback copies of A do not. A support location
is not an independent LiDAR return: registration can reuse B neighbors, and
neither registration nor support certifies object identity or absence of motion.
High overlap alone does not merge queries. Complete-link duplicate grouping
requires both box IoU >=.8 and support Jaccard >=.8. Group scores are log-mean-exp
of member query scores, so repeated query count does not create extra votes.
These groups are geometric hypotheses, not GT instance labels.

S requires at least four A support locations, four valid B support locations
and B coverage >=.5. There must be at least two groups. The .5 majority criterion
is fixed in advance; no source-B registration calibration was available.
Pooled saved source supervision calibrated the near-maximum probability gap tau
and cumulative unknown-output KL budget epsilon; target GT is absent from this
calibration. C is the union of A and B near-maximum sets within S, never forcibly
including old Top1. C=S produces exactly zero partial-label task supervision.
No reference relation is claimed verified by target-only box support.

Three normalized gradient steps and the original feature keep term are shared.
The endpoint unknown constraint collapses full query probabilities into all
unqualified query probabilities plus the total mass in S, then bounds
KL(g_initial || g_endpoint). It is checked at actual trial endpoints, against
the same initial distribution throughout all three steps, not merely penalized.
The shuffled control permutes groups within equal query-count bins, preserving
both group and query cardinality. Binning can leave some sets unchanged.

All new parameter states are fixed before C is decoded. C is an unfitted
registered observation, not an independent new scene or selection set.
DEV768/EXP1536 are historically exposed development cohorts; all 2,304
direction-inputs, including 154 protected aliases, remain in the denominator.
Protected aliases reuse the archived no-update result without new feature,
media or task-GT indexing. FINAL receives no new access or scoring.

## Interpretation boundaries

R127 global -> conditional comparisons change both spatial qualification and
duplicate-group normalization. They cannot isolate pure visibility causality.
Conditional -> partial and partial -> endpoint constraint are controlled
factor comparisons. R127 global states are reused, not rerun, on the same
qualified observations to distinguish changes in eligibility from loss form.
Same-information direct conditional and direct partial readouts preserve every
unqualified query probability and the total S mass, retaining within-group
source ratios. They can use all the same candidate-set information without
updating parameters. Target labels only audit sealed results and set quality.
Any loss reduction, nonzero update or fewer changed queries is insufficient
for adoption without net grounding benefit over those controls.
