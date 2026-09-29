# TA-STVG C0.5 and C2-T: fixed development qualification

Registered before new predictions. User attachment 9f8509aa authorizes these
two sequential iterations only. No adaptation, gradients, spatial expert, S1,
OPD, cross-domain tuning, or production promotion.

## Question and inherited evidence
Old C0: all six video-damage CIs crossed zero. C1 temporal candidate oracle
gain +14.431 pp, clean +14.810 pp; spatial +.705 pp. Separate corruption damage,
candidate support, and critic ranking. Original exposed VidSTG-test 32 parents,
one query each; same VidSTG-source TA checkpoint, same original 20–200 observed
frames, native two offsets, original precision, no new source/confirmation set.

## C0.5: STVG sampled-grid adaptation of temporal-corruption concepts
Official reference: https://arxiv.org/abs/2403.20254 and
https://github.com/Alvin-Zeng/temporal-robustness-benchmark/tree/a46eee452222fa67958c81c49496e712dedefeea
Official config uses 1/5/10 percent continuous frames in the annotated action
center. This experiment is NOT the official THUMOS corruption implementation.
On the student's observed grid, take ceil(p * number of observed GT-event
frames), minimum one, centered in the eligible ordered observations, early
tie. Record actual fraction and duplicate severities. No frame resampling.
Only GT interval is used to generate the panel, never GT boxes or model errors.
No GT is an input to frozen student or critic. Existing labels are exposed
development data; do not claim this session never accessed annotations.

Five fixed operators applied only at those positions, severity changes only
count: black RGB0; motion blur linear 31-pixel kernel scaled by min(H,W)/224,
seeded angle[-45,45], reflect boundary; occlusion seeded rectangle H/2 x W/2
(25% image area), black, no GT object location; overexposure RGB+100 clipped;
packet-loss simulation 20 seeded neighboring-observation block copies of
H/12 x W/3, alternating previous/next clean donors, no frame deletion.
These are own deterministic operators, especially motion/occlusion/packet loss
differ from the reference Wand/hand-overlay/glitch implementations. No result
based amplitude changes. 32 x 15 =480 cells, clean32 exact reuse; identical
operators/selected positions across severity may reuse output with explicit receipt.

Primary endpoint is equal-family/equal-severity corrupted-minus-clean tIoU and
corrected vIoU, averaged within each parent before 10,000 paired parent bootstrap,
seed20260929. Qualifies for continued benchmark development if BOTH 95% CI upper
bounds <0. Also report each family/severity, sIoU, >5pp damage, actual counts,
and finite-panel uncertainty. Per-condition positives are descriptive, no picking
the worst condition as main after seeing results. This 32-source development
qualification does not establish a final paper benchmark.

## C2-T: same frozen C1 candidates, after C0.5 analysis
Exactly original first16 x clean/noise_medium/defocus_medium/jpeg_medium=64;
no changes to already sealed candidate generation, no C0.5 result-based selection.
UniversalVTG existing best.pth plus PE-Core-L14-336, unifier disabled, frozen,
same sampled corrupted RGB as student, uniform 2fps positions within observed
extent with nearest existing observation, feature-time conversion fps=None.
No clean feature reuse for corrupted inputs. Text may be cached per query.

Fixed critic bridge r(I)=max_j sigmoid/logistic-confidence(j)*tIoU(I,J_j),
where J are pre-NMS expert proposals with release thresholds and topk. Convert
raw proposals explicitly through expert._convert_segments_to_seconds and clamp
to observed extent. Teacher intervals NEVER replace student candidates. This
tests this particular proposal-overlap bridge, not a native calibrated candidate
likelihood or all possible UniversalVTG critics. Native candidate index0 wins ties.
Cache expert outputs once. No GT-based score tuning, threshold, or fallback.

Report Native, Expert-selected, Candidate Oracle, exact uniform expectation;
tIoU and vIoU, clean gain, three-corruption within-parent macro gain, and paired
excess=(corrupted gain-clean gain). Pairwise: exclude GT ties <=1e-12; expert
ties <=1e-12 score .5. Average pairs within cell, then conditions within parent.
Report overall and low/middle/high score-gap terciles set from all64 cells'
unlabeled score gaps before scoring. Cluster uncertainty by16 parents; report
counts and parents per bin, bootstrap excludes empty parent/bin observations.
High-margin accuracy is exploratory qualification, not a learned reliability gate.
Next-OPD resource gate: corrupted tIoU gain CI lower>0 and high-margin pair
accuracy CI lower>.5; clean/corrupt excess and negative tails still constrain claim.
Failure stops OPD implementation in this turn, not all critic routes.

## Resource and checks
Cumulative GPU-process cap3600s per phase; new artifacts<=8GiB, free>=8GiB.
CPU semantic tests; clean native bitwise parity for a smoke; pixel invariance
outside selected mask; all predictions/critic scores sealed before scoring;
GT boxes scoring only; dual implementation metrics; independent scalar/selection
and pairwise readback; model state hashes unchanged. Failures retained.
Publish sanitized code/protocol/complete scalar results and verify remote bytes.
