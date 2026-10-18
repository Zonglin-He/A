# One DINO-supported supplement to joint four-parameter correction

## Result and fixed decision

Adding source-calibrated relative DINO support improves HC and improves over geometry-only C in Vid, but does NOT restore Vid performance relative to Top1 mean A. The single supplement fails the prelocked two-domain mean screen. Stop this family of simple geometry+DINO-score combination rules; no further beta/transform/kernel/gate search, independent confirmation, dense OPD or production change. Four-parameter correction remains a research representation; this is not a universal rejection of all correction methods.

## What changed, and what did not

E is the only new formal method group. It reuses original exposed HC16/Vid16, saved Before query-start trajectories with inherited state, Source sensitivity, four planned Native positions, ORIGINAL admitted56/50 frames and168/144 retained candidates, constant(cx,cy,logw,logh) correction, full-trajectory scope and final-distribution MBR. Frozen A/B/C/D outputs and scores are reused unchanged; no old method is rerun as a new experiment.

Only the candidate cost changes: l_ik=log(s_max)-log(s_ik), with1e-12 numerical floor; E minimizes sum_i min{lambda,min_k[d_ik²+beta*lambda*l_ik]}. The old geometric scale and lambda remain bit-identical; beta is one dimensionless value shared across domains. Top1 relative cost is0; low-score candidates remain available, without a.35 filter. These scores are original DINO target-token occurrence support (context branch) or its original non-context score, not complete-event/role judgments, calibrated identity probabilities or geometric variances. No Student-distance prior is added.

Exact finite candidate/outlier assignment enumeration is used. For fixed assignments the added score cost is constant in delta, so the conditional arithmetic mean remains the exact four-parameter optimum. Neither energy reduction nor semantic-score improvement is treated as a task metric.

## Source-only choice before any new target output

Original source-train72 is used only for score-range/pool/transform diagnostics, not new network training. The original exposed source-validation12/domain chooses ONE shared positive beta. The complete predeclared grid is{1/16,1/8,1/4,1/2,1,2}; beta0 is an explicitly reported reference control, not an eligible supplement. All source-grid outputs are sealed before source evaluation. Selection maximizes equal-domain/equal-query unobserved actual-Decoder sIoU gain; fixed source Native full-v is only a tie-break. The source domains have no parent overlap with target32. No target score, old identity label, per-domain beta, temperature or target gate is used in generation.

| beta | Pooled source-validation unobserved gain pp | Pooled source-validation Native full-v gain pp |
|---:|---:|---:|
| 0 | +7.550803 | +5.266044 |
| 0.0625 | +7.882438 | +5.324103 |
| 0.125 | +4.111423 | +3.120281 |
| 0.25 | +4.111423 | +3.120281 |
| 0.5 | +4.640784 | +3.120281 |
| 1 | +4.632447 | +3.115179 |
| 2 | +4.075265 | +3.036632 |

The selected beta is1/16, also the smallest predeclared positive value. No lower-beta extension follows. The source pooled gain increases7.550803 ->7.882438pp relative to beta0; this is calibration evidence, not independent source validation or proof of target transfer. All individual source rows and both domains are disclosed in SOURCE_BETA_CALIBRATION.

## Primary target results: changes against original Before (pp)

| Arm | HC full ΔvIoU | HC unobserved ΔsIoU | Vid full ΔvIoU | Vid unobserved ΔsIoU |
|---|---:|---:|---:|---:|
| A_mean | +0.689673 | -2.675767 | +1.931277 | +6.176844 |
| B_robust_top1 | +0.114785 | -3.957152 | +1.160544 | +5.571481 |
| C_joint | +2.902905 | +1.566093 | +0.713212 | +2.642743 |
| D_C_center | +2.675869 | +2.074503 | +1.058173 | +2.810908 |
| E_DINO_support | +5.591755 | +5.866705 | +1.174127 | +5.617031 |

E full-vIoU is29.161837% HC /9.668131% Vid; unobserved actual-frame sIoU44.539224% /46.884240%. Before baselines are23.570083% /8.494004% full-v, not raw Frozen. All16 parents/domain remain, including Vid two no-reference no-ops. Time is identical across arms. Missing GT is unknown, not zero. The observations use actual Decoder frames; benchmark interpolation is reported separately.

| E comparison | HC full-v pp [95% CI] | HC unobserved pp [95% CI] | Vid full-v pp [95% CI] | Vid unobserved pp [95% CI] |
|---|---:|---:|---:|---:|
| E − Base | +5.591755 [+0.469179,+13.515140] | +5.866705 [-0.724748,+14.595241] | +1.174127 [-0.254078,+2.701257] | +5.617031 [-0.924404,+14.301241] |
| E − A_mean | +4.902082 [+0.482233,+10.467746] | +8.542471 [-0.290619,+19.601415] | -0.757150 [-1.710875,-0.030887] | -0.559813 [-3.514433,+2.896704] |
| E − B_robust_top1 | +5.476969 [+0.687603,+11.551898] | +9.823856 [+0.691266,+21.262863] | +0.013583 [+0.000000,+0.040750] | +0.045550 [+0.000000,+0.136650] |
| E − C_joint | +2.688849 [-0.365473,+6.898466] | +4.300612 [-2.129612,+13.128478] | +0.460915 [+0.000000,+1.382745] | +2.974288 [+0.000000,+8.922863] |
| E − D_C_center | +2.915886 [-0.873758,+7.577414] | +3.792202 [-4.021200,+13.296006] | +0.115954 [-1.657376,+1.529498] | +2.806123 [-1.655661,+8.787807] |

E retains HC candidate benefit and is above C in both domains, but Vid E−A remains−.757150pp full-v/−.559813pp unobserved. Vid E−A full-v95% interval[−1.710875,−.030887] is negative on this exposed panel; it is still not independent confirmation. The locked mean decision fails because Vid A gain was not restored, not because an arbitrary all-CIs-significant condition was imposed. Source-start sensitivity has the same E−A directional conclusion.

## Candidate changes versus accepted-frame changes

A prelocked two-factor diagnostic crosses final C/E candidate indices with C/E accepted-frame masks, refits the conditional four-parameter mean and globally applies it. For rejected frames only, an unaccepted latent best candidate is defined at each final estimate; it is not an actual accepted reference. All diagnostic outputs seal with E before target GT. Endpoints exactly reconstruct C and E; hybrids are not selectable method groups. Finite two-factor Shapley values average both replacement orders. This algebraic attribution depends on that latent-index convention and is not a unique semantic/causal decomposition.

| E−C attribution | HC full-v / unobserved pp | Vid full-v / unobserved pp |
|---|---:|---:|
| candidate_choices | +2.688849 / +4.300612 | +0.513459 / +3.298362 |
| accepted_frame_mask | +0.000000 / +0.000000 | -0.052544 / -0.324075 |

HC E and C both retain all56 admitted frames:10 candidate choices change, with0 acceptance changes. All HC E−C gains therefore come from the choice change under this fixed model. E still chooses18 non-Top1 boxes,16 below.35; this is not hard restoration of the old winner threshold. It preserves H002/H010 gains and removes C’s H011 failure, but also loses H008 benefit. H009 remains a severe failure.

Vid E and C both reject9/50 frames, but their masks are NOT identical: one previously accepted frame is rejected and one rejected frame is accepted. There is one candidate change on commonly accepted frames. The finite decomposition attributes+.513459pp full-v/+3.298362pp unobserved to candidate changes and−.052544/−.324075pp to acceptance changes, summing exactly to+.460915/+2.974288. Equal rejection counts do not imply equal rejected frames or zero acceptance effect.

All Vid E−C metric gain is concentrated in V010: full-v gain over Base+1.255 ->+8.629pp, unobserved+7.921 ->+55.510pp. The other15 have unchanged saved metrics. E is also equal to B in saved metrics on15/16 Vid cases, with only+.013583pp cohort full-v/+ .045550pp unobserved over B. Thus the remaining Vid deficit is close to the earlier Top1-robust deficit, rather than evidence that all remaining loss is caused by switching candidates. No new cutoff adjustment is attempted.

## Reference quality, negative tails and uncertainty

| Domain | Exact-known references | Pool IoU>=.5 | Top1 good | C selected good | E selected good | E selected known |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 43 | 30 | 20 | 22 | 24 | 43 |
| vidstg | 24 | 22 | 11 | 8 | 9 | 21 |

These are postseal geometric IoU counts, not newly certified participant/role labels. Alternative boxes receive no old Top1 identity tag; six old role-unknown queries remain unknown. Candidate scores and frame acceptance improve some references but do not certify a correct whole-tube correction.

HC E is positive versus Before on14/16 full-v queries, versus12/16 C; full-v<-5pp harms are1(E),2(C),5(A). E removes H011’s C−22.536pp full-v/−59.327pp unobserved collapse, restoring A’s+1.054/+1.425. It preserves H002+6.818/+8.074 and H010+2.017/+1.692. It retains H009−10.169/−22.027 and H003−2.630 full-v. Corrected positive/negative examples are all retained.

HC E−Before full-v95% interval is[+.469179,+13.515140]pp while unobserved interval[−.724748,+14.595241] crosses0. Its saved-score leave-one full-v range[+2.156914,+6.642484] and unobserved[+2.297279,+7.726283] are positive, but this is fixed inherited output sensitivity, not stream replay or independent confirmation. Vid full-v severe harms remain1(E) versus0(A), notably V007; E does not restore all A successes. All signs, severe tails, per-parent values, source sensitivity and strata are public. No subgroup is selected as an operational rule.

## Verification, recovery and cost

Independent auditing checks232 conditional-assignment games/45090 assignments, 351 source metric/selection checks, 320 output reconstructions (maximum error0.0), 1280 metric checks, 384 factorial identities, 426 old-input/ABCD identities and324 aggregate/CI checks. The source-selected beta is independently reproduced. All old inputs/pins and CURRENT/MAIN/PAPER are unchanged.

The first complete CPU scoring pass failed ONLY at strict JSON serialization because a NumPy int64 behavior count was not a native JSON scalar. No model or output generation was repeated. The original pinned scorer is unchanged; a separate json002 wrapper converts NumPy scalars at the JSON boundary and repeats postseal CPU scoring for export. The original failure and wrapper SHA are preserved, and this repetition is not counted as independent evidence. Reported serialized-scoring cost excludes the lost first pass; no online latency claim is made.

Source calibration CPU1.001829s, cached target generation0.804256s, successful serialized scoring pass3.294471s. Original feature/expert work, preflight, repeated first scoring pass and audits are separate. New GPU/model/expert/training/Student-update/new-visual-label counts are0.

The permitted finding is narrower than method adoption: relative DINO candidate support contains useful information discarded by C, especially in HC, but this fixed simple combination still fails to recover Vid Top1-mean performance. Stop this low-cost selector-rule family without declaring all four-parameter correction impossible, without choosing E for HC/A for Vid, and without additional dense OPD or source/target tuning.

[All results](RESULTS.json) · [Fixed decision](DECISION.json) · [Full source beta calibration](SOURCE_BETA_CALIBRATION.json) · [Reference diagnostics](DIAGNOSTICS.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Figure](DINO_SUPPORT.svg)

Original query/parent text, GT coordinates, raw candidates/correction vectors, media, weights and personal records are excluded from public files.
