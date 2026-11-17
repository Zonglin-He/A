# R130 — Fixed-support partial-label control and bounded radius sensitivity

**Original-rho primary: does not pass all prelocked comparisons; do not promote it to a validated mechanism.**

R129 remains unchanged: its protected complete primary failed. The user now explicitly selected its unguarded partial_F ablation for a new, bounded development study. R130 does not relabel R129’s original primary or use a new radius as its primary after scoring.

## Scope and reuse

Same three epoch100 source-only/R91 prepared states, native-foundation initialization, A256, six decoder anchors, registered A/B, complete F_AB matching reader and frozen initial-A RGB late/competition context. Only 4,896 point Adapter Up parameters change. Original feature keep weight1, normalized gradient3 steps and Armijo descent remain; only the unknown-output KL hard guard is absent. Source thresholds n4/registration fraction.5/tau.3356115962106995 remain fixed.

All 2,304 direction-inputs remain: DEV768/EXP1536, six directions, 154 protected aliases reuse zero without new feature/media/GT indexing. These are historically exposed development cohorts and directions can reuse frames. FINAL, source training, backbone architecture, external models, EMA and additional loss terms are unchanged/unused.

Original-rho partial_F, zero/direct readouts and archived controls are bit-identically reused from R129. Twelve active engineering replays reproduce the original parameter states exactly; this checks compatibility but is not a new independent performance confirmation. New neural inference runs only five update arms on 257 partial-active inputs: proper half/double and wrong half/base/double. C is decoded after every fitted state is fixed and is never an acceptance/selection input.

## Main results

Six-direction equal-weight mIoU in %, with Acc@25/50. No platform-specific, per-case or best-radius GT selection.

| Method | DEV mIoU | EXP mIoU | DEV Acc25 | EXP Acc25 | DEV Acc50 | EXP Acc50 |
|---|---|---|---|---|---|---|
| zero_F | 17.279671 | 16.418423 | 32.161458 | 30.989583 | 14.322917 | 12.044271 |
| direct_partial_F | 17.298035 | 16.436529 | 32.161458 | 31.054688 | 14.453125 | 12.109375 |
| proper_half_F | 17.292880 | 16.411944 | 32.161458 | 30.989583 | 14.453125 | 11.979167 |
| partial_F | 17.394852 | 16.498101 | 32.291667 | 31.184896 | 14.583333 | 12.044271 |
| proper_double_F | 17.196333 | 16.411401 | 32.031250 | 30.989583 | 14.322917 | 11.979167 |
| wrong_half_F | 17.279106 | 16.381167 | 32.161458 | 30.794271 | 14.453125 | 12.109375 |
| wrong_base_F | 17.293569 | 16.255043 | 32.291667 | 30.598958 | 14.453125 | 12.044271 |
| wrong_double_F | 16.923891 | 15.970533 | 31.510417 | 30.013021 | 14.062500 | 11.783854 |
| direct_wrong_F | 15.457328 | 14.370778 | 29.036458 | 27.408854 | 12.369792 | 9.765625 |
| R129_guard_F | 17.279671 | 16.416507 | 32.161458 | 30.989583 | 14.322917 | 12.044271 |

Primary is partial_F at rho0. Half/double are sensitivity endpoints only. A factor changes each proposed normalized step before the same Armijo backtracking, not a guaranteed final parameter-displacement ratio. Actual norms/accepted steps are reported below.

## Matched no-guard identity control

S is exactly fixed. Within each equal query-count group-size bin inside S, a deterministic seeded maximum-feasible disjoint swap exchanges retained and excluded labels. This preserves |C|, the number of member queries, each size-bin’s label counts and the update budget, and never uses GT or q0. Of257 active inputs,252 can be effectively shuffled;5 cannot and explicitly retain identical updates rather than being called mismatches. All rows remain in the denominator; an effective-only comparison is also reported.

Correct C is derived from initial A/B plausibility, while randomized C need not preserve its initial probability mass or q0. This is the requested matched support/cardinality experiment; a win demonstrates the value of plausible-set identity against arbitrary labels, not by itself RGB/geometry-specific identity reasoning.

| Cohort | Active | Effective shuffle | q0 in S | q0 in C | q0 in shuffled C | Initial conditional mass C | Initial mass shuffled C |
|---|---|---|---|---|---|---|---|
| DEV | 81 | 79 | 50 | 50 | 2 | 0.880629 | 0.028157 |
| EXP | 176 | 173 | 123 | 122 | 10 | 0.891044 | 0.049019 |

| Cohort | Subset | Correct arm | Control | n | Gain pp | Better | Worse | Unchanged |
|---|---|---|---|---|---|---|---|---|
| DEV | all | partial_F | zero_F | 768 | +0.115181 | 5 | 3 | 760 |
| DEV | all | partial_F | direct_partial_F | 768 | +0.096817 | 4 | 3 | 761 |
| DEV | all | partial_F | wrong_base_F | 768 | +0.101283 | 9 | 6 | 753 |
| DEV | effective_shuffle | partial_F | zero_F | 79 | +1.119736 | 5 | 3 | 71 |
| DEV | effective_shuffle | partial_F | direct_partial_F | 79 | +0.941205 | 4 | 3 | 72 |
| DEV | effective_shuffle | partial_F | wrong_base_F | 79 | +0.984622 | 9 | 6 | 64 |
| EXP | all | partial_F | zero_F | 1536 | +0.079678 | 6 | 4 | 1526 |
| EXP | all | partial_F | direct_partial_F | 1536 | +0.061572 | 7 | 5 | 1524 |
| EXP | all | partial_F | wrong_base_F | 1536 | +0.243057 | 18 | 9 | 1509 |
| EXP | effective_shuffle | partial_F | zero_F | 173 | +0.707426 | 6 | 4 | 163 |
| EXP | effective_shuffle | partial_F | direct_partial_F | 173 | +0.546670 | 7 | 5 | 161 |
| EXP | effective_shuffle | partial_F | wrong_base_F | 173 | +2.158012 | 18 | 9 | 146 |

## Radius sensitivity, with matched radius controls

| Cohort | rho factor | Correct mIoU | Wrong mIoU | Correct vs zero pp | Correct vs direct pp | Correct vs wrong pp | Better/worse vs zero |
|---|---|---|---|---|---|---|---|
| DEV | 0.5 | 17.292880 | 17.279106 | +0.013209 | -0.005155 | +0.013774 | 2/2 |
| DEV | 1 | 17.394852 | 17.293569 | +0.115181 | +0.096817 | +0.101283 | 5/3 |
| DEV | 2 | 17.196333 | 16.923891 | -0.083338 | -0.101702 | +0.272442 | 5/6 |
| EXP | 0.5 | 16.411944 | 16.381167 | -0.006479 | -0.024585 | +0.030777 | 2/1 |
| EXP | 1 | 16.498101 | 16.255043 | +0.079678 | +0.061572 | +0.243057 | 6/4 |
| EXP | 2 | 16.411401 | 15.970533 | -0.007022 | -0.025128 | +0.440868 | 6/7 |

All endpoints, including those below zero/direct controls, are retained. No requirement that all three radii or all directions improve was imposed, and no best endpoint replaces rho0.

| Cohort | Setting | Active | Nonzero | Accepted steps | Armijo rejects | Mean parameter L2 | Mean task before | Mean task after | Mean keep after |
|---|---|---|---|---|---|---|---|---|---|
| DEV | proper_half | 81 | 81 | 243 | 11 | 0.05055599 | 0.13814586 | 0.07294106 | 0.00088024 |
| DEV | proper_double | 81 | 81 | 243 | 73 | 0.15493821 | 0.13814586 | 0.02092002 | 0.00723046 |
| DEV | wrong_half | 81 | 81 | 243 | 0 | 0.05298444 | 8.18846036 | 7.46417925 | 0.00103504 |
| DEV | wrong_base | 81 | 81 | 243 | 0 | 0.10442429 | 8.18846036 | 6.80898944 | 0.00397730 |
| DEV | wrong_double | 81 | 81 | 243 | 1 | 0.19925518 | 8.18846036 | 5.69819303 | 0.01424181 |
| EXP | proper_half | 176 | 176 | 528 | 15 | 0.05103417 | 0.12773794 | 0.06498928 | 0.00087281 |
| EXP | proper_double | 176 | 176 | 528 | 139 | 0.14837014 | 0.12773794 | 0.01808853 | 0.00614398 |
| EXP | wrong_half | 176 | 176 | 528 | 8 | 0.05264729 | 7.89862905 | 7.25880000 | 0.00102909 |
| EXP | wrong_base | 176 | 176 | 528 | 11 | 0.10372803 | 7.89862905 | 6.64887931 | 0.00396207 |
| EXP | wrong_double | 176 | 176 | 528 | 12 | 0.19882987 | 7.89862905 | 5.63945341 | 0.01422025 |

## Six directions and scene-frequency weighting

| Cohort | Direction | Zero | Direct partial | Half | rho0 primary | Double | Wrong rho0 |
|---|---|---|---|---|---|---|---|
| DEV | waymo→drone | 18.193613 | 18.193613 | 18.193613 | 18.193613 | 18.193613 | 18.193613 |
| DEV | waymo→quad | 32.566884 | 32.566884 | 32.569930 | 33.019131 | 32.015199 | 33.574052 |
| DEV | drone→waymo | 14.711135 | 14.711135 | 14.711135 | 14.711135 | 14.711135 | 14.876586 |
| DEV | drone→quad | 16.631282 | 16.741469 | 16.707490 | 16.739598 | 16.739598 | 15.730516 |
| DEV | quad→waymo | 9.725965 | 9.725965 | 9.725965 | 9.725965 | 9.538785 | 9.725965 |
| DEV | quad→drone | 11.849145 | 11.849145 | 11.849145 | 11.979668 | 11.979668 | 11.660684 |
| EXP | waymo→drone | 15.090863 | 15.090863 | 15.090863 | 15.090863 | 15.070463 | 15.024276 |
| EXP | waymo→quad | 30.897831 | 30.897831 | 30.897831 | 30.897831 | 30.666890 | 30.744160 |
| EXP | drone→waymo | 14.718295 | 14.718295 | 14.718295 | 14.706796 | 14.706796 | 14.718295 |
| EXP | drone→quad | 16.943861 | 17.052497 | 17.134926 | 17.669434 | 17.400577 | 16.336757 |
| EXP | quad→waymo | 11.330441 | 11.330441 | 11.100500 | 11.100500 | 11.100500 | 11.166196 |
| EXP | quad→drone | 9.529250 | 9.529250 | 9.529250 | 9.523182 | 9.523182 | 9.540577 |

| Cohort | Control | rho0 raw gain pp | rho0 scene-frequency weighted gain pp |
|---|---|---|---|
| DEV | zero_F | +0.115181 | +0.035172 |
| DEV | direct_partial_F | +0.096817 | +0.028939 |
| DEV | wrong_base_F | +0.101283 | -0.004291 |
| EXP | zero_F | +0.079678 | +0.062846 |
| EXP | direct_partial_F | +0.061572 | +0.051401 |
| EXP | wrong_base_F | +0.243057 | +0.145341 |

Weights use existing full-cohort scene counts within covered scenes, not new full evaluation. Full target-platform, scene-cluster, Acc25/50, every arm and negative row are in results/tasks. They are not independent new-scene confirmation, and frame rows are not treated as independent scene replicates.

## All18 existing good/bad records

The predeclared audit revisits all11 improved and7 worsened R129 direction-inputs, not a favorable selection. These represent 18 physical frames and 6 target-scene clusters. The detailed anonymous case table records whether better-IoU candidates lie outside S, inside S but outside C, or remain inside C, plus ranks and the new correct/wrong/radius outcomes. Labels are used only after sealing for this diagnosis. IoU or geometric group membership alone does not establish actual instance identity.

| Where a possible IoU improvement is blocked | Records |
|---|---|
| C_has_better_candidate_but_internal_order_unconstrained | 15 |
| S_has_better_but_C_excludes_all_better | 1 |
| source_already_IoU_oracle | 2 |

Cases where the source is already the IoU oracle are reported separately. Acc25 coverage and continuous improvement opportunities are both retained; a small within-instance box change is not automatically called an identity error. No case-specific rule or update acceptance uses these diagnostic GT values.

## Unfitted C observation

| Cohort | Arm | C mIoU | vs zero C pp |
|---|---|---|---|
| DEV | zero_C | 16.492047 | +0.000000 |
| DEV | direct_partial_C | 16.565750 | +0.073703 |
| DEV | proper_half_C | 16.466769 | -0.025278 |
| DEV | partial_C | 16.455844 | -0.036202 |
| DEV | proper_double_C | 16.588932 | +0.096885 |
| DEV | wrong_base_C | 16.350880 | -0.141167 |
| EXP | zero_C | 16.204012 | +0.000000 |
| EXP | direct_partial_C | 16.276875 | +0.072863 |
| EXP | proper_half_C | 16.268633 | +0.064621 |
| EXP | partial_C | 16.332358 | +0.128345 |
| EXP | proper_double_C | 16.373588 | +0.169575 |
| EXP | wrong_base_C | 15.923976 | -0.280036 |

C is descriptive only. Different registered observations are not automatically independent scenes, and no C outcome chooses a radius or retained prediction.

## Verification and execution decision

Independent CPU matching/readout maximum probability error 1.79e-06; all reused references are bit-exact. Supports/cardinalities, every normalized parameter step, all fixed boxes and all new state tensors are audited. Independent oriented/batch IoU maximum disagreement 6.66e-16; all scalar means, paired counts and scene-frequency weights are independently recalculated.

The prelocked primary comparison is original rho0 versus own zero, same-information correct direct partial and same-radius unguarded within-S wrong labels, in both DEV/EXP raw and scene-weighted means. This gate does not demand six positive directions or three positive radii. Passing retains an exposed-development working candidate only; it does not claim robust external validation or change the registered production method. No extra model, loss, radius, candidate-set threshold or training experiment follows automatically. R129’s protected-primary failure remains recorded.
