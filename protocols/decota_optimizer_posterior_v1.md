# DeCoTA Evidence → Optimization → Scope → Persistence

User authorization: attachment 8cf6bbae, complete execution of its conditional
route. This is a new isolated experiment. It does not promote CURRENT_METHOD,
overwrite C1/P0/P1, or restart paused full-query/historical queues.

## R1: complete matched optimizer/admission factorial and temporal posterior

Original fixed VidSTG/HC-STVG-v2 32 development +16 confirmation sources per
dataset; one query per source; two orders; clean and the original five 5%
corruptions. All sources have historical exposure. 576 unique pixels, 1,152
logical arrivals. Official same-domain EMA checkpoints and original sampling.
Each query has four cached DINO observations; this is not a 25% expert stream.

Spatial uses the **same frozen P1 arrival-before state** for every arm. Slow LN
inheritance is retained as that reference trajectory; current experimental
corrections are discarded. This controls the optimizer intervention without
mixing it with changed subsequent states. Independent online trajectories are
a later R5 experiment and must not be claimed from this R1.

Full 4×2 matrix: Direct, Critic-All, Critic-Admit, Critic-Top1 × Adam/SGD.
Direct retains old admission and single4 targets, 5 L1 +2 GIoU /planned4.
Critic-Admit applies the exact cached old accepted flag before the existing
multi-proposal IoU energy. Critic-Top1 additionally keeps the highest cached
target score (stable original index tie). All uses all valid cached proposals.
Each energy has temperature1/score-softmax1/mean valid observed frames.
1792 query/LN parameters, 10 steps and earliest own-objective minimum0..10;
query and optimizer reset each arrival. Native temporal output stays fixed.
All-Adam reuses the complete hash-verified P1 trajectory, not a copied mean.

Two additional All arms: Adam lr=.03×a; Adam proposal with final interpolation
theta_use=theta_initial+a(theta_selected-theta_initial). The latter runs ordinary
unmodified Adam internally, selects its own energy minimum, and interpolates
once after selection. It is a radial parameter interpolation, **not** a
functional trust-region guarantee. a is mean normalized DINO score entropy
complement over valid frames; singleton a=1, empty a=0. It is concentration,
not a probability of correctness. No loss scaling and no confidence gates.

Plain SGD has no momentum. A separate fixed lr is calibrated for each dataset
and objective using all32 development clean order1 initial states. Functional
displacement is mean absolute normalized cxcywh difference over the complete
sampled tube and all4 coordinates. Match median first-step displacement to
the corresponding Adam.03 first-step median. Log grid25 points1e-6..1e3,
first ascending target crossing, at most12 geometric bisections; choose smallest
absolute median error, then smaller lr. Preserve all trials and relative error;
if no crossing, label calibration approximate rather than hiding mismatch.
No GT, task metrics, confirmation examples, or second-order gradients enter
calibration. Final lrs sealed before factorial prediction.

Temporal is CPU, no model execution or gradient. Use cached original C1
actionness cost and exact legal s<e native span log probabilities, independently
for each of the original two offsets. Full q∝p0 exp(-cost), beta=1, prelocked
without GT tuning. Extent groups exact physical twice-centre frame_s+frame_e+1
and preserves p0 centre mass exactly, tilting lengths within that centre.
Report singleton group fraction: irregular grids may reduce available extent
changes. MAP centre may change despite unchanged centre marginal.
Native/Hard projected actionness target/Full/Extent/PM tau.9 all read out the
native two-offset envelope. No merged-grid fabricated native likelihood.
PM is an explicitly re-evaluated historical readout control, not a new method.
Spatial boxes fixed P1 before, and native boxes as secondary control.

All1,152 spatial and576 temporal payloads hash-sealed before CPU GT. Report
official dense v/t/s, interval recall/precision/duration; all flow, corruption,
clean, both orders, native-good/bad, duration thirds; source equal means and
paired10,000 source bootstrap (seed20261004). Log every proxy/gradient/update/
functional displacement/GT trajectory, correctly selected candidates harmed,
observed/unobserved support, positive and negative tails. GT is posthoc only.

## Conditional continuation

R2 picks one spatial optimizer family from **development** results, preserving
mean and reducing severe loss; confirmation is reporting, not re-selection.
If Full/Extent is not positive with positive paired confidence intervals on
both datasets' corruption panels, do not project an ineffective teacher into
the514-param head; mark T1 skipped by this precondition. This is a scoped
resource decision, not proof all temporal adaptation is impossible.

R3 exact <=81 track hypotheses uses cached DINO text score +native box IoU
+adjacent evidence IoU. Continuity is an identity proxy, not an identity
guarantee. Occupancy sampling requires a usable temporal posterior and a
separate observation/cache lock. R4 contrasts joint/u-only/different-LN
authority with explicit fixed definitions. R5 actual independent LN1/16 online
chains, R6 25/50/100% schedules and cross-domain qualification require preceding
usable mechanisms and their own sealed input/checkpoint locks. No stage is
called complete from a plan. Optional L-BFGS/Newton/new region-critic/duration
memory are expressly deferred in the attachment.

Preserve engineering failures and pin revisions. Do not retune lambda/beta,
endpoints, grids, samples or scores after confirmation. Public code, protocol,
anonymous results, negative cases, plots and exact remote receipt required.
Private media, annotations, weights, raw predictions and conversation excluded.
