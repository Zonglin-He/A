# Fixed native ReferDINO supervision: Direct / fresh Fixed / Live, HC128

Actual output-box-policy OPD with a fixed cached native teacher. No teacher refresh is required by this experiment. Candidates are sampled in native output-box logits and mapped to E-retaining coordinates; they are **not** independently decoded complete candidate trajectories. Every real state update does rerun the actual full decoder trajectory.

## Measured conclusion

With all three requested seeds41/42/43 retained, Live has a small positive development signal over E: full vIoU +0.517669pp [0.026719,1.084970], actual-node sIoU outside8 +1.138573pp [0.055994,2.424209]. It improves full vIoU over the strengthened fresh-Fixed control by +0.517677pp [0.057591,1.058830]; outside8 +1.036252pp [-0.000496,2.254606] remains uncertain. It has NOT established superiority over Direct: +0.054974pp [-0.033792,0.180305] full vIoU and +0.089274pp [-0.041594,0.237249] outside8. This configuration does not establish the requested2–3pp full-tube gain or an independent OPD advantage over simple direct native pseudo-box supervision.

Across seeds, Live−Direct full vIoU is +0.095429/+0.091708/−0.022215pp for41/42/43; all three intervals cross zero. Live−fresh-Fixed full vIoU is positive with positive parent intervals in all three seeds. Fixed has new samples every round, so the measured difference cannot be explained merely by reusing its first32 candidates. The result concerns this original likelihood and fixed-proposal control, with no claim of importance-corrected off-policy optimization or universal superiority over all fixed objectives.

All128 parents remain. For seed-averaged metrics, Live improves77/128, degrades43, and leaves8 unchanged in full vIoU;4 lose more than3pp and3 more than5pp. Worst losses: RC036−8.579918pp, RC126−6.645705pp, RC067−6.131660pp. Strong gains also remain: RC091+22.556714pp, RC097+15.537131pp, RC045+10.920319pp. Positive averages do not remove these opposing effects. No parent/step/seed gates were introduced.

Absolute scores (%), averaging seed METRICS only: E full vIoU27.210302/outside8 sIoU50.243917; Direct27.672998/51.293216; fresh-Fixed27.210294/50.346238; Live27.727971/51.382490. Live three-seed full-vIoU standard deviation is0.066873pp; this is descriptive over only3 seeds, not a population interval.

The sole postseal engineering recovery was a CPU assertion on averaging identical tIoU floats (1.11e−16 rounding). Real outputs still have exactly equal time/intervals; only derived seed-mean checking uses1e−14 tolerance. All original failure logs and pin amendments remain; no prediction, GT selection, numerical optimization setting, GPU rerun or scientific configuration changed.

Decision: preserve this result as a completed development comparison. Retain evidence that on-policy proposals help relative to this fresh-Fixed implementation, and the unresolved Live−Direct comparison separately. Do not promote production, scan parameters, run reserved32 or automatically launch a new interface experiment.

## Locked protocol

Same exposed128 HC parents/queries and8 observations, E+finalMBR time/Source/decoder cache as the conditional-reference probe. Teacher target ONLY native ReferDINO selected8 trajectory, unchanged across all methods, rounds and seeds. Previous conditional T(S0) Direct was not reused. Eight independent256D per-video content states, frozen pretrained weights, zero-state exact E;40 Adam steps,lr.01,32 antithetic candidates,sigma.025,tau.05. Geometry cost mean8[5L1+2(1-GIoU)] and original Gaussian likelihood analytic baseline/control variate. The soft weights are constructed geometric supervision, not ReferDINO native probabilities.

Direct recomputed once per parent (deterministic). Fixed samples NEW candidates from initial mean every round; Live samples NEW candidates from current decoded mean. Noise paired for each seed/parent/round. Seeds41,42,43 explicitly requested before locking; all reported. No scan, teacher/Source calls, new experts, new network training or observation positions. NH16 and reserved32 not run. All896 fits/128×8 output trajectories sealed before new GT evaluation.

## Primary parent-level effects

Average each stochastic method metric across3 seeds WITHIN each parent, then bootstrap128 parents (10000 resamples). This is an expectation over reported seeds, not a box-averaged ensemble and not384 independent videos. Intervals condition on these three seeds; they do not establish robustness to all RNG seeds. Units pp.

| Comparison | Full vIoU [95% CI] | Known actual sIoU outside8 [95% CI] |
|---|---:|---:|
| Direct_minus_E_MBR | +0.4627 [-0.0453, +1.0433] | +1.0493 [-0.0360, +2.3350] |
| Fixed_seedmean_minus_E_MBR | -0.0000 [-0.0683, +0.0694] | +0.1023 [-0.0440, +0.2720] |
| Live_seedmean_minus_E_MBR | +0.5177 [+0.0267, +1.0850] | +1.1386 [+0.0560, +2.4242] |
| Live_seedmean_minus_Direct | +0.0550 [-0.0338, +0.1803] | +0.0893 [-0.0416, +0.2372] |
| Live_seedmean_minus_Fixed_seedmean | +0.5177 [+0.0576, +1.0588] | +1.0363 [-0.0005, +2.2546] |

## Seed 41

| Comparison | Full vIoU [95% CI] | Outside8 actual sIoU [95% CI] |
|---|---:|---:|
| Fixed_41_minus_E_MBR | +0.0061 [-0.0646, +0.0803] | +0.1410 [-0.0101, +0.3210] |
| Live_41_minus_E_MBR | +0.5581 [+0.0653, +1.1265] | +1.1756 [+0.1082, +2.4588] |
| Live_41_minus_Direct | +0.0954 [-0.0087, +0.2451] | +0.1263 [-0.0497, +0.3295] |
| Live_41_minus_Fixed_41 | +0.5520 [+0.0921, +1.0982] | +1.0345 [+0.0142, +2.2399] |

## Seed 42

| Comparison | Full vIoU [95% CI] | Outside8 actual sIoU [95% CI] |
|---|---:|---:|
| Fixed_42_minus_E_MBR | -0.0002 [-0.0735, +0.0752] | +0.0840 [-0.0759, +0.2670] |
| Live_42_minus_E_MBR | +0.5544 [+0.0190, +1.1604] | +1.1225 [+0.0271, +2.4300] |
| Live_42_minus_Direct | +0.0917 [-0.0239, +0.2913] | +0.0732 [-0.0970, +0.3054] |
| Live_42_minus_Fixed_42 | +0.5546 [+0.0584, +1.1316] | +1.0385 [-0.0268, +2.2861] |

## Seed 43

| Comparison | Full vIoU [95% CI] | Outside8 actual sIoU [95% CI] |
|---|---:|---:|
| Fixed_43_minus_E_MBR | -0.0059 [-0.0705, +0.0556] | +0.0819 [-0.0544, +0.2403] |
| Live_43_minus_E_MBR | +0.4405 [-0.0274, +0.9878] | +1.1176 [+0.0378, +2.4015] |
| Live_43_minus_Direct | -0.0222 [-0.1456, +0.0583] | +0.0683 [-0.0318, +0.1621] |
| Live_43_minus_Fixed_43 | +0.4464 [+0.0083, +0.9676] | +1.0358 [+0.0046, +2.2561] |

## Tails and actual updates

All parents and known/unknown semantics retained; GT is only postseal evaluation. No per-parent output gates or best-step selection.

- Direct: positive77, negative43, unchanged8; worse than−3pp:4; worse than−5pp:3. All signed rows and best/worst5 preserved.
- Fixed_seedmean: positive75, negative44, unchanged9; worse than−3pp:0; worse than−5pp:0. All signed rows and best/worst5 preserved.
- Live_seedmean: positive77, negative43, unchanged8; worse than−3pp:4; worse than−5pp:3. All signed rows and best/worst5 preserved.
- Fixed_41: positive74, negative45, unchanged9; worse than−3pp:0; worse than−5pp:0. All signed rows and best/worst5 preserved.
- Live_41: positive79, negative41, unchanged8; worse than−3pp:4; worse than−5pp:3. All signed rows and best/worst5 preserved.
- Fixed_42: positive79, negative40, unchanged9; worse than−3pp:0; worse than−5pp:0. All signed rows and best/worst5 preserved.
- Live_42: positive76, negative44, unchanged8; worse than−3pp:4; worse than−5pp:3. All signed rows and best/worst5 preserved.
- Fixed_43: positive76, negative43, unchanged9; worse than−3pp:0; worse than−5pp:0. All signed rows and best/worst5 preserved.
- Live_43: positive79, negative41, unchanged8; worse than−3pp:4; worse than−5pp:3. All signed rows and best/worst5 preserved.

Actual state updates:35840; complete trajectory decoder calls:36864 (offset calls:73728); candidate geometry feedback evaluations:7864320. Per-method fit counts/losses in DECISION_PUBLIC.json; lower anchor loss or likelihood loss is not sufficient task evidence.

## Cost, audit and decision scope

New GPU allocation1556.748s, including model loading, guards, actual decode/backpropagation and artifact overhead. No new Source/ReferDINO/DINO forwards. Prior expert/image-encoding cost has already been paid and is separately identified; it is not free. Direct/Fixed/Live use equal optimizer-step budgets, while actual feedback/gradient compute can differ. Per-query and per-fit timing retained.

Independent saved-artifact audit reconstructs paired RNG/noises/actions, fresh Fixed batches, current Live centers, mapping, geometry costs/weights, likelihood gradients and all actual Adam updates; checks native target/zero state/time/physical support, independent full/outside8 metrics and all parent bootstrap intervals. This is not a second full GPU replication.

This is one fixed configuration on previously exposed development data. No seed/domain/parent winner selection. A negative setting is not a claim about all hyperparameters or all OPD; a positive development effect is not independent confirmation. Live must be judged against BOTH Direct and the strengthened fresh Fixed control, plus E and severe losses. No automatic parameter sweep, held32 evaluation, additional observations or production promotion.
