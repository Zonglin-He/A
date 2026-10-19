# Scientific / engineering review

Native ReferDINO labels are fixed and common. OPD requires current student proposal feedback here, not changed teacher weights or changed teacher boxes. Fixed is strengthened to40 NEW random batches from initial policy, paired with Live in every round. Direct is newly run against native labels. Output-box exploration and actual deterministic full decoder trajectories are distinguished.

Seeds41/42/43 were explicitly selected by user before lock/GPU; all outcomes retained. Primary averages seed metrics per parent then bootstraps128 parents; no pseudoreplication as384 videos, no best seed. All predictions before new GT, unknowns and zero-time/no-original-reference cases retained, current original E/time exact. No source/teacher inference or learned new head. Old conditional supervision/crop teacher/held-first Fixed results remain historical, not substituted here.

A fixed numerical configuration is a mechanism comparison, not an optimal tuning claim. Different objective gradient scales can affect optimization. More candidate diversity is controlled for by fresh Fixed. Direct40 vs OPD40 matches steps but not all feedback computation; actual timing and counts reported. No stronger outcome conclusion follows from smaller loss, larger motion or teacher weighting alone.

## Measured conclusion

With all three requested seeds41/42/43 retained, Live has a small positive development signal over E: full vIoU +0.517669pp [0.026719,1.084970], actual-node sIoU outside8 +1.138573pp [0.055994,2.424209]. It improves full vIoU over the strengthened fresh-Fixed control by +0.517677pp [0.057591,1.058830]; outside8 +1.036252pp [-0.000496,2.254606] remains uncertain. It has NOT established superiority over Direct: +0.054974pp [-0.033792,0.180305] full vIoU and +0.089274pp [-0.041594,0.237249] outside8. This configuration does not establish the requested2–3pp full-tube gain or an independent OPD advantage over simple direct native pseudo-box supervision.

Across seeds, Live−Direct full vIoU is +0.095429/+0.091708/−0.022215pp for41/42/43; all three intervals cross zero. Live−fresh-Fixed full vIoU is positive with positive parent intervals in all three seeds. Fixed has new samples every round, so the measured difference cannot be explained merely by reusing its first32 candidates. The result concerns this original likelihood and fixed-proposal control, with no claim of importance-corrected off-policy optimization or universal superiority over all fixed objectives.

All128 parents remain. For seed-averaged metrics, Live improves77/128, degrades43, and leaves8 unchanged in full vIoU;4 lose more than3pp and3 more than5pp. Worst losses: RC036−8.579918pp, RC126−6.645705pp, RC067−6.131660pp. Strong gains also remain: RC091+22.556714pp, RC097+15.537131pp, RC045+10.920319pp. Positive averages do not remove these opposing effects. No parent/step/seed gates were introduced.

Absolute scores (%), averaging seed METRICS only: E full vIoU27.210302/outside8 sIoU50.243917; Direct27.672998/51.293216; fresh-Fixed27.210294/50.346238; Live27.727971/51.382490. Live three-seed full-vIoU standard deviation is0.066873pp; this is descriptive over only3 seeds, not a population interval.

The sole postseal engineering recovery was a CPU assertion on averaging identical tIoU floats (1.11e−16 rounding). Real outputs still have exactly equal time/intervals; only derived seed-mean checking uses1e−14 tolerance. All original failure logs and pin amendments remain; no prediction, GT selection, numerical optimization setting, GPU rerun or scientific configuration changed.

Decision: preserve this result as a completed development comparison. Retain evidence that on-policy proposals help relative to this fresh-Fixed implementation, and the unresolved Live−Direct comparison separately. Do not promote production, scan parameters, run reserved32 or automatically launch a new interface experiment.

