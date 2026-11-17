# R129 measured interpretation

This is post-seal descriptive interpretation. No thresholds, steps, primary arm or model were changed after scoring. The prelocked complete primary failed; the unguarded ablation is not promoted after the fact.

## 1. Replacing exact probability fitting has a small positive signal

- DEV: unguarded partial vs zero +0.115181 pp; vs same-information direct partial +0.096817 pp. Scene-weighted counterparts +0.035172/+0.028939 pp. Improved/worsened 5/3; largest one-row positive contribution 0.081083 pp.
- EXP: unguarded partial vs zero +0.079678 pp; vs same-information direct partial +0.061572 pp. Scene-weighted counterparts +0.062846/+0.051401 pp. Improved/worsened 6/4; largest one-row positive contribution 0.036906 pp.

Conditional KL still declines; restricting support while fitting the full conditional ratios is insufficient. The partial-versus-conditional comparison changes loss form with the same supported observations. This supports retaining partial supervision as a research candidate, not asserting reliable identity supervision. The same-information direct partial control itself has only about +.018 pp in each cohort.

## 2. The endpoint constraint executes, but erases all observed positive F changes

The controlled partial->partial_guard comparison changes only the endpoint guard. It rejects 877 DEV and 1,772 EXP trial endpoints and reduces mean parameter displacement. It is not an inactive implementation. All five DEV and six EXP IoU improvements from unguarded partial disappear; all three DEV and three of four EXP losses also disappear. The remaining full-primary result is exact zero task change on DEV and a tiny net loss on EXP. This source-calibrated budget is too restrictive to retain this recipe’s observed ranking benefits; merely accepting nonzero parameter updates is not task adaptation success. These results do not establish that loosening it would generalize.

## 3. The teacher sets seldom challenge the actual first choice

Among the partial-active inputs, old q0 lies in S for 50 DEV and 123 EXP rows. C excludes it in zero and one of those rows, respectively. Thus this mechanism almost never provides a direct reason to displace the current first choice. It mostly removes lower-ranked alternatives and leaves internal ordering among plausible candidates undetermined. Among recoverable zero errors in this scope, C contains an Acc@25-capable candidate in 8/28 DEV and 17/70 EXP rows. This is post-score task-quality diagnosis, not a deployment filter. Spatial support did not become reliable language-instance identity merely through qualification.

## 4. Six-direction gains are not uniform or independently confirmed

In EXP the unguarded partial positive changes all occur in Drone→Quad (+.725573 pp); Drone→Waymo, Quad→Waymo and Quad→Drone decline, and the two Waymo-source directions have zero change. DEV has gains in Waymo→Quad, Drone→Quad and Quad→Drone. The recipe therefore does not establish stable improvement in all six directions. Scene weighting leaves the aggregate partial increment positive but does not create new scenes or independent confirmation.

On C, which was never fitted or used for acceptance, partial versus zero is DEV -0.036202 pp, EXP +0.128345 pp. It therefore does not show consistent unfitted-observation transfer across both cohorts.

The only wrong-binding update arm in the lock is partial+guard with shuffled groups. Its near-zero behavior cannot independently verify the mechanism of the unguarded positive arm. No new shuffled-unconstrained run is added after seeing GT. This is a concrete missing discriminating control, not permission to call the positive ablation a validated new method.

## Decision

Do not adopt the complete R129 recipe or resume tuning its safety threshold on these outcomes. Retain the unguarded partial ablation as a bounded positive development observation and retain every negative comparison. The next justified question, if separately authorized, is whether partial supervision contains valid candidate competition information under a matched unguarded correspondence control; improving teacher qualification must address instance identity rather than only point-count coverage. The current run ends without starting another algorithm or source training. FINAL and the registered main method remain unchanged.

## Resource boundary

One exclusive GPU. The formal joint suite took 1113.849 seconds including CPU work, loading and serialization. This is the multi-arm experiment cost, not a measured single-method deployment latency. All preceding global controls were reused. No external visual model or new raw PointNet backbone forward was added. Source thresholds use saved source outputs; this run performs no source retraining.
