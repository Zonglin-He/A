# Frozen critic feedback near E:64-query qualification

Same original HC32/Vid32 exposed development cohort. Three spatial states(Source, half E delta, full E) and final MBR plus three other highest-probability distinct native physical intervals. Old critic weights are frozen. No OPD, QR/LN update, expert call or training occurs in this stage. All candidate scores sealed before new GT.

| Domain | Tubes | E+MBR vIoU% | Joint-selected vIoU% | E-only time choice% | GT candidate oracle% | Joint-E pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 384 | 27.558379 | 28.755053 | 27.189778 | 35.068076 | +1.196674 [-1.302265,+3.937259] |
| vidstg | 384 | 22.453299 | 20.372082 | 23.426921 | 25.402220 | -2.081217 [-6.133241,+0.955098] |

| Domain | Spatial v-order mean [95% CI] | Temporal v-order mean [95% CI] | Temporal t-order mean [95% CI] |
|---|---:|---:|---:|
| hc2 | 0.629310 [0.471264,0.778736], n=29 | 0.482894 [0.379693,0.581744], n=29 | 0.473946 [0.369732,0.576245], n=29 |
| vidstg | 0.526316 [0.328947,0.719298], n=19 | 0.443677 [0.322465,0.568121], n=25 | 0.425287 [0.317615,0.534875], n=29 |

The unit is a query, not a candidate pair. All strict pairs are retained in each query record, exact critic ties receive half credit, GT-quality ties use the prelocked1e-8 tolerance; no strict pairs means unknown. Unsupported E queries are retained with duplicate spatial states. Spatial/time v-order here evaluates complete-tube quality; t-order separately tests event-boundary direction.

## Headroom and time-support checks

| Domain | Oracle-E pp | Time oracle at E-E pp | Spatial oracle at MBR-E pp | Joint tIoU change pp | Shortened queries | v up/t down |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | +7.509697 | +2.612814 | +4.088053 | -1.638908 | 15 | 5 |
| vidstg | +2.948920 | +2.383447 | +0.509860 | +0.427731 | 15 | 1 |

Complete per-query deleted/added actual decoder-frame counts, exact-known GT support, E sIoU on removed/retained/added locations, interval duration/start fractions, scores and all12 v/t/s metrics are public. Missing GT support stays unknown. A vIoU gain through discarding hard frames is not automatically event-boundary improvement. No GT oracle candidate is used for deployment.

## Prelocked decision

```json
{
  "status": "stop_current_critic_before_OPD",
  "advance_stage2": false,
  "ranking_both_domains_positive_evidence": false,
  "domain_Joint_nonnegative": false,
  "macro_positive_CI": false,
  "uncertainty_not_automatic_expansion": true,
  "not_all_critic_or_OPD_impossible": true
}
```

Equal-domain macro Joint-E: -0.442272pp,95% CI [-2.8348145175091815, 1.6503161494282197].

The advancement rule is a resource decision: query-mean spatial and temporal(v-order)95% lower>.5 in both domains, both domain Joint-E means>=0, macro paired lower>0. Uncertain ranking is not a pass and does not automatically expand data/change thresholds. Old source32/33 to target9/30 spatial failure stays recorded; changing the candidate neighborhood does not presume the old critic has become reliable.

If the gate fails, stop this critic before any optimizer experiment; do not search a temperature, reward, step size or new model to rescue this same feedback. This does not prove all critics, joint inference or OPD impossible. If it passes, only the prelocked source-scaled five-round external time2/space4 experiment is eligible, with fresh finite resource lock and first-two-case actual timings. This stage is not an OPD score or a backbone-agnostic system: the critic is source trained and consumes TA-STVG features.

## Actual cost and audit

Source caches held exact boxes/logits/E but no full maps for new regions, so one frozen Source capture per query was required; no historical Before replay or new expert. Cached Source ROI features were checked exact against the original critic extraction within each domain. Checkpoints, Source output/logits and frozen model states were verified. Native final-envelope candidates are independently rebuilt by direct two-offset enumeration. All64 rankings/metrics/bootstrap comparisons are independently audited.

[Complete anonymous results](RESULTS_PUBLIC.json) · [Exact configuration and source-only conditional stage2 lock](CONFIGURATION.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Actual cost](ACTUAL_COSTS.json) · [Qualification plot](FEEDBACK_QUALIFICATION.svg).

## Root interpretation and closure

Do not start stage2. The candidate set DOES contain improvements: full oracle−E+7.509697pp HC and+2.948920pp Vid. The fixed critic realizes Joint−E+1.196674pp HC (CI[−1.302265,3.937259]) and−2.081217pp Vid (CI[−6.133241,.955098]); equal-domain macro−.442272pp (CI[−2.834815,1.650316]). Both temporal and spatial primary ranking intervals include.5, so positive evidence is unestablished rather than statistically proven universal reversal.

At fixed MBR, Vid spatial selection loses−3.347768pp (CI[−7.069891,−.801421]), whereas the available spatial-oracle gain is only+.509860pp. This makes false spatial switching costly. HC Joint v improves but tIoU falls−1.638908pp; it is not evidence of better event boundaries. Fixed-E time selection−.368601pp HC/+ .973622pp Vid also does not establish a bidirectional event signal. The old critic should not be promoted just because the candidate neighborhood changed.

All64 queries remain in selection results. Ranking n is29HC spatial/29HC temporal,19Vid spatial/25Vid temporal(v); others have only GT quality ties and are explicitly unknown. Generic interval positive/negative/neutral fields on accuracy compare to zero, NOT chance; [chance-centered counts](ROOT_AGGREGATION_AND_CHANCE_AUDIT.json) give the proper above/below.5 counts.

Scientific budget:262.143521seconds shared-GPU occupancy and349.131845seconds from lock to scored gate (5.82minutes), within300/900limits; no budget reset. Root audit/report overhead is separate in ACTUAL_COSTS.0OPD fits,0new experts or training. First CPU score failed only while exporting NumPy integers to JSON; the original scorer/lock/log/controller failure are preserved and only scalar serialization was repaired. Original candidate/model/score outputs were not rerun or changed; CPU metrics were recomputed inside the original deadline. Initial audit was affected by the legacy GT loader changing CPU threads4→2; restoring measurement4 yields bit-exact all768 critic scores without relaxing tolerance. All failure evidence and the separate precision replay remain private receipts.

Stage2 parameters were source-locked before target GT, but stage2 was never started because its condition failed. No automatic threshold/model/data/optimizer search follows. This is a valid negative feedback qualification of THIS frozen source-trained critic on THIS exposed64/E neighborhood; it is not a failure result for an unrun joint OPD or proof against every event critic.
