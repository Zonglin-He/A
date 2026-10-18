# Final posterior decisions and fixed-budget observation allocation

Both prelocked development gates stop on the original exposed64-parent panel. Final-distribution MBR improves on Native in both domains but does not meet the unified replacement criteria. Greedy posterior coverage does not improve the real-event four-frame allocation sufficiently and increases important zero-hit risks. No extension, new DINO, new Student fit or later-stage experiment is triggered.

## T0: expected utility after merging the physical distribution

The old256 experiment already constructed the merged two-offset distribution q and took its MAP. The old cached64 MBR chose each offset separately and merged two hard decisions. This experiment isolates the remaining order of operations: build the same full q first, then maximize expected physical interval tIoU over all final-grid legal decisions. q is an uncalibrated independent-view working model; correlated offsets are not newly calibrated. This is standard finite expected utility, not a new MBR theory.

| Target / fixed After | Native v/t % | Projection v/t % | Merged MAP v/t % | Offset MBR v/t % | Final MBR v/t % |
|---|---:|---:|---:|---:|---:|
| hc2 | 22.500215 / 43.574672 | 26.323341 / 52.823955 | 25.798190 / 52.383898 | 25.084289 / 51.378361 | 25.730505 / 52.061780 |
| vidstg | 16.861480 / 34.547397 | 18.940866 / 38.454197 | 19.765989 / 39.447451 | 17.432508 / 36.390281 | 19.356644 / 39.370565 |

| Final MBR paired ΔvIoU pp [95% parent CI] | vs Native | vs projection | vs old merged MAP | vs old offset MBR |
|---|---:|---:|---:|---:|
| hc2 | +3.230290 [+0.001103, +6.400661] | -0.592837 [-2.215732, +1.113612] | -0.067686 [-2.141620, +1.674622] | +0.646215 [-0.636795, +1.869319] |
| vidstg | +2.495164 [+0.734275, +4.717198] | +0.415778 [-0.609475, +1.538345] | -0.409346 [-1.544217, +0.540162] | +1.924136 [+0.745570, +3.270242] |

The mean threshold is at least +0.5pp versus Native and at least −0.5pp versus projection in each target. HC misses the projection margin by .092837pp. The wide paired interval does not prove inferiority or noninferiority. Final MBR has lower mean vIoU than old merged MAP in both targets, with both differences crossing zero; neither new extra value nor stable dominance by MAP is established. It improves over the old offset-MBR means, so those two MBR orders must remain distinct.

HC fast-motion Q4 has8 parents: final MBR−projection is −3.304065pp [−6.468692,−.578871], triggering the prelocked −1pp mean risk. HC short0/small3 and Vid short7/fast4 are UNKNOWN under the8-parent evaluability floor, never pass. Vid small8 passes this descriptive risk test. These are exposed-panel gates, not final multiplicity-adjusted superiority claims.

The primary geometry is the same saved original online After in all arms. Source and Before sensitivity, all signed parent differences, event/start/motion/target-size bins, zero temporal intersections, added/lost known support, known IoU and unknown support remain in RESULTS. sIoU at fixed geometry stays unchanged when the temporal readout changes. Unknown frame support is not a negative identity label.

A prelocked T0 After extension cannot be executed from matching original P1 payloads because those full fit/After caches were retired. Substituting another spatial method or switching the primary endpoint to Source would change the question. No such substitution or expansion was made.

## S0: four-frame coverage under q and under the real event

N uses the original physical-bin Uniform4 inside Native; U uses the same rule across the complete input; P greedily maximizes E_q[min(1,hits/2)] over four distinct available decoder frames. Projection Uniform4 is an auxiliary baseline. The exact gain is .5 times posterior mass of containing intervals with fewer than two previously selected hits. Deterministic physical-distance ties are fixed before scoring. No expert frames were acquired: the measured outcome here is event-time coverage, not object identity, DINO reference quality or actual adaptation benefit.

| Target | Real coverage N / U / P / projection4 | Zero-hit % N / U / P / projection4 | P−N [95% CI] | P−U [95% CI] |
|---|---:|---:|---:|---:|
| hc2 | 0.765625 / 0.671875 / 0.781250 / 0.843750 | 18.750 / 3.125 / 15.625 / 9.375 | +0.015625 [-0.046875, +0.078125] | +0.109375 [-0.046875, +0.250000] |
| vidstg | 0.765625 / 0.671875 / 0.687500 / 0.765625 | 12.500 / 21.875 / 28.125 / 12.500 | -0.078125 [-0.140625, -0.015625] | +0.015625 [-0.062500, +0.093750] |

Vid has0 improving,5 worsening and27 unchanged parents for P−N coverage. Its CI upper bound is below the +.05 practical threshold; zero-hit rises from4/32 to9/32. HC P−N is positive but only+.015625, and P zero-hit5/32 exceeds U1/32. HC fast-motion P−U=−.1875 and Vid small-target P−N=−.0625 also trigger locked coverage risk. All groups and missing within-event spread values remain visible. The unique fixed extension trigger is false, so no extension is run to seek significance.

Model coverage and real-event coverage diverge. Mean q coverage N→P rises .987126→.991274 in HC and .996332→.999156 in Vid, while real coverage changes+.015625/−.078125. This is not proof that all posterior allocation is useless. It shows this frozen model distribution and capped-two-hit allocation did not satisfy the locked benefit/risk criteria. The failure cannot be assigned to reference identity because no new reference was examined.

The base noninferiority-to-U condition is P−U≥0. Before any new scoring, ANALYSIS_GATE_LOCK (bound by EVALUATION_LOCK) clarified the separate posterior-added-value requirement P−U>0, consistent with stopping a core claim when plain U is equally good. Both variants produce the same stop here. The scorer and all original locks remain unchanged.

## Measurement, cost and verification

The original TA-STVG cross-domain sources are HC target←Vid source checkpoint5ab12c86… and Vid target←HC source47d8f158…. The panels are historically exposed and use different source checkpoints and original projection center fractions1.0/.5. The data do not identify an intrinsic domain-specific mechanism. There is one query per parent, so query and parent macro means coincide. Parent bootstrap uses10,000 fixed-seed draws; the8-parent subgroup floor is an engineering evaluability minimum, not a precision theorem. These intervals describe the saved outputs at fixed original online After trajectories; they do not replay the online stream after removing parents. One query per parent does not remove inherited-trajectory coupling.

All64 five-readout/four-layout outputs were sealed before new GT scoring. Readout wall time was 5.355689s, with T0 sum 3.383170s and S0 sum 0.422114s. The separate exhaustive numerical audit took 103.874555s; it is verification overhead, not fresh model inference or an end-to-end speedup claim. New GPU/model/DINO/training/Student-update counts are all0.

Independent verification covered315,403 actual legal decisions by direct physical tIoU, max error3.122503e−15; Source-logit reconstruction of q differs by at most6.245005e−17; greedy physical-membership/objective/tie error is at most3.287301e−16. Thirty adjacent/irregular/point-mass/uniform/sparse/dense synthetic cases add822 utility and43,374 endpoint-partition checks. Anonymous JSON reaggregation passes6,132 means/paired-CI/count/gate checks. The old Native/projection/per-offset-MBR trajectories reproduce6,336 Source/Before/After per-query scalars bit-for-bit. The official metric implementation was also checked against DenseTube for2,880 scalars by the postseal scorer.

## Decision scope and files

T0 and S0 both stop under the prelocked development tree. S1 actual experts/fits, crossed J0/Direct, online-stream checks, second backbone and C0 confirmation are untriggered stages, not incomplete measurements. No temperature, resolution, model, encoder or feedback variant is launched. CURRENT/MAIN/PAPER and prior positive/negative results remain unchanged. No universal impossibility claim follows.

- [All anonymous per-parent metrics, signed differences, CIs and strata](RESULTS_INITIAL.json)
- [Locked gate outcomes, risks and unknown groups](DECISION.json)
- [Actual configuration, timing and pre-GT clarification](ACTUAL_PROTOCOL.json)
- [Independent aggregate audit](ROOT_AGGREGATE_AUDIT.json) and [numerical verification](VERIFICATION.json)
- [Aggregate figure](DECISION.svg)
- [Protocol](../../../../protocols/stvg_posterior_decision_tree_v1.md)

Raw q/utilities/logits, GT coordinates, original query/parent identifiers, media, weights and tensors remain private. Public rows contain anonymous scalars and offline group bins only.
