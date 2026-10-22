# HC64: joint event completeness, matched null, and paired boundary correction

**Three actual fixed studies completed. No candidate is admitted to the main method.** The conservative matched-null variant produces4 complete-vIoU improvements and2 harms, with0 losses>5pp, but changes only7 intervals and gives+.0129pp tIoU. Directly applying the same feedback to more local boundary pairs increases gains to17 and harms to16 and lowers tIoU. The requested combination of broad improvement and a low harm proportion is not established. Full remains paused at HC1902 scored/Vid1373 saved.

## What changed in the feedback

Previous independent hidden-change peaks can cut inside an event. This study evaluates both boundaries as a single interval explaining the predicate evidence, including its interior and what is excluded. It uses the exact saved original8 ReferDINO object-token logits and a fixed selected whole track; no framewise identity switches, new teacher or actionness head.

A query-only dependency rule extracts main-clause lexical predicates and their clausal/conjoined verbs, excluding noun-attached descriptions when a main verb exists. The target nominal reference follows wh-determiner/wh-person, nominal-subject, then first-noun fallback. Exact old event token mapping reproduces64/64. No raw text is published. Predicate and nominal matching logits are normalized over the same15 tracked queries; their target-track ownership difference is the proposed identity-relative event signal, not an originally calibrated event probability.

For each query and offset, a disposable ridge1 linear projection maps normalized concatenated six-layer temporal hidden to these pseudo-semantic margins at8 observations. Full8 and8 leave-one projections are fitted. Label centering/scaling in each fold excludes its heldout label; raw video features remain available. Sigmoid intercepts match the native posterior expected occupancy. No GT duration enters this calibration. The original eight teacher logits are temporally fused, so the folds test regression transfer, not independent withheld-teacher inference.

For every candidate interval I, integrate each predicate soft mask p over physical frame cells and compute

\[J_k(I)=\frac{\int_I p_k(t)\,dt}{|I|+\int p_k(t)\,dt-\int_I p_k(t)\,dt},\qquad R_1(I)=\frac1K\sum_k\log J_k(I).\]

This combines semantic coverage and background spill across both boundaries, while permitting different predicates to occur at different times. No unverified textual event ordering is imposed. J is a ratio of expected intersection/union under an operational soft mask, not expected true tIoU.

The guard requires positive aggregate leave-one prediction gain at both offsets, positive interval-score gain under all18 full/leave-one views and no loss in any predicate under the combined full views. Otherwise it returns the exact original interval. This is a heuristic against unsupported changes, not a guarantee against shared teacher errors.

The proposal-level semantic comparison is inspired by [CPL, CVPR2022](https://minghangz.github.io/uploads/CPL/CPL_paper.pdf). Its [official code](https://github.com/minghangz/cpl/blob/main/models/cpl.py) uses trained proposal-conditioned transformers and masked-word reconstruction; those trained capabilities are not reproduced by our per-query regression. [Point-supervised completeness](https://arxiv.org/abs/2108.05029) additionally uses labeled action points and pseudo-background training. Our expert observation positions are not action-boundary labels, so that result cannot validate this feedback.

## Mechanism-led revisions, with all original failures retained

V1: joint soft-mask completeness, native anchor-class mass preservation, original physical MBR. Controls remove the guard, use independent semantic endpoint rises/falls with the same guard, or use the old raw event response with the same projection/scoring/guard. It improves average tIoU+.4586pp but all22 accepted intervals expand. Candidate score/log-length correlation is.9353. Even weak nearly uniform masks can reward expansion; semantic agreement does not remove this structural bias.

V2: subtract a time-uniform mask with identical total predicate evidence before posterior reweighting:

\[R_2(I)=\frac1K\sum_k\left[\log J_k(I;p_k)-\log J_k(I;p_k\equiv\bar p_k)\right].\]

A constant mask now gives zero for every interval; the complete-video interval gives zero for every mask. Both are unit-checked. No strength/temperature scan occurs. All projected masks are reused; no new fit or text pass. Shifted controls move the logit profile by one-quarter timeline and restore its physical occupancy, rather than changing the evidence mass.

V3: same V2 feedback and guard, but test whether the posterior MBR suppresses useful local corrections. Enumerate at most9 valid native-grid pairs, moving each original boundary by at most one sampled node while preserving native eight-anchor membership. The guarded arm chooses maximum worst-view gain among feasible local pairs; the unguarded arm chooses combined score. Native tie preference is fixed. This is direct pair correction, not another posterior or query-learning claim.

Every study is locked before its predictions and independently sealed before scoring. V2 and V3 are explicitly adaptive development revisions of already-exposed HC64, not fresh confirmation or independent sources. No GT selects a branch, radius, state or query-specific configuration.

## Complete comparison

| Stage / arm | vIoU % | tIoU % | Δv G6 pp | Δt G6 pp | v improved / harmed | t improved / harmed | severe v / t harms >5pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 JointGuard | 29.8856 | 56.4199 | +0.1904 | +0.4586 | 9 / 9 | 12 / 9 | 1 / 1 |
| V1 Joint | 30.0966 | 56.6838 | +0.4014 | +0.7225 | 15 / 15 | 18 / 15 | 1 / 2 |
| V1 IndependentGuard | 29.7319 | 55.9072 | +0.0367 | -0.0542 | 3 / 4 | 3 / 4 | 0 / 0 |
| V1 RawJointGuard | 29.7934 | 56.2226 | +0.0982 | +0.2613 | 7 / 5 | 9 / 5 | 0 / 1 |
| V2 ExcessGuard | 29.7948 | 55.9742 | +0.0996 | +0.0129 | 4 / 2 | 4 / 3 | 0 / 0 |
| V2 Excess | 29.8040 | 56.0200 | +0.1088 | +0.0587 | 7 / 4 | 7 / 5 | 0 / 0 |
| V2 ShiftExcessGuard | 29.7210 | 55.9727 | +0.0259 | +0.0114 | 3 / 4 | 5 / 4 | 0 / 2 |
| V2 RawExcessGuard | 29.7559 | 55.9336 | +0.0607 | -0.0277 | 3 / 2 | 3 / 3 | 0 / 0 |
| V3 LocalGuard | 29.7938 | 55.6687 | +0.0986 | -0.2927 | 17 / 16 | 16 / 16 | 1 / 6 |
| V3 Local | 29.7929 | 55.7040 | +0.0977 | -0.2574 | 28 / 27 | 27 / 26 | 1 / 9 |
| V3 ShiftLocalGuard | 29.7254 | 55.7426 | +0.0302 | -0.2188 | 14 / 18 | 13 / 15 | 0 / 3 |
| V3 RawLocalGuard | 29.2913 | 55.2540 | -0.4038 | -0.7073 | 13 / 19 | 10 / 22 | 1 / 4 |

NativeG6 is 29.6952% vIoU/55.9613% tIoU. Original Source is 26.8938%/54.8230%. All NativeG6 interval/box outputs are bitwise identical across studies and to the prior sealed baseline.

## Benefit versus risk, not mean gain alone

| Preregistered stage primary | Harmed /64, complete v | Harmed / changed intervals | Harmed / changed v scores | Δv95% CI pp | Δt95% CI pp |
|---|---:|---:|---:|---|---|
| V1 JointGuard | 9/64 (14.06%) | 9/22 (40.91%) | 50.00% | [-0.2317, +0.6319] | [-0.1512, +1.0797] |
| V2 ExcessGuard | 2/64 (3.12%) | 2/7 (28.57%) | 33.33% | [+0.0021, +0.2223] | [-0.1854, +0.2062] |
| V3 LocalGuard | 16/64 (25.00%) | 16/37 (43.24%) | 48.48% | [-0.4063, +0.6003] | [-1.1048, +0.5255] |

CIs use10,000 fixed-seed paired parent draws; one query per parent. They are descriptive after iterative development and do not account for method selection/exposure. The narrow positive complete-v interval for V2 is not proof of generalization, and its temporal interval spans zero.

V2 has58 unchanged complete-v scores, so its3.125% all-query harm rate must be read alongside28.57% harm among its7 changed intervals (33.33% among6 changed-v scores), and only4 improved queries. V3 increases coverage but harms16/37 changed intervals (43.24%;48.48% of changed-v scores), and six tIoU losses exceed5pp. The guard is not a correctness test.

| Stage primary | Short Δt G6 pp (13) | Long Δt G6 pp (51) | Fixed-native-space Δv pp |
|---|---:|---:|---:|
| V1 JointGuard | -0.7368 | +0.7633 | +0.2223 |
| V2 ExcessGuard | +0.0000 | +0.0162 | +0.0858 |
| V3 LocalGuard | -0.3881 | -0.2683 | +0.0919 |

GT-duration groups are offline descriptions only. No duration-conditioned mixture is evaluated or deployed.

## Same-case outcomes

| Query | V1 Δt pp | V2 Δt pp | V3 Δt pp |
|---|---:|---:|---:|
| RC009 | -10.5237 | +0.0000 | -10.5237 |
| RC016 | +6.2500 | +0.0000 | +10.2289 |
| RC025 | +0.0000 | +0.0000 | +0.0000 |
| RC029 | +0.0000 | +0.0000 | +0.0000 |
| RC042 | -0.9751 | +0.0000 | +5.0399 |

V1 repairs RC016 by expansion but harms RC009 by the same tendency. V2 correctly abstains on these ambiguous cases but supplies no correction. V3 repairs RC016 and RC042 with paired moves, yet again damages RC009. RC025/RC029 lack positive projection transfer and remain native in guarded versions. Local search does not fix misaligned semantic supervision.

V2 real versus shifted evidence has only+.0015pp tIoU, with a broad CI crossing zero. V3 real versus shifted tIoU is−.0739pp, also inconclusive. These controls do not establish that the learnt pseudo-semantic signal reliably identifies the described event rather than consistent but irrelevant temporal structure.

## Cost and verification

No new Source encoder, decoder, visual head, ReferDINO call, RGB decoding, backbone training or query-gradient update. V1 parses64 texts with the existing CPU Stanza parser and fits disposable tiny linear regressions against already saved expert-token responses. V2/V3 reuse those exact projected masks. All stages preserve48 maintained runtime and434 provider source hashes; current full remains paused.

Mean cached research-panel feedback costs: V1 91.74ms/query (CPU parse and both relative/raw projections), V2 27.41ms, V3 20.28ms. These include experimental controls, exclude cache creation and initial text-parser loading, and are not measured end-to-end main-method deployment costs.

All192 new query records across three seals precede their corresponding new scoring. They contain64 unique queries/parents, not192 independent samples.2304 official/DenseTube metric checks agree to6.66e-16;18 independent spatial replay checks are bitwise exact. All V3 outputs obey the one-node paired bound and unchanged anchor membership. Scalar MAE values, means and CIs never changed; a post-seal summary corrects only the original scorer’s improved/harmed count orientation for this lower-is-better metric, preserving original reports/receipts privately.

**Disposition:** retain V2 as a conservative research result only; do not substitute it for the main temporal module. V1 and V3 remain measured negative tradeoffs. Broader gains with low harm are not obtained, and changing reward strength or adding query steps has no support from this experiment. The remaining requirement is a feedback signal calibrated to event semantics and background, not agreement of replicas of the same object-grounding proxy. Full remains paused; no new full or default promotion.

[Summary and cases](summary.json) · [V1 full results](../joint_event_hc64_v1/report.json) · [V2 full results](../joint_event_hc64_v2/report.json) · [V3 full results](report.json) · [V1 signed](../joint_event_hc64_v1/signed.tsv) · [V2 signed](../joint_event_hc64_v2/signed.tsv) · [V3 signed](signed.tsv) · [Verification](verification.json) · [Semantic/resource audit](semantic_audit.json)
