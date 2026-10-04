# Spatial-DeCoTA P0: original Direct versus proposal-distribution energy critic

This is the user-authorized episodic spatial qualification of the later C1–Scale06 interface. Both arms reset spatial LN/query/Adam to the same checkpoint source for each input, adapt only the original 1,792 spatial parameters, and emit the current tube with the unchanged native temporal interval. No temporal adaptation or LN inheritance is executed.

| Corruption panel | Frozen vIoU % | Direct − Frozen pp [95% CI] | Critic − Frozen pp [95% CI] | Critic − Direct pp [95% CI] |
|---|---:|---:|---:|---:|
| Vid search | 16.215 | 1.688 [-2.047, 6.482] | 2.145 [-0.282, 4.356] | 0.458 [-4.628, 4.539] |
| Vid confirm | 32.953 | 4.259 [0.892, 9.121] | 2.984 [-2.014, 8.675] | -1.275 [-3.666, 0.047] |
| HC2 search | 29.867 | 2.036 [-1.587, 5.049] | 5.333 [3.314, 7.346] | 3.297 [1.046, 6.137] |
| HC2 confirm | 26.478 | 3.678 [0.998, 6.986] | 3.975 [1.016, 7.352] | 0.297 [-0.976, 1.852] |

Paired source-macro differences average both orders and five corruptions within source; 10,000 paired source bootstrap draws, seed 20261004. Each dataset has 32 development and 16 source-disjoint confirmation sources; all are historically exposed. These are not fresh tests or full-dataset benchmarks. There are 576 unique inputs / 1,152 logical arrivals per arm. The episodic second-order readouts are identical and reused, not additional independent measurements.

## What changed and what stayed fixed

- Direct uses the unchanged original C1 Scale06 admission, accepted top-one DINO anchors, and sum(5 L1 + 2 GIoU)/planned4.
- Critic uses all nonempty geometrically valid cached supports, M≤3 on each observed frame, without the old score/margin admission. Its objective is the valid-frame mean of `−logsumexp(logsoftmax(existing_score/1) + IoU(box, proposal)/1)`. Both temperatures are prelocked to one; no sweep.
- The original context path retains phrase-score NMS .5 and truncate-three-then-valid filtering. The historical fallback cached only a single selected proposal; it remains singleton, without invented candidates.
- Both use original four native-interval observations, matching input hashes and checkpoints, fresh Adam .03 / (.9,.999) / eps1e-8 / wd0, ten updates, and the earliest minimum of their own objective over steps0..10.
- Empty evidence is a no-op. Best-step selection never uses GT. Ordinary IoU is retained even when disjoint boxes give zero gradient.
- This changes the supervision-construction block (support/admission/reduction/objective), so an effect cannot be uniquely attributed to energy or uncertainty preservation. It remains a DINO-coordinate proxy, not a demonstrated native-policy OPD mechanism or elimination of pseudo supervision.
- Old C1 online spatial-after gains +4.3812/+3.7314 pp included inherited LN. They are not the episodic Direct control measured here. Current-query benefit does not establish persistent benefit.

## Empty support, zero gradients and proxy/task mismatch

| Confirmation corruption, unique inputs | Arm | Empty inputs | Zero initial gradient | Step0 selected | Proxy improved but vIoU harmed | All observed support initially disjoint |
|---|---|---:|---:|---:|---:|---:|
| Vid (n=80) | direct | 25 | 25 | 25 | 10 | 0 |
| Vid (n=80) | critic | 25 | 25 | 25 | 9 | 0 |
| HC2 (n=80) | direct | 4 | 4 | 4 | 13 | 0 |
| HC2 (n=80) | critic | 0 | 0 | 0 | 18 | 0 |

A zero gradient is the evaluated proxy derivative, not proof of task correctness. No-op can protect good tubes while also preventing correction of wrong, disjoint tubes. Per-step post-hoc GT values diagnose execution, and never choose the returned step. UPDATE_DIAGNOSTICS.json retains every path and actual displacement; reduced motion alone does not prove a better objective.

## Clean and negative tails

| Confirmation | Arm | Clean vIoU change pp [95% CI] | Corruption >5 / >20 pp harms, logical | .3 correct→wrong / wrong→correct | .5 correct→wrong / wrong→correct |
|---|---|---:|---:|---:|---:|
| Vid | direct | 4.073 [0.247, 9.207] | 8 / 0 | 0 / 0 | 0 / 14 |
| Vid | critic | 2.263 [-4.519, 8.680] | 8 / 8 | 8 / 0 | 0 / 12 |
| HC2 | direct | 4.821 [2.047, 8.023] | 6 / 0 | 0 / 2 | 0 / 32 |
| HC2 | critic | 4.330 [1.566, 7.491] | 4 / 0 | 2 / 0 | 0 / 34 |

Every condition, order, dense sIoU, correctness transition, source value and gross gain/loss is public. The harm counts include the two identical order readouts; they are not independent samples. Temporal tIoU is exactly invariant for every arm/input.

## Largest positive and negative current-query cases

| Dataset | Arm | Panel / anonymous source / corruption | Selected step | ΔvIoU pp | Proxy loss change |
|---|---|---|---:|---:|---:|
| vidstg | direct | confirm / 35 / motion_blur_5 | 10 | -11.099 | -0.169114 |
| vidstg | direct | confirm / 37 / motion_blur_5 | 10 | 34.922 | -1.686848 |
| vidstg | critic | confirm / 35 / exposure_5 | 10 | -37.478 | -0.104754 |
| vidstg | critic | confirm / 37 / frame_freeze_5 | 9 | 34.423 | -0.241238 |
| hc2 | direct | confirm / 47 / motion_blur_5 | 9 | -16.485 | -2.395037 |
| hc2 | direct | confirm / 33 / frame_freeze_5 | 8 | 25.826 | -1.597084 |
| hc2 | critic | confirm / 47 / occlusion_5 | 10 | -13.436 | -0.129959 |
| hc2 | critic | confirm / 33 / frame_freeze_5 | 10 | 23.470 | -0.180877 |

These examples illustrate tails; they do not establish a general causal mechanism or override paired source results.

## Actual cost and audits

| Dataset | Unique paired inputs | New DINO calls | Formal full-backbone forwards | Spatial backwards | Worker wall min |
|---|---:|---:|---:|---:|---:|
| vidstg | 288 | 0 | 0 | 4370 | 3.32 |
| hc2 | 288 | 0 | 0 | 5700 | 3.85 |

Two extra full-backbone clean smoke inputs check native bitwise parity, all-step cached/full Direct and Critic trajectories, and selected-state reinsertion with unchanged native temporal logits. They are separate from formal cache-only replay. All formal payloads are sealed globally before GT scoring. Independent NumPy objectives and Adam arithmetic, every source reset, best-state selection, official/vectorized dense scoring and public summary regeneration pass. Worker wall time includes replay, host IO and logging; it is not pure GPU kernel time.

## Root interpretation and counted resources

The fixed critic retains a material positive point estimate in both confirmation panels, with positive paired evidence on HC2. It does **not** establish a two-dataset pass: Vid's interval crosses zero and its four unique corruption >20pp harms all involve anonymous source35. For exposure5, Frozen vIoU45.616% falls to8.137% with Critic while Direct returns35.158%; energy improves from−.283775 to−.388529. The current-state proxy therefore does not guarantee correct current-query actuation. This is not evidence for a universal failure of critic objectives.

Both Vid confirmation arms have9/16 sources with positive corruption-average gain; the largest source accounts for46.89% of Direct positive gain and48.57% of Critic positive gain. HC2 has13/16 and11/16 respectively, with largest positive shares35.60% and30.96%. Mean gains alone must not hide this concentration.

No confirmation input with nonempty critic evidence has an entirely disjoint support and zero initial gradient. In Vid development, four such corruption inputs do occur. Thus ordinary IoU's flat disjoint region is a measured limitation in development, but it does not explain the confirmation severe harms. Vid confirmation's25/80 zero-gradient inputs are empty evidence in both arms. The energy-improved / GT-harmed cases have nonzero updates.

Gradient scales differ without an equivalent displacement reduction: on nonempty Vid confirmation inputs, mean initial gradient norm is1.199 for Direct versus.148 for Critic, but selected parameter displacement is4.209 versus4.294. HC2 displacement is5.453 versus4.774. Adam and own-objective state selection prevent an interpretation that a small energy gradient simply means a small update. The comparison also changes retained support and admission, so it does not isolate the energy formula alone.

The two smoke inputs execute12 offset full-model forwards: per input,2 frozen plus2 Direct-state and2 Critic-state reinsertion. Smoke adds80 backward calls; formal fits execute10,070. The COST field `spatial_replay_evaluations` counts objective fit paths and excludes constructor/parity replay. No new DINO inference occurs. These are operation counts and worker wall measurements, not GPU kernel timings.

The CPU scorer's unmatched-parenthesis error was detected by AST parsing before GT. Its original source was preserved, only that syntax was repaired, and revision001 was pinned; prediction settings and GPU payloads were not changed or replayed. All subsequent audits pass.

## Scoped decision

- vidstg: `inconclusive` for this fixed critic P0.
- hc2: `positive_current_panel` for this fixed critic P0.

No production registry is promoted. No LN-persistence P1, cross-domain, new teacher, gate, preservation term or loss sweep is launched by this P0. A later experiment needs a concrete mechanism supported by these measured outcomes.

Artifacts: [protocol](../protocols/tastvg_decota_critic_p0_v1.md), [complete anonymous results](../results/tastvg_decota_critic_p0/2026-10-04/), [root audit](../results/tastvg_decota_critic_p0/2026-10-04/ROOT_AUDIT.json), [public audit](../results/tastvg_decota_critic_p0/2026-10-04/PUBLIC_AUDIT.json).

After this P0 readback, the user explicitly authorized the matched LN1/16 inheritance experiment. It is a new continuation; P0 itself did not automatically trigger or complete P1.
