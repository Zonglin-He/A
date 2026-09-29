# O1.1: Residual Scale Readout Test

Amplification removes selection inertia, but none of alpha8/16/32 improves mean vIoU on the fixed24-arrival trajectory. Exact Full-Rerank agreement increases modestly and non-monotonically. Scale explains why the original readout barely moved; it is not a sufficient explanation for the missing task gain.

Only CPU readback was performed. Same24 non-expert arrivals, fixed saved O1 arrival weights,768D features, native scores and candidate order. S(alpha)=native_score+alpha*(phi@arrival_w), alpha in {1,8,16,32}. No learning-rate change, retraining, expert calls, backbone forward or online trajectory rerun.

| Alpha | Changed vs Frozen | Exact Full-Rerank agreement | tIoU (%) | vIoU (%) | Delta tIoU vs alpha1 (pp) | Delta vIoU vs alpha1 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1/24 | 1/24 (4.17%) | 46.2867 | 17.9538 | +0.0000 | +0.0000 |
| 8 | 16/24 | 5/24 (20.83%) | 46.4284 | 17.5656 | +0.1417 | -0.3882 |
| 16 | 17/24 | 4/24 (16.67%) | 46.6493 | 17.6065 | +0.3626 | -0.3473 |
| 32 | 19/24 | 3/24 (12.50%) | 46.0061 | 17.3296 | -0.2806 | -0.6242 |

Frozen: tIoU 46.2867%, vIoU 17.9538%. Full Rerank reference: tIoU 50.0614%, vIoU 18.6286%. Full Rerank uses current-arrival expert information and is not a same-budget method or oracle.

## Paired changes and scope

| Alpha | Delta tIoU 95% interval (pp) | Delta vIoU 95% interval (pp) | vIoU gains / losses / unchanged | vIoU harms >5pp |
|---|---:|---:|---:|---:|
| 1 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0 / 0 / 24 | 0 |
| 8 | [-1.2622, +1.4588] | [-1.1456, +0.1919] | 5 / 8 / 11 | 1 |
| 16 | [-1.0738, +1.7514] | [-1.0844, +0.2189] | 6 / 8 / 10 | 1 |
| 32 | [-2.2206, +1.4352] | [-1.5892, +0.1381] | 6 / 9 / 9 | 2 |

All amplified tIoU/vIoU intervals include zero. They are descriptive paired bootstraps of these24 sealed outcomes (10000 samples, seed20260929), conditional on one already-exposed stream and shared fixed states. They do not estimate robustness over online trajectories or establish that amplification is generally harmful. No scale is promoted or selected for deployment.

Alpha8/16/32 move many predictions, but top1 agreement is only5/4/3 out of24. Relative to alpha1, newly matching Full choices number4/3/3; alpha32 also loses one previous match. Most moved choices therefore do not recover Full Rerank. Small tIoU gains at8/16 do not translate into tube gains; simply giving this learned residual more weight is insufficient in this screen.

## Preserve both useful and harmful changes

| Arrival / anonymous parent | Condition | Alpha1 t/v (%) | Alpha8 t/v (%) | Alpha16 t/v (%) | Alpha32 t/v (%) |
|---|---|---:|---:|---:|---:|
| 15 / Q15 | frame_drop_5 | 66.0194 / 45.9092 | 56.1983 / 39.3507 | 56.1983 / 39.3507 | 56.1983 / 39.3507 |
| 16 / Q07 | frame_freeze_5 | 14.8810 / 9.3297 | 14.3678 / 9.0080 | 14.3678 / 9.0080 | 0.4902 / 1.9980 |
| 18 / Q12 | occlusion_5 | 87.3016 / 30.9563 | 92.5926 / 32.5931 | 92.5926 / 32.5931 | 92.5926 / 32.5931 |

Arrival18/Q12 moves toward Full Rerank and improves both metrics. Arrival15/Q15 also matches Full Rerank after amplification but loses task accuracy: critic agreement is not ground-truth correctness. Arrival16/Q07 becomes substantially worse at alpha32 without matching the Full choice. These are descriptive readbacks; all24 cases remain in ROWS.json.

## Verification and completion

All96 selections were sealed before consuming teacher-score values. Cached teacher scores were then used solely for agreement and sealed before consuming cached GT-derived metrics. Input hashing validates bytes but does not use those values to choose scales or candidates. Alpha1 exactly reproduces the O1 scores/selections. Independent math.fsum dot products and NumPy selection agree for all24/96; original input files and production registration retain their hashes. No raw GT file was reread: task values are gathered from O1's already dual-implementation-verified per-candidate metrics.

Independent public audit reconstructs 96 selections and 30 means/intervals plus all agreement and harm counts. Maximum independent residual-dot error is 5.55e-17.

This readout multiplier is NOT equivalent to multiplying the online learning rate: a different learning rate would change subsequent gradients and states. Results rule out a simple claim that these fixed learned directions only need more voting weight to yield gains at the tested scales. They do not establish that all linear representations, score normalization, slow-state mechanisms or Slow-Fast TTA are impossible.

Only O1.1 was executed. No normalization rerun, LR grid, prototype, memory, reliability gate, spatial experiment or new deployment setting was added. The next mechanism remains undecided; production stays unchanged.

New GPU/model/expert/update work:0. Measured selection/agreement/metric-readback CPU stages total 0.090702s, excluding Python imports, development, independent audit, reporting and publication. Raw weights/features/media stay local; anonymous scores, all four readouts and task values are exported.

Reproduce: scripts/run_tastvg_o11_scale_readout_v1.py prepare -> select -> agreement -> score; scripts/report_tastvg_o11_scale_readout_v1.py. Public scalar readback: scripts/audit_tastvg_o11_public_v1.py <result-directory>.
