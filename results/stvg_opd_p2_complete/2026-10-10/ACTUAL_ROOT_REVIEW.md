# Fixed OPD P2: complete mechanism evidence

Full substantially outperforms shuffled feedback in both source directions and both setting means. Full also exceeds Fixed Rollout on VidSTG cross-domain and on the five-condition same-domain mean in each direction. The Full–Direct confidence intervals include zero in all four setting means. These data support feedback correspondence and, in those settings, refreshed on-policy rollouts; they do not establish reliable superiority over direct L1/GIoU distillation.

## Cohort and protocol

TA-STVG source-only checkpoints, two target panels of 128 historical parent sources with one query per parent (256 parents total). Cross-domain clean uses both locked stream orders; same-domain uses frame drop, freeze, exposure, motion blur and occlusion at the original 5% physical burst severity. Every arm/condition/order resets to its own qualified source state. Query residual and Adam reset per query; LN persists only within its own stream. Native WHEN is fixed and the configured final round is read out. HC2 uses .01/.025/.05/40 rounds/LN1/16/M32; VidSTG uses .03/.1/.25/10 rounds/LN1/8/M32. There was no formal-result tuning or cohort selection.

All 7,168 logical predictions sealed globally before GT. 5,632 were new formal fits and 1,536 are exact reuse of complete original P0 streams with matched source, inputs, configuration and history. Qualified fits are never accepted as formal predictions. The 128-parent panels are historically exposed confirmation panels, not fresh unseen cohorts.

## Paired effects

Differences are parent-macro vIoU percentage points; brackets are 95% intervals from 10,000 paired parent bootstrap resamples. Same-domain means first average the five conditions within the same query/parent, rather than treating conditions as independent samples.

| Setting | Full − Frozen | Current (After − Before) | Inherited (Before − Frozen) |
|---|---:|---:|---:|
| P2_hc2_cross_clean | +1.346 [-0.046, +2.526] | -0.430 [-1.789, +0.669] | +1.776 [+1.417, +2.157] |
| P2_hc2_same_5percent | +3.931 [+2.497, +5.304] | +0.045 [-1.428, +1.405] | +3.885 [+3.263, +4.550] |
| P2_vidstg_cross_clean | +3.871 [+2.430, +5.290] | +2.037 [+0.753, +3.317] | +1.834 [+1.327, +2.367] |
| P2_vidstg_same_5percent | +2.175 [-0.114, +4.383] | +0.410 [-1.782, +2.472] | +1.765 [+0.988, +2.566] |

| Setting | Full − Direct | Full − Shuffled | Full − Fixed Rollout |
|---|---:|---:|---:|
| P2_hc2_cross_clean | +0.576 [-0.236, +1.433] | +3.837 [+2.257, +5.335] | +1.124 [-0.233, +2.244] |
| P2_hc2_same_5percent | +0.603 [-1.069, +2.267] | +6.320 [+4.595, +7.991] | +3.357 [+1.942, +4.691] |
| P2_vidstg_cross_clean | -0.213 [-0.625, +0.216] | +8.820 [+6.861, +10.979] | +3.860 [+2.415, +5.334] |
| P2_vidstg_same_5percent | +0.523 [-0.343, +1.494] | +10.222 [+7.455, +13.033] | +2.579 [+0.579, +4.487] |

The HC2 cross-domain Full gain is +1.346 pp with an interval crossing zero; the VidSTG cross-domain gain is +3.871 pp with a positive interval. HC2 same-domain gains are dominated by inherited state (+3.885 pp) while mean current adaptation is +0.045 pp and inconclusive. P3 is the predeclared query/LN/zero-writeback separation needed to resolve this inherited-state mechanism. These P2 decompositions alone do not attribute effects to a particular parameter block.

## Complete failure and signal evidence

All parent effects, severe negative tails, expert-quality strata, duration/motion strata and query types remain in the export. For Full, cross-domain parent losses exceeding 5/20 pp are 11/2 on HC2 and 6/1 on VidSTG. On the equal five-condition same-domain mean they are 10/3 and 15/6. Average gains do not imply reliable correction on every source.

Root actually viewed four complete plot pairs and six distinct private RGB cases: success, current harm and positive-expert-reward/current-task-harm in each direction. The HC2 successes can move to the correct person even with a wrong DINO reference; HC2 failures visibly move from the correct person to a distractor. VidSTG success tightens a broad tube to the child; failures show a face-sized pseudo box or a box spanning a distractor. Native WHEN does not move. Case selection is deterministic posthoc description after the complete population, not an efficacy estimate or online selector.

Across all Full condition/order cells with GT-evaluable admitted observations, 321/797 HC2 cells and 176/575 VidSTG cells have positive mean central expert-agreement change with negative current vIoU. These are descriptive repeated cells, not independent-parent rates or a causal intervention. The six saved signal chains include 16,320 independent sample/GT frame-IoU checks. Older P0 samples use explicitly labeled offline float64 sigmoid readback; newer traces use same-call saved original GPU actions. Offline best sample is never a deployment prediction.

## Actual cost and integrity

Full fit wall time per arrival is 1.356 s (HC cross), 1.441 s (HC same), .347 s (Vid cross) and 0.343 s (Vid same). Capture is shared and must not be counted once per arm. Reused expert/capture measurements are not cold-start latency; reused P0 fit costs are the original measurements, not new P2 GPU time. Fit wall time includes in-fit CPU precision audit overhead. Complete recorded component/capture/CPU/backward/memory metadata is retained in COST.json.

Root actually read all 7168 immutable complete fit dictionaries, 179200 rounds and 12845056 state coordinates; all saved mathematical dictionaries dispatched exactly by their original pinned revision. It independently checked 64512 dense metric scalars and 14336 observation/unobserved readouts. Maximum independent dense error is 1.89e-15. Public CPU recomputation compares 239,436 scalars on all 7,168 anonymous rows and all 256 parents.

Root-only helper failures (an incorrect all-zero source-LN assumption and immutable-receipt resume collision) are preserved separately. Their repairs read the actual qualified source tensors and verify existing immutable population receipts exactly. Neither modifies experiment predictions, source resets, source weights, scoring or scientific locks. All earlier numerical repair failures and original code/runtime remain preserved. Checks are independent in Gaussian/IoU/softmax/Adam/state/dense calculations and native-head VJP; they do not constitute an independently reimplemented complete decoder Jacobian or a proof of CUDA transcendental arithmetic.

Private RGB, raw query/caption, GT geometry, native boxes, sampled actions, weights, fit/gradient/Adam tensors and credentials are excluded. Public anonymous scalar rows independently reproduce aggregate arithmetic, not private model inference. All actual negative findings are published. P2 completion authorizes only the original fixed P3 continuation; P3–P6 still require their own real qualification, global prediction seal, root/case/view/public/archive closure. EATA and every historical paused queue remain paused.
