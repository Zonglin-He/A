# HC independent anchor content states: E / Direct / Fixed / Live

Completed development experiment, all16 NH parents/one query each. Frozen original Vid-source EMA TA-STVG, original E+finalMBR/four planned physical nodes and cached crop/full-image DINO evidence. No new experts, DINO/Source encoding, learned head, pretrained weight/LN updates or new data. This is not independent confirmation.

## Intervention and supervision

K independent256D temporary vectors (K<=4) are added only to their physical-frame content tokens before spatial block0 temporal self-attention. All six frozen blocks decode the complete original actual trajectory. No vector is broadcast over frames; two offset decoders have no cross-offset attention. Zero state exactly restores Source and E through the fixed center-difference/log-size E map. NH013 has anchors only in offset1; its unsupported offset0 stays exactly original, not claimed supervised.

Teacher: admitted cached crop target-score top1, otherwise admitted original target-score top1, otherwise missing; no E joint residual selection or GT-dependent support. All16/61 references remain, including harmful/unknown/time-wrong cases. Teacher target scores are not calibrated correctness probabilities.

Common localization cost is5L1(cxcywh)+2(1−GIoU), mean1/K. Direct minimizes this cost. Fixed/Live form detached softmax(−cost/.05) targets on32 antithetic native-logit Gaussian samples, then the original likelihood/control-variate gradient passes through the current actual decoder mean to the temporary states. It is an output-coordinate stochastic policy, NOT sampled content states/realizable stochastic decoder trajectories. Every updated deterministic full trajectory is genuinely decoded. This state/teacher/cost combination is a route comparison, not a single-factor causal comparison against old fixed-IoU1792 results.

Fixed holds the initial samples/teacher feedback while fitting its current likelihood; Live resamples around each updated mean and re-evaluates the same frozen localization cost. Teacher RGB/boxes do not refresh, so no new expert calls are needed. Both arms have identical first samples, targets, gradient and update. Original HC numerical configuration is locked:40 steps,lr.01,sigma.025,tau.05,M32,Adam(.9,.999),eps1e-8; no scan, best-step selection, rollback or writeback. Direct/Fixed/Live share the same state initialization and40-update allowance.

## Main paired differences, pp [parent bootstrap95%]

| Contrast | complete vIoU | unobserved actual sIoU | observed actual sIoU |
|---|---:|---:|---:|
| Direct_minus_E_MBR | -0.1784 [-0.6436,+0.1604] | +0.0340 [-0.3695,+0.6126] | -1.4870 [-5.4162,+0.6675] |
| Fixed_minus_E_MBR | +0.0227 [-0.0398,+0.0701] | +0.0645 [-0.0903,+0.2351] | +0.2347 [+0.0403,+0.5254] |
| Live_minus_E_MBR | -0.1423 [-0.6115,+0.2113] | +0.1038 [-0.3493,+0.7564] | -1.4921 [-5.3235,+0.5846] |
| Live_minus_Fixed | -0.1650 [-0.6190,+0.1701] | +0.0393 [-0.5039,+0.7062] | -1.7268 [-5.4757,+0.2420] |
| Live_minus_Direct | +0.0361 [-0.0017,+0.0879] | +0.0698 [-0.0074,+0.1751] | -0.0051 [-0.1008,+0.1027] |

All16 parents are the complete-v and unobserved-actual main denominator; observed-actual GT is available for15 parents, and missing GT stays unknown. All64 outputs/full update traces sealed before new HC GT access. Temporal intervals and tIoU are unchanged; unobserved-actual supervision is not inflated by interpolated frames.

## Absolute means and anchor fitting

| Arm | full vIoU % | actual-unobserved sIoU % | actual gradient updates | initial→final teacher cost, supported-query mean |
|---|---:|---:|---:|---:|
| E_MBR | 35.305477 | 56.660359 | 0 | — |
| Direct | 35.127097 | 56.694328 | 640 | 1.309963→1.107210 |
| Fixed | 35.328199 | 56.724877 | 640 | 1.309963→1.271289 |
| Live | 35.163204 | 56.764163 | 640 | 1.309963→1.104631 |

A lower teacher cost confirms optimization, not task correctness or cross-frame generalization. Direct−E measures interface value; Live−Fixed measures this refresh configuration; Live−Direct measures value beyond direct supervision. One HC direction can establish a result without Vid positivity, but these repeatedly exposed16 parents cannot serve as confirmation.

## Every signed result versus E, pp

| Alias | refs | Direct v | Fixed v | Live v | Direct unobserved s | Fixed unobserved s | Live unobserved s |
|---|---:|---:|---:|---:|---:|---:|---:|
| NH001 | 4 | -0.320008 | -0.361502 | -0.288231 | -0.682077 | -0.744529 | -0.610389 |
| NH002 | 4 | +0.000028 | -0.010571 | -0.005920 | -0.018497 | -0.023357 | -0.021422 |
| NH003 | 4 | -0.088194 | +0.150850 | +0.243619 | -0.096827 | +0.197064 | +0.454502 |
| NH004 | 4 | +1.061013 | +0.091056 | +1.227616 | +3.610528 | +0.227719 | +4.123837 |
| NH005 | 4 | +0.050694 | +0.040943 | +0.038654 | +0.198438 | +0.165295 | +0.152157 |
| NH006 | 4 | +0.026484 | +0.004335 | +0.033636 | -0.003315 | +0.003171 | +0.003073 |
| NH007 | 4 | -0.001999 | +0.073980 | +0.028333 | -0.072933 | +0.071397 | -0.009060 |
| NH008 | 4 | +0.211541 | +0.051317 | +0.175476 | -0.459659 | -0.091342 | -0.380771 |
| NH009 | 4 | -0.857034 | +0.086751 | -0.929692 | -1.407406 | +1.043748 | -1.571558 |
| NH010 | 3 | -0.041345 | -0.017334 | -0.033540 | -0.060119 | -0.045459 | -0.059094 |
| NH011 | 4 | +0.141127 | +0.114764 | +0.165397 | +0.150866 | +0.155027 | +0.126043 |
| NH012 | 3 | +0.016056 | -0.002084 | +0.008442 | +0.022022 | -0.001879 | +0.011306 |
| NH013 | 3 | -3.101980 | -0.016024 | -3.024235 | -0.699376 | -0.066767 | -0.657096 |
| NH014 | 4 | -0.104364 | -0.041393 | -0.096882 | -0.027583 | +0.038302 | -0.017812 |
| NH015 | 4 | +0.120311 | +0.122180 | +0.114971 | +0.078721 | +0.064061 | +0.076254 |
| NH016 | 4 | +0.033595 | +0.076293 | +0.065989 | +0.010716 | +0.039837 | +0.040885 |

All per-parent signs, leave-one sensitivity, severe tails, state movements, update counts, observation/offset maps and interpolation contributions are retained in RESULTS_PUBLIC.json. No favorable-parent filtering or new-source selection occurred.

## Engineering, audits and costs

Incremental GPU lease including loading, engineering probe, checks and3s conservative guard: 98.491s. Actual deterministic trajectory decoder calls 1985, offset PosDecoder calls 3970, state backwards 1920. New Source/DINO/expert calls0; new network/head training0. Old RGB/feature/decoder-input/DINO acquisition costs are excluded; this is not first-deployment latency.

Owned unit hard cap880s+5s stop inside900s GPU cap,MemoryHigh16G/Max20G/Swap1G,CPU400%,shared nonblocking gpu.lock and resource floors. Pre-execution001 repaired progress-file mutable handling and bound CPU-imported helper dependencies before GPU start; original code preserved, scientific settings unchanged. No scientific output was altered or rerun for scoring. After the64-output barrier, the first scoring entry read GT then failed on the existing three-item truths return contract; only unpacking was corrected. A retry and independent scorer were stopped by amendment filename ordering before GT, then ordering was corrected without weakening any hash check. All failed entries and original scorer are retained; failed CPU scoring/readback work is not fully timed. See ENGINEERING_RECOVERY_PUBLIC.json.

Mechanism audit independently checks physical offset mapping, zero states, paired E output geometry, candidate coordinates/localization costs/weights, Fixed initial target reuse/Live current-mean provenance, analytic gradients and CPU Adam replay. CPU/GPU elementary float32 operations use documented numerical bounds; deterministic zero-state comparisons remain exact. The audit is not another GPU/model replication. Postseal independent official/dense and actual-node metric/parent-statistic readback is reported separately.

## Decision scope

Report every contrast rather than turning a local fit or direct improvement into OPD evidence. The2–3pp objective is complete-cohort full-v. If only Direct improves, credit the state/supervision interface. If Live improves over E but not Direct/Fixed, OPD-specific gain remains unestablished. A finite negative result concerns this layer0/K×256/cost/budget recipe, not every frame-conditioned state. No automatic layer/step/temperature/scale scan, new expert, new cohort or confirmation is started. CURRENT/MAIN/PAPER and historical paused queues remain unchanged.


## Measured decision for this finite recipe

Direct−E full-v−.1784pp; Fixed−E+.0227; Live−E−.1423, Live−Fixed−.1650 and Live−Direct+.0361. All these full-v paired intervals cross0. Live unobserved-actual-s+.1038pp also crosses0. No2–3pp complete-cohort gain or independent OPD advantage is established; no new HC confirmation is launched. This is an actual OPD result, not a direct-readout-only gate.

Each adapted arm performs640 updates. Mean teacher cost1.309963 falls to1.107210/1.271289/1.104631 for Direct/Fixed/Live; state norm and1007 changed nodes per arm prove real adaptation, not correctness. All48 final trajectory changes are nonconstant in center/log-size space, so the previous common4D shift constraint was genuinely removed. The33 unsupported NH013 offset0 nodes remain exactly E in every arm/step. More accurate fitting than Fixed does not establish better localization. No fresh correct-reference positive control was run for this new interface, so teacher quality versus attainable/optimized transfer is not fully separated.

Preserve successes: NH004 Direct/Live full-v+1.0610/+1.2276pp, with actual-unobserved-s+3.6105/+4.1238. NH011 Live full-v+.1654pp. Preserve harm: NH013 Direct/Live full-v−3.1020/−3.0242pp and observed-actual-s−27.7090/−27.1797; neither is deleted or used for output selection. Other signs and mean cancellations remain in full results. Full-v has no<-5pp event, but the observed-node severe tail is material.

The uniform residual was a demonstrated information bottleneck; removing it with these layer0 independent content states and the fixed budget was not sufficient to obtain the sought task gain. That is not evidence that all frame-conditioned states, deeper interventions, other supervision or OPD are impossible. The exact tested configuration is archived rather than promoted or automatically tuned.
