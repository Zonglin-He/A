# Fixed OPD P6: existing temporal signals, temporal head and offline supervised failure diagnostic

On the original fixed 32 historical parent sources per target, actionness projection improves vIoU relative to Native in both directions. The existing temporal-head update has a positive vIoU interval on HC2 but its VidSTG interval contains zero. On VidSTG the head is below actionness projection by 0.689 pp [−1.610, −0.020], conditional on this small historical cohort. The offline GT-supervised head gives substantially larger VidSTG gains, so weak deployment evidence and head capacity must be distinguished. No main-method change or result-driven parameter selection follows from these diagnostics.

Original attachment 029fff2e section 9 and LATER_DESIGN_LOCK fixed 32 hash-selected historical parents per target before the current P0 GT scoring. Each contributes one clean query, its complete original physical frame grid and two original offsets. Four deployment arms yield 256 outputs for 64 queries. All 256 deployment outputs globally sealed before any P6 GT read or offline GT-head fitting. Qualification outputs are not accepted as formal.

| Deployment arm | Actual implementation and state history |
|---|---|
| Native | Original frozen boxes and native interval. |
| ActionnessProjection | Existing original median/MAD physical-cell actionness projection with the original native prior; frozen boxes and no optimizer. |
| TemporalHead | Existing 66,306-parameter two-layer temporal head; per-query source-head reset and fresh AdamW, five steps, last output with eta .25. Frozen prefix and boxes; genuine GPU native reinsertion. |
| SpatialOPD | Exact complete original P0 on-policy order1 fit for the matched query and input. Its full 128-query stream predecessors and LN history remain intact. |
| Offline_GT_Head | Separately supervised CPU head fit after all deployment seals, same five steps and eta .25, first legal pair maximizing physical tIoU on each offset. Not available at deployment or a fair deployable baseline. |

The temporal head retains original NLL plus margin .2, AdamW betas (.9,.999), epsilon 1e-4, no weight decay, prior .1 and original decoding. VidSTG source → HC2 uses the existing vid_to_hc1 representative lr .001 and center fraction1 without HC2 tuning; HC-STVG2 source → VidSTG uses original lr .1 and center fraction .5. Source checkpoints are VidSTG SHA256 5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83 and HC-STVG2 SHA256 47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036. SpatialOPD retains HC .01/.025/.05/40/LN1/16/M32 and Vid .03/.1/.25/10/LN1/8/M32, original Uniform4/Top1 frozen DINO and 1792 joint parameters.

SpatialOPD is a descriptive matched-input component reference with its original longer state history; it is not a freshly reset 32-query online stream or a history-matched temporal-versus-spatial causal contrast. Main SpatialOPD Native WHEN remains unchanged. Existing temporal alternatives are evaluated separately under the original finite P6 diagnostic.

Query macro and parent macro are equal because each original parent has one query. Intervals use 10,000 paired original-parent bootstrap draws, seed20261006. Intervals are conditional on the historical roster, source checkpoints and stated state histories, not fresh-source generalization guarantees. All ten pairwise contrasts per target and all three metrics are retained without multiplicity adjustment.

| Target | Arm | vIoU | tIoU | sIoU | vIoU−Native, pp [95% CI] | >5/>20pp harm parents |
|---|---|---:|---:|---:|---|---:|
| hc2 | Native | 22.663 | 43.575 | 47.542 | reference | — |
| hc2 | ActionnessProjection | 26.417 | 52.824 | 47.542 | +3.755 [+0.653, +6.744] | 4/1 |
| hc2 | TemporalHead | 25.619 | 48.337 | 47.542 | +2.956 [+0.908, +5.188] | 1/0 |
| hc2 | SpatialOPD | 22.500 | 43.575 | 47.427 | -0.162 [-4.414, +2.549] | 3/1 |
| hc2 | Offline_GT_Head | 26.507 | 49.858 | 47.542 | +3.844 [+1.499, +6.540] | 0/0 |
| vidstg | Native | 12.742 | 34.547 | 35.515 | reference | — |
| vidstg | ActionnessProjection | 14.377 | 38.454 | 35.515 | +1.634 [+0.152, +3.348] | 1/0 |
| vidstg | TemporalHead | 13.688 | 37.369 | 35.515 | +0.945 [-0.174, +2.296] | 2/0 |
| vidstg | SpatialOPD | 16.861 | 34.547 | 48.513 | +4.119 [-0.135, +8.124] | 2/1 |
| vidstg | Offline_GT_Head | 28.506 | 80.406 | 35.515 | +15.764 [+10.823, +21.225] | 0/0 |

Temporal arms keep all boxes fixed, so their fixed-GT-support sIoU is identical to Native. SpatialOPD retains Native time, so its tIoU is identical to Native. Offline GT supervision is not a task-score upper bound: frozen boxes, finite updates and shrink can still harm an individual query.

| Target | Complete paired contrast | vIoU, pp [95% CI] | tIoU, pp [95% CI] | sIoU, pp [95% CI] |
|---|---|---|---|---|
| hc2 | ActionnessProjection − Native | +3.755 [+0.653, +6.744] | +9.249 [+3.504, +15.147] | +0.000 [+0.000, +0.000] |
| hc2 | TemporalHead − Native | +2.956 [+0.908, +5.188] | +4.762 [+1.490, +8.251] | +0.000 [+0.000, +0.000] |
| hc2 | SpatialOPD − Native | -0.162 [-4.414, +2.549] | +0.000 [+0.000, +0.000] | -0.116 [-6.822, +5.148] |
| hc2 | Offline_GT_Head − Native | +3.844 [+1.499, +6.540] | +6.283 [+2.985, +9.920] | +0.000 [+0.000, +0.000] |
| hc2 | TemporalHead − ActionnessProjection | -0.799 [-3.435, +1.900] | -4.487 [-9.727, +0.516] | +0.000 [+0.000, +0.000] |
| hc2 | SpatialOPD − ActionnessProjection | -3.917 [-9.116, +0.691] | -9.249 [-15.147, -3.504] | -0.116 [-6.822, +5.148] |
| hc2 | Offline_GT_Head − ActionnessProjection | +0.089 [-3.124, +3.356] | -2.966 [-8.812, +2.657] | +0.000 [+0.000, +0.000] |
| hc2 | SpatialOPD − TemporalHead | -3.118 [-7.828, +0.632] | -4.762 [-8.251, -1.490] | -0.116 [-6.822, +5.148] |
| hc2 | Offline_GT_Head − TemporalHead | +0.888 [-0.495, +2.261] | +1.521 [-1.336, +3.883] | +0.000 [+0.000, +0.000] |
| hc2 | Offline_GT_Head − SpatialOPD | +4.006 [-0.008, +9.148] | +6.283 [+2.985, +9.920] | +0.116 [-5.148, +6.822] |
| vidstg | ActionnessProjection − Native | +1.634 [+0.152, +3.348] | +3.907 [+0.891, +7.253] | +0.000 [+0.000, +0.000] |
| vidstg | TemporalHead − Native | +0.945 [-0.174, +2.296] | +2.821 [+0.251, +5.739] | +0.000 [+0.000, +0.000] |
| vidstg | SpatialOPD − Native | +4.119 [-0.135, +8.124] | +0.000 [+0.000, +0.000] | +12.998 [+5.959, +19.626] |
| vidstg | Offline_GT_Head − Native | +15.764 [+10.823, +21.225] | +45.859 [+36.898, +55.196] | +0.000 [+0.000, +0.000] |
| vidstg | TemporalHead − ActionnessProjection | -0.689 [-1.610, -0.020] | -1.085 [-2.674, +0.194] | +0.000 [+0.000, +0.000] |
| vidstg | SpatialOPD − ActionnessProjection | +2.485 [-2.687, +6.884] | -3.907 [-7.253, -0.891] | +12.998 [+5.959, +19.626] |
| vidstg | Offline_GT_Head − ActionnessProjection | +14.130 [+9.265, +19.660] | +41.952 [+32.564, +51.790] | +0.000 [+0.000, +0.000] |
| vidstg | SpatialOPD − TemporalHead | +3.174 [-1.628, +7.351] | -2.821 [-5.739, -0.251] | +12.998 [+5.959, +19.626] |
| vidstg | Offline_GT_Head − TemporalHead | +14.819 [+9.869, +20.370] | +43.037 [+33.793, +52.828] | +0.000 [+0.000, +0.000] |
| vidstg | Offline_GT_Head − SpatialOPD | +11.645 [+5.162, +18.599] | +45.859 [+36.898, +55.196] | -12.998 [-19.626, -5.959] |

| Target | Head−Native positive/negative parents | Gross gain/loss, pp | Worst parent, pp | SSL/GT gradient cosine mean | Negative cosine count |
|---|---:|---:|---:|---:|---:|
| hc2 | 15/8 | 3.511/0.555 | -8.696 | 0.3400 | 8/32 |
| vidstg | 14/5 | 1.369/0.423 | -5.470 | -0.2988 | 25/32 |

The gradient cosine is computed in the existing temporal-head parameter space between the actual first SSL gradient and the same-source offline GT-head gradient. It diagnoses signal alignment on this saved query; it is not a deployment selection criterion. Original source/offset support and legal-pair maximum tIoU, teacher-target tIoU, all five update steps, actual SSL losses, finite last-output tIoU, observed/unobserved spatial means and every negative case stay in ROWS.

| Target | Recorded complete temporal query s | Capture/fit/audit/reinsertion s | GPU reinsertion s | Original spatial fit s | Offline CPU GT head s | Temporal peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 3.1655 | 1.8392 | 0.1547 | 1.4200 | 0.0247 | 4.930 |
| vidstg | 3.0184 | 2.0470 | 0.2175 | 0.3391 | 0.0198 | 13.989 |

Timings are actual recorded wall measurements with observation recording, frozen capture, reinsertion and existing cache conditions; they are not cold deployment benchmarks. Original SpatialOPD fitting time is retained from its exact historical fit and no spatial fit is rerun for P6. The temporal query includes its actual recorded components once. No new DINO inference is performed in P6; original frozen packed expert evidence is hash-bound and fully read back. Offline CPU supervised cost is reported separately.

Four qualification queries have eight complete original-versus-recorder GPU fits with scientific tensors and predictions bitwise equal, source process hashes unchanged, then the first two formal fits per target must match their qualified recorded fits before acceptance. Root reads all64 complete GPU head math dictionaries, all64 complete original spatial dictionaries and all64 offline supervised dictionaries; the complete two-layer head backward, native-head-output VJP, every active AdamW state, source reset, last/shrunk output and full original spatial1792 chain are independently checked. The original CPU64 formulas use explicit float32 accumulation error bounds; they do not certify the full decoder Jacobian, CUDA transcendental kernels or future numerical/OOM safety. The first offline supervised CPU fits are compared with original unrecorded CPU fits bitwise; CPU/native-logit agreement uses declared accumulation bounds and interval equality, not a CPU/GPU-bitwise claim.

Actual complete counts: `{"complete_GPU_head_math_dicts": 64, "complete_offline_GT_math_dicts": 64, "complete_spatial_math_dicts": 64, "deployment_outputs": 256, "formal_queries": 64, "head_backward_rounds": 320, "head_path_state_coordinates": 25461504, "observed_unobserved_checks": 128, "official_dense_scalar_checks": 960, "offline_CPU_GT_fits": 64, "offline_GT_backward_rounds": 320, "offline_GT_path_state_coordinates": 25461504, "qualified_formal_complete_pairs": 4, "second_opaque_prediction_input_receipt_SHA_checks": 256, "spatial_rounds": 1600, "spatial_state_coordinates": 114688}`. Official/dense comparison is made for all64queries ×5arms ×3metrics, and all original input/prediction/receipt SHA/bytes are checked again after the full root scan.

Six distinct post-hoc cases per the fixed maximum/minimum temporal effect and least gradient cosine selection are illustrative, not efficacy estimates. The actual original spatial-action/sample-to-GT diagnostic has 8,320 comparisons. Old fits without saved GPU action tensors use the explicitly labeled offline float64 sigmoid of their saved sample logits; this is not the newer same-GPU-action reward audit. Empty GT-evaluable admission supports are explicitly reported. All six report PNG/PDF pairs and six original-pixel RGB case sheets were actually viewed. A display-only revision improves bar labels and selects five RGB positions including actual GT support; original plots/cases/RGB/GT geometry/predictions/scores remain intact. Green boxes show original official spatial GT; frames with no spatial GT are labeled honestly. Private RGB/query/GT geometry is never published.

Three unsealed CPU helper problems remain preserved with source/runtime/receipts: original P0 input interpretation, an unlaunched scorer signature binding, and the display-helper module path. The final runtime revision002 predates every P6 GPU fit; original scientific fits, inputs, parameters, old payloads and original receipts are unchanged. The display-only helper changes do not affect any prediction or scientific runtime.

The anonymous public package preserves every actual score, paired contrast, negative tail, cost, signal chain, configuration, source hash and root/visual receipt. Its portable audit reproduces scalar arithmetic and exact public byte bindings; it cannot recreate private media, GT arrays, prefix tensors or fits. Scientific conclusions remain conditional: existing temporal evidence can help, head adaptation is not uniformly superior to the direct projection, and the offline VidSTG diagnostic leaves a large gap. No new temporal branch or automatic route promotion is warranted. P6 phase closure requires successful remote byte verification and archive maintenance; whole-suite closure additionally requires all original P1–P6 closing receipts and a real FINAL_COMPLETION. EATA and every historical paused queue remain paused.
