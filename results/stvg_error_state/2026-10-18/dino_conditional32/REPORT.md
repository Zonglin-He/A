# Student-conditioned DINO trajectory distillation:32-query pilot

All32 predictions sealed before new GT. HC16/Vid16 are deterministic subsets of the exposed original64, not independent confirmation. E and time final-MBR remain fixed. Three new arms use frozen DINO conditional feedback and a four-dimensional episodic external correction distribution. No QR/LN/backbone update.

| Domain | Source vIoU% | E vIoU% | One correction% | Fixed24% | Dynamic3% |
|---|---:|---:|---:|---:|---:|
| hc2 | 26.399499 | 26.861045 | 27.015681 | 26.341629 | 21.651524 |
| vidstg | 21.380083 | 28.187972 | 27.946652 | 28.163688 | 27.601864 |

| Domain | Contrast | Complete vIoU pp [95% CI] | Unobserved actual sIoU pp [95% CI] | Loss >5pp /16 |
|---|---|---:|---:|---:|
| hc2 | One_correction_minus_E_MBR | +0.154636 [-0.327328,+0.823511] | +0.240870 [-0.416596,+1.212640] | 0/16 |
| hc2 | Fixed_sample_distill_minus_E_MBR | -0.519417 [-1.765531,+0.602658] | -0.141653 [-1.749586,+1.386989] | 2/16 |
| hc2 | On_policy_distill_minus_E_MBR | -5.209521 [-10.532588,-0.597026] | -5.969576 [-12.900480,+0.278558] | 4/16 |
| hc2 | On_policy_distill_minus_Fixed_sample_distill | -4.690104 [-8.805721,-1.125146] | -5.827923 [-11.280211,-0.969625] | 4/16 |
| hc2 | On_policy_distill_minus_One_correction | -5.364157 [-10.527907,-0.888699] | -6.210446 [-12.968043,-0.264505] | 4/16 |
| vidstg | One_correction_minus_E_MBR | -0.241320 [-0.686274,+0.000820] | -0.387589 [-0.885682,-0.019464] | 0/16 |
| vidstg | Fixed_sample_distill_minus_E_MBR | -0.024284 [-0.194114,+0.106179] | -0.046859 [-0.353265,+0.289453] | 0/16 |
| vidstg | On_policy_distill_minus_E_MBR | -0.586109 [-2.006161,+0.278458] | -0.765391 [-2.701133,+0.738218] | 1/16 |
| vidstg | On_policy_distill_minus_Fixed_sample_distill | -0.561824 [-1.817395,+0.175506] | -0.718532 [-2.380407,+0.484959] | 1/16 |
| vidstg | On_policy_distill_minus_One_correction | -0.344788 [-1.370107,+0.318685] | -0.377802 [-1.902968,+0.871857] | 1/16 |

## Mechanism actually run

Native900 context queries are retained. Each student reference takes the content query of the nearest original encoder proposal by IoU. An additive attention mask isolates student candidates and removes the matched native slot from that added query's support. Ordinary native replay was exact; cloned references passed the prelocked1e-4 box tolerance in both domains. These are engineering checks, not grounding-quality proof.

The teacher corrects each sampled state at the original four images. Target token score>=.35 supplies an anchor correction; no score calibration or identity-probability claim. Shared4 teacher means are averages over accepted anchors, then the student Gaussian mean is the exact forward-KL mixture projection. This tests on-policy teacher-conditioned distillation, not the old fixed-E IoU-reward/Adam OPD objective. Fixed and dynamic sample24 actual states each; one-correction allocates24 repeated logical references and computes duplicates only once. Candidate processing count and decoder passes are different.

No accepted anchor returns its input sampled action as target. With finite Gaussian samples, averaging identity targets can still introduce sampling drift; unchanged feedback is not automatically unchanged distribution mean. Invalid normalized references get whole-action no-op, no clipping or replacement. These counts and all signed/unknown cases are retained in RESULTS_PUBLIC.json.

## Decision

`stop_this_teacher_interface_no_LR_rescue`. No automatic expansion or LR rescue. A negative result applies to this content initialization, frozen DINO reference interface, shared4 representation and Gaussian mean distillation; it does not invalidate every form of OPD or iterative correction. Sparse4 images with many candidate states are not dense independent full-video evidence. Time/tIoU is identical across all arms.

## Actual cost

GPU lease 68.941s; original-anchor full image/text encoding-forward 9.817s (includes native decoder); additional conditional decoder 3.315s. New encoded frames 112, logical references 7116, unique references 4724. No Source forward, source training, Qwen, SAM or new backbone. Existing E generation cost excluded; this is not complete first online deployment latency.

## Evidence and limitations

All five outputs and all signed paired query-level bootstrap intervals are public. Missing GT stays unknown, all no-support/no-op cases remain in denominator, no GT step selection. Private RGBs, original Query/parent identifiers, model weights, raw tensors and GT coordinates are excluded. Source/E exact prior metrics, all saved teacher-target-to-student updates and metrics were independently recomputed. No new identity labels.

## Necessary interpretation correction after saved-path readback

The dynamic HC loss is not clean evidence that state-conditioned visual feedback caused damage.15/16 HC dynamic queries receive zero accepted conditional feedback. All their teacher targets equal their sampled actions, yet empirical Gaussian mean projection moves the student. Their saved mean displacement is exactly the sum of the three epsilon sample means (error<1e-12). They contribute **-5.223792pp** to the overall **-5.209521pp** HC dynamic change; the single feedback-present query contributes +.014270pp. This is an implementation-level failure of the desirable no-feedback/no-update invariant. Fixed24 identity-target paths can also drift. No alternative teacher states were evaluated after GT, no new prediction or rule changed.

One-correction has no such sampling-mean drift because its repeated reference is the mean. It receives any qualified feedback on HC4/16 and Vid6/16 (Vid4 no E support retained). Its measured +.154636/-.241320pp changes have intervals spanning zero, so stable added utility is unestablished. Dynamic Vid11/16 receive some feedback; its overall -.586109pp interval also crosses zero.

The proper closure is: engineering decoder replay passed; this limited conditional content interface has not demonstrated reliable useful feedback; the specific distribution update confounds the on-policy comparison. Stop this implementation without LR rescue. It does **not** refute student-conditioned OPD as a mechanism. A no-feedback/no-update contract would be necessary before interpreting any future repaired implementation, but no repaired experiment is silently included here. See NO_FEEDBACK_DRIFT_DIAGNOSIS.json and CONCLUSION.json.

CPU independent-audit recovery only corrected FP32-vs-FP64 logarithm precision to match measured encoding; original1e-12 tolerance, all32 predictions and GT scores unchanged. All2016 actions,7116 conditional-anchor outputs,140 distillation rounds,96 final learned trajectory arrays,640 metric scalars and48 paired bootstrap summaries pass the final readback.
