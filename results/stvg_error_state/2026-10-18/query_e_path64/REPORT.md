# Query-only E64: sealed update-path diagnosis

CPU-only retrospective readback of every saved step from288f2876. No model forward/backward, fit, expert call, parameter change, new cohort or GT-selected deployment step. MBR, teacher E and query-only interface remain fixed.

## Main findings

The evidence supports prioritizing the actual update response before blaming all severe failures on E. H128 moves farther from E while real localization collapses; H008 sparse first improves its four-point objective, then loses both geometric and real quality and becomes flat-reward/no-update. Dense avoids that collapse but does not reproduce the full E benefit.

Feedback alignment remains a distinct problem: in HC9 cases, Dense ends closer to E yet has lower real vIoU;4 cases lose both E overlap and vIoU. The latter4 account for39.208147 of50.427315 negative parent-pp (77.752%). Vid2 negative cases both have net E-overlap gain. These are descriptive categories, not pure causal allocations or identity labels. E overlap averages all decoder frames and differs from GT/event/interpolation evaluation support; this mismatch does not uniquely identify a bad teacher or wrong identity.

## All64 final-state diagnosis

| Domain / arm | Mean E overlap change pp | Mean vIoU change pp | E up / v down | E down / v down | Net E loss queries |
|---|---:|---:|---:|---:|---:|
| hc2 / Sparse_OPD | -3.934418 | -2.848706 | 4/32 | 7/32 | 16/32 |
| hc2 / Dense_OPD | +0.855650 | -1.030547 | 9/32 | 4/32 | 5/32 |
| hc2 / Fixed_rollout | -0.584034 | +0.032809 | 10/32 | 5/32 | 10/32 |
| hc2 / Dense_Direct | +2.925330 | +1.294859 | 10/32 | 2/32 | 5/32 |
| vidstg / Sparse_OPD | -0.174865 | +0.328909 | 2/32 | 0/32 | 6/32 |
| vidstg / Dense_OPD | +0.206511 | +0.229027 | 2/32 | 0/32 | 6/32 |
| vidstg / Fixed_rollout | +0.040171 | +0.061186 | 3/32 | 3/32 | 9/32 |
| vidstg / Dense_Direct | +2.363362 | +0.511457 | 1/32 | 1/32 | 4/32 |

Full64 and active56 summaries are separate in RESULTS_PUBLIC.json. Every path is available in the eight PATHS files; no-reference/no-update/unknown and positive cases remain. Signs use fixed1e-10 tolerance; a small numerical decrease is not itself a scientifically meaningful failure.

## Fixed cases

| Case / arm | E overlap initial -> final % | vIoU initial -> final % | Actual updates |
|---|---|---|---:|
| H008 / Sparse_OPD | 67.614762 -> 23.856692 | 60.524579 -> 2.238566 | 6 |
| H008 / Dense_OPD | 67.614762 -> 69.868977 | 60.524579 -> 60.627595 | 40 |
| H128 / Dense_OPD | 28.854595 -> 10.369711 | 44.669483 -> 10.389052 | 40 |
| H128 / Dense_Direct | 28.854595 -> 27.604250 | 44.669483 -> 45.203972 | 40 |
| V038 / Dense_OPD | 75.438773 -> 75.728148 | 56.983348 -> 56.142706 | 10 |
| V038 / Dense_Direct | 75.438773 -> 75.435812 | 56.983348 -> 56.575808 | 10 |
| H119 / Dense_OPD | 47.153178 -> 47.564909 | 28.586077 -> 36.957890 | 40 |
| H119 / Dense_Direct | 47.153178 -> 48.158792 | 28.586077 -> 35.761646 | 40 |
| V016 / Dense_OPD | 11.981005 -> 18.222155 | 18.803103 -> 20.672985 | 10 |
| V016 / Dense_Direct | 11.981005 -> 20.288571 | 18.803103 -> 20.576807 | 10 |

H008 sparse four-point E overlap67.286268% ->67.765027% at step2 (exact trace retained), then50.9937% at step3 and0 at step6; vIoU60.524579% ->60.5841% ->48.2245% ->2.238566%, then unchanged through step40 because rewards become exactly flat. Dense final E overlap69.868977%, vIoU60.627595%. This is not simply a teacher-quality failure.

H128 Dense E overlap28.854595% ->10.369711% and vIoU44.669483% ->10.389052%; Direct ends at E overlap27.604250% and vIoU45.203972%. Thus the severe Dense result is not evidence that faithfully fitting E alone caused this loss.

V038 Dense first shows real improvement through early/middle saved steps but later loses it: E overlap75.438773% ->75.728148%, final vIoU56.983348% ->56.142706%. This combines a weak net E response with late deterioration. No intermediate step becomes a deployment rule.

H119 remains a positive control: Dense E overlap47.153178% ->47.564909% while vIoU28.586077% ->36.957890%; a small all-frame E-overlap change can accompany a large task benefit, so small mean E gains cannot alone establish a response-capacity limit. V016 moves closer to a globally poorer E yet improves real vIoU, further showing that partial fitting need not reproduce the complete teacher result.

## Local update response

| Domain / Dense OPD | Actual updates | First-order parameter loss descent | Same-rollout surrogate rises | Negative mean-gradient/actual-output cosine |
|---|---:|---:|---:|---:|
| hc2 | 1279 | 1141 | 1005 | 178 |
| vidstg | 220 | 202 | 173 | 55 |

For each SAME frozen rollout, the original Gaussian likelihood/control-variate surrogate is recomputed before and after the actual saved decoder update. Parameter gradient dot actual step often predicts descent while finite decoder output raises that surrogate. H008 sparse step3 predicts−0.189964 but actual same-rollout loss rises about1350.908119, with output L2 step1.773065 and four-point E overlap down16.771291pp. H128 Dense step1 predicts−3.074680 but the same-rollout loss rises389.807091, output step1.320034 and E overlap down1.408254pp. These isolate an actual local-update mismatch rather than an incorrect GT checkpoint choice.

The auxiliary local free-mean optimum is analytically derived from the existing frozen-rollout objective; its scale ratio is diagnostic only. It is neither a new output readout nor proof that every large step hurts. The surrogate includes a KL baseline and is not true task loss: H008 Dense can improve E and retain real localization even though this local surrogate rises. Do not turn these counts into an automatic gate or claim all Adam/sigma/temperature mechanisms have been causally isolated.

## Decision scope

Keep the same frozen-decoder query-only interface, fixed E/MBR and original predictions. Prioritize a future matched test of actual output-step/response control before modifying the teacher or merely increasing rounds/temperature. No such test is launched by this diagnosis. HC E-up/GT-down cases separately require feedback-use/metric-support investigation; optimization alone cannot be assumed to fix them. Do not splice cached paths, pick GT-best steps, infer identity errors from IoU, expand data or introduce temporal adaptation.

[Summary and all64 rows](RESULTS_PUBLIC.json) · [Fixed case paths](CASES.json) · [Local response diagnostics](LOCAL_RESPONSE.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [All64 scatter](ALL64_E_V_CHANGE.svg) · [Case plots](CASE_PATHS.svg).

## Additional current-code boundary

The current PosDecoder refreshes intermediate predicted boxes via `pred_boxes = new_pred_boxes.detach()` before subsequent layers. Thus saved autograd directions do not include that reference-update dependency, while a real decoder recomputation refreshes the references. This is a competing explanation for finite-response mismatch; it does not establish how much came from stop-gradient, curvature, Adam momentum or step size. No new finite-difference/model test was run. [Current source evidence](STATIC_SOURCE_EVIDENCE.json).
