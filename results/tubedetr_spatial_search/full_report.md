# Selected TubeDETR spatial configurations: complete two-domain results

All 12,205 predictions (VidSTG 10,303 / 732 parents; HC2 1,902 / 227 parents) have complete seals and separate CPU scores. Root first verified all new payload/receipt hashes, request/plan/parameter/execution-amendment bindings, finite positive boxes, unchanged Source and fixed time/frame arrays. Every signed outcome, mean and parent-cluster interval was independently recomputed; 32 reference metric checks per domain also passed. See full_root_verification.json and each domain's full_root_verification.json. The original VidSTG audit was reused through small hashes, without rescanning its population at HC2 completion.

| Domain | Metric (%) | Native Source | Selected method | Delta (pp) |
| --- | --- | ---: | ---: | ---: |
| VidSTG | Dense sIoU | 41.437894 | 41.213025 | -0.224869 |
| HC2 | Dense sIoU | 55.894585 | 56.438276 | +0.543690 |
| VidSTG | vIoU | 13.116927 | 15.929212 | +2.812284 |
| HC2 | vIoU | 15.425087 | 27.148968 | +11.723881 |
| VidSTG | tIoU | 28.674021 | 35.231971 | +6.557950 |
| HC2 | tIoU | 26.800739 | 47.322454 | +20.521715 |

The two-domain positive-spatial-increment objective was not achieved. VidSTG remains negative: dense sIoU delta -0.224869 pp, parent-cluster 95% interval [-0.505827, +0.049920]. HC2 is positive: +0.543690 pp, interval [+0.172163, +0.904879]. Unobserved-frame spatial deltas are -0.243439 / +0.533390 pp. The 826 / 104 queries losing more than five spatial points and all other negative outcomes are retained in the two all_signed_full_results.json.gz files. No query was replaced and full scores did not trigger reselection.

Source always uses the backbone's native boxes and MAP time; extra MBR is not renamed Source. Selected intervals came from the earlier fixed method and were not tuned here. The temporal and combined vIoU gains are therefore not attributed entirely to spatial tuning. The development search selected 2,048 configurations per domain on an exposed 128-query panel; these complete populations also have prior research exposure and are not independent unexposed tests. Model weights remain frozen and no new teacher/expert calls were made.

Frozen protocol: plan.json, full_plan.json, selected_profiles.json and priority_resume_plan.json. The last file only changes user-authorized queue priority; the original numerical GPU AST and all bound source files remain intact. Both domain final seals and final_seal.json retain their publication_pending-at-sealing fields; the later maintenance publication receipt determines remote publication status. This completes this selected-configuration run, without claiming that VidSTG spatial tuning succeeded. LLaVA is the next authorized pending experiment.
