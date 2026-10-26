# TubeDETR selected spatial configuration: complete VidSTG population

2026-10-26: all 10,303 queries / 732 parent videos have complete prediction seals and independent CPU scores. The selected development profile b331286fb446e910 was frozen before this run. Source retains the backbone's native boxes and MAP interval; Ours retains the previously fixed Selected_interval. Full results did not select or change any parameter.

| Metric (%) | Native Source | Selected method | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Dense sIoU | 41.437894 | 41.213025 | -0.224869 |
| Unobserved-frame sIoU | 41.430386 | 41.186947 | -0.243439 |
| vIoU | 13.116927 | 15.929212 | +2.812284 |
| tIoU | 28.674021 | 35.231971 | +6.557950 |

The target of a positive spatial increment was not achieved on this complete VidSTG population. The parent-cluster 95% interval for the dense sIoU change is [-0.505827, +0.049920] percentage points. There are 2,364 spatial wins, 2,526 losses and 5,413 ties; 826 queries lose more than five spatial percentage points. The temporal interval is fixed from the earlier method, so its improvement is not attributed to spatial tuning. All positive and negative per-query scalar results are retained in all_signed_full_results.json.gz. This population and the development panel had prior research exposure; this is not an independent unexposed test.

Root first verified every new payload and receipt SHA, request/plan/parameter/amendment binding, all 61,818 finite arrays, positive box sizes, and bitwise equality of Source, physical frames, observations, native intervals and fixed selected intervals against the sealed reference. Only after complete prediction verification did a separate audit read labels, independently recompute every signed result and aggregate, parent-cluster intervals, and 32 reference-metric arm checks. The original reference NPZ files were opened only for array equality, without rehashing old full evidence or model weights. See full_root_verification.json.

The HC2 population is still running and is not included in this report. This domain report does not represent both-domain completion or final publication of the complete tuning task. LLaVA remains paused until the TubeDETR task has completed its required verification and publication.
