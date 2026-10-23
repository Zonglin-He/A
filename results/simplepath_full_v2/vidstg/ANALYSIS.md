# SimplePath + all-video Native2D: complete VidSTG result

TA-STVG, HC-STVG-v2 source checkpoint to official VidSTG10303 queries/732 parent sources. Fixed native final-layer temporal prior, kappa64/beta1/one round/context1.5, one all-video prototype pair, SimplePath, one ReferDINO/eight fixed observations. Equal-query aggregation; 10000 parent-cluster bootstrap samples. No full-result parameter selection.

| Method | m_tIoU (%) | m_vIoU (%) | vIoU@0.3 (%) | vIoU@0.5 (%) | Unobserved sIoU (%) |
|---|---:|---:|---:|---:|---:|
| Source | 33.804566 | 11.777655 | 13.636805 | 3.562069 | 33.025378 |
| SharedInit | 33.804566 | 13.415324 | 17.606522 | 7.541493 | 35.000813 |
| Interpolation | 33.804566 | 14.228927 | 18.480054 | 7.929729 | 35.512989 |
| OursS | 33.804566 | 14.030472 | 18.227701 | 7.822964 | 35.781463 |
| Full | 34.604378 | 14.383314 | 19.052703 | 8.143259 | 35.768467 |
| NoLocal | 34.604378 | 14.085590 | 18.606231 | 8.191789 | 35.000813 |
| Independent | 34.604378 | 13.826629 | 18.091818 | 7.298845 | 34.534441 |
| Global6 | 34.766274 | 14.268486 | 18.965350 | 8.327672 | 35.332099 |

Full minus Source: tIoU +0.799812 pp, 95% parent-cluster CI [0.5811005429039172, 1.0320784019259015]; complete vIoU +2.605659 pp, CI [2.2451871242640093, 2.9720500021345226].
The predefined Source+0.9 pp query-equal m_tIoU point-estimate acceptance is **target_not_met**, actual +0.799812 pp. A point-estimate pass does not imply the CI lower bound exceeds0.9. No retrospective threshold or parameter change.

Time: 3651 improved / 2786 harmed / 3866 unchanged. Source-relative declines exceeding5 pp: temporal 663, complete-v 1162, union 1637/10303 (15.8886%). Temporal/v counts overlap; the union counts each query once.

| GT duration group | Queries | Full-Source tIoU (pp) | Full-Source vIoU (pp) |
|---|---:|---:|---:|
| short_le5s | 6060 | +0.103052 | +1.262735 |
| long_gt5s | 4243 | +1.794948 | +4.523670 |

Full minus OursS (NativeTemporal alias) complete-v +0.352842 pp, CI [0.24505598130669273, 0.4666261150990796]. Shared initialization is a mechanism intermediate; no deletion ablation is claimed. NoLocal and Independent hold Full time/evidence fixed. Global6 is an imported frozen reference, never the production prior.

All10303 predictions were sealed before independent fixed-label CPU scoring; 82424 official-vs-dense metric checks, maximum absolute error 3.77e-15. Original Source and Global6 published summaries reproduce; every receipt/prediction/capture SHA and original bound source/resource/input check passed. No inference or scoring was repeated by this verification.

All signed per-query results, both-sign outcomes, severe declines, fixed-support mechanism data, extent/missingness analysis, distance curves and worker-segment natural cache costs remain in report.json and signed/severe TSVs. Complete-box jointly available quality and zero-filled missing-slot quality are distinct endpoints. NativeTemporal=OursS is an alias. Natural cache acquisition times are not uncached deployment costs; fair timing and privately reviewed qualitative figures are separate pending deliverables. Raw media, captions, GT coordinates, features, masks and weights remain private.
