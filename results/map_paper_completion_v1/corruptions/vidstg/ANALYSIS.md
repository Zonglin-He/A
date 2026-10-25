# Complete controlled-view current-MAP results: vidstg

**All128queries/128parents ×3views have been recomputed, sealed, officially scored and replay-audited.** The same transformed RGB drives native Source, eight-observation expert context, actual auxiliary ROI pooling, prototype update, joint-MAP and new-interval final DP. No final-box-only conversion is used. Original native Source and clean/current FullMAP are explicit matched-cohort references. These are exposed historical confirm panels, not a full target set or new unseen confirmation.

| View | Native Source t / v % | FullMAP t / v % | Δv Full−Source pp [95% CI] | Δv Full−CleanFull pp [95% CI] | Severe t/v/union Full vs same-view Source |
|---|---:|---:|---:|---:|---|
| Blur | 31.889488 / 12.297238 | 34.972705 / 14.395653 | +2.098415 [-0.171280, 4.477127] | -0.897567 [-2.147878, 0.371300] | 18/23/32 |
| JPEG | 31.908497 / 13.369802 | 34.785155 / 14.295422 | +0.925620 [-1.092325, 3.023889] | -0.997798 [-1.911201, -0.145619] | 20/24/35 |
| Dim | 31.104485 / 13.433187 | 34.752583 / 14.660561 | +1.227374 [-1.066231, 3.508701] | -0.632659 [-1.523966, 0.205764] | 15/24/31 |

Clean references on the same128queries: native Source t/v=32.121535/13.322903%; current FullMAP=35.018482/15.293220%. These absolute values must not be compared directly with complete-target tables.

All three Vid same-view FullMAP−nativeSource vIoU intervals cross zero, despite positive mean differences. JPEG lowers FullMAP relative to its own clean output by0.997798pp with CI[-1.911201,-0.145619], while same-view Source−cleanSource vIoU has an interval crossing zero. Therefore a blanket corruption-robustness advantage is unsupported. Blur and Dim clean-to-corrupt FullMAP vIoU intervals cross zero.

## All signed and duration-specific effects

| View / signed comparison | Δt pp [95% CI] | Δv pp [95% CI] | Severe t/v/union | Rescue union |
|---|---:|---:|---|---:|
| Blur / FullMAP-Source | +3.083217 [1.072183, 5.316125] | +2.098415 [-0.171280, 4.477127] | 18/23/32 | 54 |
| Blur / Source-CleanSource | -0.232047 [-2.070639, 1.638662] | -1.025665 [-2.090074, 0.010000] | 22/15/28 | 16 |
| Blur / FullMAP-CleanFullMAP | -0.045777 [-1.465273, 1.297867] | -0.897567 [-2.147878, 0.371300] | 18/17/27 | 19 |

Blur duration groups (≤5s and >5s), all pointwise parent-cluster intervals and observed/unobserved effects are retained in `Blur/report.json`; every scalar signed outcome is in that view's TSVs.

| JPEG / FullMAP-Source | +2.876659 [0.782318, 5.305369] | +0.925620 [-1.092325, 3.023889] | 20/24/35 | 55 |
| JPEG / Source-CleanSource | -0.213038 [-1.657598, 1.171046] | +0.046899 [-0.474031, 0.600034] | 10/6/13 | 12 |
| JPEG / FullMAP-CleanFullMAP | -0.233327 [-1.484394, 0.866439] | -0.997798 [-1.911201, -0.145619] | 11/13/20 | 19 |

JPEG duration groups (≤5s and >5s), all pointwise parent-cluster intervals and observed/unobserved effects are retained in `JPEG/report.json`; every scalar signed outcome is in that view's TSVs.

| Dim / FullMAP-Source | +3.648098 [1.429749, 6.051235] | +1.227374 [-1.066231, 3.508701] | 15/24/31 | 58 |
| Dim / Source-CleanSource | -1.017050 [-3.294666, 1.160052] | +0.110284 [-0.783611, 1.028377] | 21/9/22 | 17 |
| Dim / FullMAP-CleanFullMAP | -0.265899 [-1.915579, 1.350983] | -0.632659 [-1.523966, 0.205764] | 19/13/25 | 19 |

Dim duration groups (≤5s and >5s), all pointwise parent-cluster intervals and observed/unobserved effects are retained in `Dim/report.json`; every scalar signed outcome is in that view's TSVs.

## Resource and inference boundaries

Blur is5×5 Gaussianσ1.5; JPEG isquality30/subsampling2; Dim isuint8×0.5. All768source/view forwards were actually recomputed. Each Source raw payload and SHA was saved before expert lookup; compact arrays were cold-replayed and read back before only this new committed query raw scratch was deleted. The exact same transformed-RGB/context old teacher payload was reused on all768views, with original acquisition cost retained. This is zero *new* teacher calls, not zero expert deployment cost. No old cache/root was modified.

Actual native Source, old same-view compatibility, eight positions, geometry, real pooling and all current-MAP compact arrays were checked. Source/teacher/native-grid/precision/prototype settings are unchanged. There is no perturbation-specific retuning. No MBR performance comparison is introduced.

Costs are exported in `cost_segments.json`: actual per-view Source native-forward, current expert lookup wall, original teacher forward, complete query-pipeline and finish wall are separate nested quantities. They must not be added. The original per-worker total includes loading/outer-loop/decode; those components are not separately instrumented here. These research/cache timings do not replace the separately measured uncached deployment experiment. Intervals are pointwise, not familywise; the same128parents appear in three views, and no pooled384-independent-query claim is made.
