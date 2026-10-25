# Complete controlled-view current-MAP results: hc2

**All128queries/128parents ×3views have been recomputed, sealed, officially scored and replay-audited.** The same transformed RGB drives native Source, eight-observation expert context, actual auxiliary ROI pooling, prototype update, joint-MAP and new-interval final DP. No final-box-only conversion is used. Original native Source and clean/current FullMAP are explicit matched-cohort references. These are exposed historical confirm panels, not a full target set or new unseen confirmation.

| View | Native Source t / v % | FullMAP t / v % | Δv Full−Source pp [95% CI] | Δv Full−CleanFull pp [95% CI] | Severe t/v/union Full vs same-view Source |
|---|---:|---:|---:|---:|---|
| Blur | 44.460067 / 24.345615 | 51.849417 / 30.381449 | +6.035835 [3.736412, 8.401176] | -0.050045 [-1.208594, 1.041669] | 17/18/28 |
| JPEG | 46.202452 / 24.940263 | 51.992538 / 30.231433 | +5.291170 [3.018314, 7.629205] | -0.200062 [-1.896836, 1.351186] | 16/18/28 |
| Dim | 45.093437 / 24.143950 | 52.972732 / 30.191805 | +6.047855 [3.570309, 8.516641] | -0.239690 [-2.080367, 1.374099] | 16/17/27 |

Clean references on the same128queries: native Source t/v=46.509478/25.124253%; current FullMAP=52.319705/30.431495%. These absolute values must not be compared directly with complete-target tables.

HC same-view FullMAP−nativeSource vIoU intervals are positive for all three corruptions. FullMAP differences against its own clean output have intervals crossing zero. This does not establish corruption-invariance or a universal natural-shift claim.

## All signed and duration-specific effects

| View / signed comparison | Δt pp [95% CI] | Δv pp [95% CI] | Severe t/v/union | Rescue union |
|---|---:|---:|---|---:|
| Blur / FullMAP-Source | +7.389350 [5.002128, 9.857061] | +6.035835 [3.736412, 8.401176] | 17/18/28 | 78 |
| Blur / Source-CleanSource | -2.049410 [-3.925942, -0.265132] | -0.778638 [-1.807969, 0.237091] | 30/12/30 | 17 |
| Blur / FullMAP-CleanFullMAP | -0.470288 [-2.151127, 1.046369] | -0.050045 [-1.208594, 1.041669] | 17/18/25 | 28 |

Blur duration groups (≤5s and >5s), all pointwise parent-cluster intervals and observed/unobserved effects are retained in `Blur/report.json`; every scalar signed outcome is in that view's TSVs.

| JPEG / FullMAP-Source | +5.790086 [3.316155, 8.387769] | +5.291170 [3.018314, 7.629205] | 16/18/28 | 67 |
| JPEG / Source-CleanSource | -0.307025 [-2.206023, 1.397029] | -0.183990 [-1.624374, 1.067896] | 18/7/18 | 29 |
| JPEG / FullMAP-CleanFullMAP | -0.327167 [-2.170977, 1.302050] | -0.200062 [-1.896836, 1.351186] | 19/19/27 | 28 |

JPEG duration groups (≤5s and >5s), all pointwise parent-cluster intervals and observed/unobserved effects are retained in `JPEG/report.json`; every scalar signed outcome is in that view's TSVs.

| Dim / FullMAP-Source | +7.879295 [5.173876, 10.688296] | +6.047855 [3.570309, 8.516641] | 16/17/27 | 76 |
| Dim / Source-CleanSource | -1.416040 [-4.183293, 1.192243] | -0.980303 [-2.748976, 0.680160] | 32/25/36 | 29 |
| Dim / FullMAP-CleanFullMAP | +0.653027 [-1.485713, 2.658791] | -0.239690 [-2.080367, 1.374099] | 20/22/29 | 32 |

Dim duration groups (≤5s and >5s), all pointwise parent-cluster intervals and observed/unobserved effects are retained in `Dim/report.json`; every scalar signed outcome is in that view's TSVs.

## Resource and inference boundaries

Blur is5×5 Gaussianσ1.5; JPEG isquality30/subsampling2; Dim isuint8×0.5. All768source/view forwards were actually recomputed. Each Source raw payload and SHA was saved before expert lookup; compact arrays were cold-replayed and read back before only this new committed query raw scratch was deleted. The exact same transformed-RGB/context old teacher payload was reused on all768views, with original acquisition cost retained. This is zero *new* teacher calls, not zero expert deployment cost. No old cache/root was modified.

Actual native Source, old same-view compatibility, eight positions, geometry, real pooling and all current-MAP compact arrays were checked. Source/teacher/native-grid/precision/prototype settings are unchanged. There is no perturbation-specific retuning. No MBR performance comparison is introduced.

Costs are exported in `cost_segments.json`: actual per-view Source native-forward, current expert lookup wall, original teacher forward, complete query-pipeline and finish wall are separate nested quantities. They must not be added. The original per-worker total includes loading/outer-loop/decode; those components are not separately instrumented here. These research/cache timings do not replace the separately measured uncached deployment experiment. Intervals are pointwise, not familywise; the same128parents appear in three views, and no pooled384-independent-query claim is made.
