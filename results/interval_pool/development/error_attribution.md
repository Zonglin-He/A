# HC64 first-round temporal error attribution

All64 predictions, the twelve single-layer/single-offset MBRs and old Global6 were frozen before GT. This is attribution on exposed development data, not a newly deployed method or independent confirmation. MBR, physical frame endpoints and original BoxFixed remain unchanged.

| Global6 error | Mean |
|---|---:|
| Signed start error s (prediction − GT) | -0.968303 |
| Signed end error s | -0.036419 |
| Signed start / GT duration | -0.247224 |
| Signed end / GT duration | 0.064480 |
| Overcoverage s | 3.014370 |
| Undercoverage s | 2.082485 |
| Intersection s | 5.633681 |
| Predicted / GT duration | 1.311704 |
| tIoU | 0.559613 |

Early starts: 32/64; late ends: 30/64; any undercoverage: 52/64; nearly disjoint (tIoU < .1): 1/64.

| Diagnostic | Query mean [parent bootstrap 95% CI] |
|---|---:|
| Hull − original-view effective mass at tIoU ≥ .5, pp | +19.847127 [15.739566, 23.938598] |
| Hull − original-view effective mass at tIoU ≥ .7, pp | +9.588805 [6.823089, 12.550364] |
| 13-candidate tIoU oracle − Global6, pp | +7.679542 [5.572988, 10.066703] |

The independently recomputed per-layer expected-length identity has maximum error 1.95e-14 seconds. This verifies the hull operator; it does not by itself show that the actual MBR decision is too long or incorrect. All twelve original view distributions and six hull distributions have separate mass, length and endpoint-spread summaries in JSON.

## Fixed descriptive groups

Event duration: <5 s, 5–10 s, ≥10 s. Unlabeled endpoint disagreement: <.25 s, .25–1 s, ≥1 s. The disagreement is the six-layer mean of (E|start₀−start₁| + E|end₀−end₁|)/(2 fps); it includes uncertainty within each view. No group changes online inference or selects confirmation queries.

| Group | Queries | Overcoverage s | Undercoverage s | tIoU | Effective-mass .5 delta pp |
|---|---:|---:|---:|---:|---:|
| short_<5s | 13 | 5.07205 | 0.71358 | 0.41912 | +8.35175 |
| medium_5-10s | 37 | 2.71259 | 1.82419 | 0.59215 | +20.45627 |
| long_>=10s | 14 | 1.90123 | 4.03624 | 0.60409 | +28.91152 |
| low_<.25s | 0 | — | — | — | — |
| medium_.25-1s | 0 | — | — | — | — |
| high_>=1s | 64 | 3.01437 | 2.08249 | 0.55961 | +19.84713 |

## Evidence and limits

The oracle uses exactly the twelve saved single-view MBR decisions plus saved Global6. It never searches all legal intervals with GT, generates no additional DP, and is not a deployable result or the full model upper bound. The fixed-space vIoU oracle is separately labeled; it cannot be substituted for full-method accuracy.

Benefit and harm cases are selected by the declared symmetric rule and accompany the complete signed anonymous table. Mass changes and MBR outcomes are distinct: a view may gain effective mass through hull coverage, and MBR can select an interval different from expected length. Correlations are descriptive and do not isolate a causal mechanism.

未得到支持（本轮归因）：自动描述提示，非正式机制裁决。需结合全部质量、过/欠覆盖和正反案例作人工结论；第二轮固定干预才检验替换是否有效。

448 posterior attribution: not measured. No complete448 interval-posterior artifact is supplied to this attribution; no new448 computation is authorized or performed.

The two fixed rounds proceed without temperature, layer-weight, duration-prior or spatial-feedback search. Historical full panels in both domains have prior exposure; parent-isolated frozen follow-up must be called development recheck unless independent nonexposure can actually be established.
