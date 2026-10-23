# CG-STVG fixed confirmation

128fixed independent parents/domain, separate from screening but full official datasets historically exposed. All256query predictions sealed before1,280official CG/common/dense metric checks, maximum discrepancy1.4432899320127035e-15. No screening accuracy selection, no parameter changes. Native420FP32, cross-source EMA, single native grid, documented input segment-boundary conversion, exact independent-window storage schedule; see recovery document for the OOMed unmodified executable and successful audited native-equation reference.

| Target | Arm | m_tIoU (%) | m_vIoU (%) | m_sIoU (%) |
|---|---|---:|---:|---:|
| hc2 | SourceNative | 22.143354 | 14.735171 | 61.064508 |
| hc2 | Source | 44.346164 | 27.498238 | 61.064508 |
| hc2 | SharedCorrection | 44.346164 | 25.495463 | 56.643459 |
| hc2 | Space | 44.346164 | 25.239550 | 55.651934 |
| hc2 | Full | 48.313139 | 27.732755 | 55.810698 |
| vidstg | SourceNative | 16.916145 | 8.659275 | 45.937051 |
| vidstg | Source | 24.333429 | 12.428563 | 45.937051 |
| vidstg | SharedCorrection | 24.333429 | 9.844373 | 36.089300 |
| vidstg | Space | 24.333429 | 9.447854 | 35.670899 |
| vidstg | Full | 28.049772 | 11.140072 | 35.424215 |

SourceNative=original MAP; Source=MBR on identical native boxes. SharedCorrection/Space/Full share MBR. MAP→MBR improvements are decoder effects, not external visual-evidence gains.

hc2: Full−SourceMBR Δt +3.966975pp CI[1.4920597920245364, 6.592663074347839]; Δv +0.234517pp CI[-2.2455687210994073, 2.7295446507441423]. Full−Space Δv +2.493205pp CI[1.0099439117248963, 4.034300993399041]. Full−Source unobserved sIoU -5.597302pp CI[-8.71009755556777, -2.50832176803488]. SevereSource-relative union: SharedCorrection 20, Space 32, Full 58 of128. Exact t/v/union counts and all signed outcomes are retained in the domain files.
vidstg: Full−SourceMBR Δt +3.716343pp CI[1.6076823183284543, 5.9432811047088485]; Δv -1.288490pp CI[-3.596917925638583, 0.9717753246398352]. Full−Space Δv +1.692219pp CI[0.5402539118372488, 2.9400509459686086]. Full−Source unobserved sIoU -10.689287pp CI[-15.250905414637359, -6.264157726154646]. SevereSource-relative union: SharedCorrection 22, Space 24, Full 31 of128. Exact t/v/union counts and all signed outcomes are retained in the domain files.

Interpretation: temporal feedback improves tIoU and Full−Space vIoU in both directions with pointwise CIs above zero. The entire transferred pipeline does not establish a Source-relative joint-quality gain: HC Full−Source vCI crosses zero and Vid mean vIoU declines with CI crossing zero. Both domains have negative unobserved spatial quality with CIs below zero. The spatial/shared initialization transferred unchanged from TA therefore remains a limitation on this backbone; this comparison does not isolate a unique cause of the losses. No retuning follows and no universal backbone-generalization claim is supported.

These are pointwise parent-cluster CIs without familywise correction. No query removed for a poor outcome. Screening positives and confirmation negatives are both public; previously exposed fullsets are not called unseen. TA production and Fig1 remain unchanged.

Cost and execution:256actual Source forwards+256sameReferDINO joint8 acquisitions for confirmation,0teacher cache hits. Mean measured Source capture HC1.830947s/Vid1.480970s; per-query pipeline HC3.940683s/Vid3.394432s, excluding video decoding, subject parsing and some initialization. Mean original expert GPU-forward HC.176833s/Vid.163708s, feedback HC.497967s/Vid.491884s. Nested times cannot be added. Source model loading was not separately timed in the frozen loader and is unavailable, not zero. Complete historical replay/failure/unique-expert accounting is in COST_ACCOUNTING.json.

All256prediction/capture/receipt SHAs and public means were checked by postseal reporting (768hashchecks), no new model/expert or GT scoring. All owned controller/predictor/scorer/teacher processes exited and shared GPU lock availability was verified.
