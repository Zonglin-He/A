# Frozen spatial OPD + Strong: two directions, 128 parents each

Five rows, same physical decision support/MBR, frozen spatial configuration and seed41 (latest user revision; completed42/43 excluded from primary results). One prelocked query per parent. HC exact prior development128 outputs reused after full-input compatibility audit; reverse128 freshly inferred. This is a panel evaluation, not the full validation split or independent confirmation. No parameter selection. Strong posterior = 0.5 high-resolution448 final + 0.5 native224 second-pass six-layer posterior mean. Spatial adaptation and temporal evidence enhancement have separate mechanisms.

| Direction / row | full vIoU % | tIoU % | fixed actual-node sIoU % |
|---|---:|---:|---:|
| VidSTG_to_HC2 / Source | 25.463541 | 51.439514 | 47.979496 |
| VidSTG_to_HC2 / Spatial_only | 27.897703 | 51.439514 | 52.168591 |
| VidSTG_to_HC2 / Temporal_only | 26.969901 | 55.920522 | 47.979496 |
| VidSTG_to_HC2 / Full | 29.488137 | 55.920522 | 52.168591 |
| VidSTG_to_HC2 / Without_spatial_adaptation | 28.700393 | 55.920522 | 50.510433 |

## VidSTG_to_HC2 paired differences

| Comparison | full-v pp [95% paired parent CI] | tIoU pp | severe losses >5 pp |
|---|---|---:|---:|
| Spatial_only_minus_Source | +2.434162 [+0.485622,+4.272600] | +0.000000 | 13 |
| Temporal_only_minus_Source | +1.506359 [+0.518667,+2.522488] | +4.481008 | 11 |
| Full_minus_Source | +4.024596 [+1.735524,+6.221780] | +4.481008 | 19 |
| Full_minus_Without_spatial_adaptation | +0.787745 [+0.209846,+1.486938] | +0.000000 | 1 |
| Full_minus_Spatial_only | +1.590434 [+0.576657,+2.625446] | +4.481008 | 12 |
| Without_spatial_adaptation_minus_Temporal_only | +1.730492 [-0.274205,+3.587834] | +0.000000 | 13 |

Adaptation contribution relative to E+Strong: +0.787745 pp, CI [0.20984603715697087, 1.4869376273541022]. Whole-method improvement over Source cannot by itself establish OPD contribution.
| HC2_to_VidSTG / Source | 14.582710 | 37.130256 | 36.381530 |
| HC2_to_VidSTG / Spatial_only | 17.912179 | 37.130256 | 42.169709 |
| HC2_to_VidSTG / Temporal_only | 16.481081 | 40.790336 | 36.381530 |
| HC2_to_VidSTG / Full | 19.904380 | 40.790336 | 42.169709 |
| HC2_to_VidSTG / Without_spatial_adaptation | 20.182290 | 40.790336 | 42.622795 |

## HC2_to_VidSTG paired differences

| Comparison | full-v pp [95% paired parent CI] | tIoU pp | severe losses >5 pp |
|---|---|---:|---:|
| Spatial_only_minus_Source | +3.329469 [+1.253405,+5.520200] | +0.000000 | 11 |
| Temporal_only_minus_Source | +1.898371 [+0.886259,+3.099035] | +3.660080 | 2 |
| Full_minus_Source | +5.321670 [+2.949752,+7.813405] | +3.660080 | 14 |
| Full_minus_Without_spatial_adaptation | -0.277910 [-1.150458,+0.431887] | +0.000000 | 3 |
| Full_minus_Spatial_only | +1.992201 [+0.739245,+3.496635] | +3.660080 | 6 |
| Without_spatial_adaptation_minus_Temporal_only | +3.701209 [+1.735982,+5.719657] | +0.000000 | 9 |

Adaptation contribution relative to E+Strong: -0.277910 pp, CI [-1.150458401908777, 0.4318867487638045]. Whole-method improvement over Source cannot by itself establish OPD contribution.

## Cost and scope

All per-query measured cost components and peak GPU are in RESULTS_PUBLIC.json. Common Source224/448, GroundingDINO and ReferDINO are charged, including model loading separately. The original three-seed and revised single-seed matrix compute are separately disclosed; do not pool them as repeated independent parents. Single-seed component sums include each expert/common stage once; they are component sums, not independent end-to-end wall measurements. HC cache-reuse rows do not claim zero deployment cost. HC RC001 complete-input compatibility case provides a measured complete-chain profile; reverse cases run the complete chain. No weights, raw caches, private video/annotations, captions or parent identifiers are published. Historical exposure is disclosed; no methods registry is automatically promoted.

## Root saved-score verification and decision scope

All256 parents retained. 1280 official/dense metric agreement checks. Independent NumPy audits256 selected seed41 fits/10240 updates; earlier unselected42/43 are historical compute. Actual matrix new updates 8280; exact reused updates 15240. Additional single-seed cost profiles80 updates are separately counted. No candidate selection by new GT and no registry promotion.

hc2: Full−E+Strong +0.787745 pp, CI [0.20984603715697087, 1.4869376273541022]; panel_supports_spatial_adaptation_contribution; needs independent confirmation.

vidstg: Full−E+Strong -0.277910 pp, CI [-1.150458401908777, 0.4318867487638045]; spatial_adaptation_contribution_not_established_on_this_panel; total gain must not be attributed entirely to_OPD.

## Measured actual single-inference cost

| Target | complete cold single seed wall seconds | Source cold load seconds | GroundingDINO forwards | ReferDINO frames | peak allocated GPU GiB |
|---|---:|---:|---:|---:|---:|
| hc2, RC001, seed41 | 25.209731 | 5.009152 | 4 | 8 | 7.734084 |
| vidstg, BV001, seed41 | 27.041526 | 3.513889 | 4 | 8 | 20.202834 |

Actual full single-seed wall profiles include original video decoding, text parsing, Source loading,224/448 Source inference, both experts including model loads/subprocess isolation, E/MBR and40 spatial updates. They are the first prelocked case per direction, not128-case deployment averages. Peak is the maximum PyTorch allocated bytes of isolated model processes, excluding unrelated services/context-reserved memory. For reverse128, all per-case components and actual matrix wall costs (original three-seed / revised one-seed) are retained; component sums are explicitly distinguished from independently measured single-seed wall time. HC127 reused cases retain prior full compute provenance, not zero-cost deployment claims.

The fixed actual-node sIoU averages only parents with GT-supported physical input nodes; counts are recorded in ROOT_AUDIT_PUBLIC.json. All256 parents remain in full vIoU/tIoU/dense sIoU and severe-tail analysis. Both directions are frozen development/exposure-disclosed panels; HC128 directly participated in spatial/Strong development, and reverse legacy exposure prevents claiming a pristine test set.

## 中文结论（seed41，两个128父视频面板）

完整候选相比Source，HC2为+4.024596pp，VidSTG为+5.321670pp；这不是全部由OPD贡献。HC2空间适应相比E+Strong为+0.787745pp，95%配对区间[+0.209846,+1.486938]；VidSTG为−0.277910pp，区间[−1.150458,+0.431887]。本面板支持HC空间适应组件的正贡献，但不支持双向空间OPD均有贡献。两者都不足以宣称OPD独立带来2–3pp，更没有用本五行证明Live相对Fixed独立有效。

HC2留一父均值范围[+0.601594,+0.862231]pp，正增量并非单个父视频独撑。VidSTG空间适应63正39负26不变，负贡献−0.733638pp超过正贡献+0.455727pp，3父下降>5pp；最坏BV041−41.641041pp。全量保留，不删除负例后报告。HC空间适应82正42负4平、1父下降>5pp。完整方法相对Source仍有19/14父下降>5pp，不能以正均值掩盖严重尾部。

HC128已用于开发，全部legacy曝光及已知跨面板重叠披露，不是独立确认。反向沿用HC冻结空间超参数和源HC2 E标定，未做Vid专门调参；阴性范围限于此冻结配置/面板。按方向保留判断，不自动拼接每域赢家并宣称新方法成绩，不自动晋升方法登记，不追加调参或确认。
