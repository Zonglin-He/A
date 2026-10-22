# HC64: dense boundary feedback with coarse posterior preservation

**Executed on all64; not admitted to the main method.** The primary has small positive means versus Global6, but paired CIs span zero, long-event tIoU falls, and the protection-only old-token control has higher overall tIoU. Full remains paused at HC1902 sealed/scored and Vid1373 saved.

## Failure addressed and fixed design

The previous predictive event feedback cannot distinguish boundaries sharing the same eight-anchor membership. In26 harmed Direct cases,20 contain a better-than-native interval with exactly the selected membership. Increasing query optimization would fit that ambiguous supervision more fully. This experiment instead tests dense changes in the actual text-conditioned temporal hidden already saved at the two native224 offsets. No new query fitting, RGB decoding, encoder/decoder/head execution or expert call occurs.

For each offset and each of the six layers, normalize hidden vectors and take two frames on either side of a potential cut. The contrast is

\[d(t)=\tfrac12\|\mu_{\mathrm{left}}(t)-\mu_{\mathrm{right}}(t)\|_2^2.\]

Average layers, convert to bounded centered percentile ranks, interpolate in physical frame time, then average offsets. Never interleave offsets before computing changes. A candidate receives its start plus end change score. Unknown clip-edge context is neutral. Window2 and unit reward power are fixed before predictions; no scale/weight sweep.

The primary preserves native probability of every eight-anchor-membership class C:

\[q_B(I)=q_0(C(I))\frac{q_0(I)\exp b(I)}{\sum_{J\in C(I)}q_0(J)\exp b(J)}.\]

This preserves class probabilities, not the eventual MBR decision or its membership. Original physical MBR and interval-conditioned complete-box DP remain unchanged. Also report the fixed NativeG6 spatial trajectory to isolate temporal effects.

This borrows the local self-similarity principle of [UBoCo, CVPR2022](https://arxiv.org/abs/2111.14799). Its [released encoder](https://github.com/musicalOffering/uboco/blob/main/network.py) is trained to form useful similarity structure; that supporting condition is not reproduced here. Generic changes and text-conditioned hidden changes are not guaranteed to be the described event boundaries. [DSE-VTG](https://arxiv.org/html/2609.08850v1) combines aligned video-language features, multiple temporal scales and query adaptation; its query optimization does not justify fitting unreliable feedback more strongly in our different interface.

## Complete fixed comparison

| Arm | Complete vIoU % | tIoU % | Unobserved sIoU % | Boundary MAE s |
|---|---:|---:|---:|---:|
| NativeG6 | 29.6952 | 55.9613 | 50.5762 | 2.6034 |
| BoundaryCell | 29.7731 | 56.0430 | 50.7352 | 2.5905 |
| BoundaryGlobal | 29.6899 | 55.6851 | 50.7639 | 2.6048 |
| ShiftCell | 29.5887 | 55.6628 | 50.6505 | 2.6294 |
| RawCell | 29.8312 | 56.2727 | 50.5535 | 2.6013 |
| Source | 26.8938 | 54.8230 | 47.6990 | 2.6821 |

NativeG6 is bitwise identical to the saved baseline. BoundaryCell is the preregistered primary. BoundaryGlobal removes coarse-mass preservation. ShiftCell retains the same values and budget but cyclically shifts them by one quarter of the timeline. RawCell applies identical mass preservation to the old raw event contrast; it tests whether conservatism alone explains recovery. Source is the original frozen backbone.

| Primary paired difference | Mean pp | Parent-cluster95% CI pp |
|---|---:|---|
| BoundaryCell-NativeG6, v | +0.0779 | [-0.2517, +0.4175] |
| BoundaryCell-NativeG6, t | +0.0817 | [-0.5866, +0.6805] |
| BoundaryCell-ShiftCell, v | +0.1844 | [-0.2567, +0.6672] |
| BoundaryCell-ShiftCell, t | +0.3802 | [-0.3192, +1.1358] |
| BoundaryCell-RawCell, v | -0.0581 | [-0.4895, +0.4071] |
| BoundaryCell-RawCell, t | -0.2297 | [-1.0512, +0.5879] |
| BoundaryCell-Source, v | +2.8793 | [+0.8760, +4.9465] |
| BoundaryCell-Source, t | +1.2200 | [-0.5374, +2.9661] |

One query per parent;10,000 fixed-seed paired parent bootstrap draws. These are descriptive intervals on previously exposed development, not independent confirmation. No arm is selected for full inference from this table.

## Does the new feedback actually distinguish boundaries?

It varies in 2282/2283 multi-candidate anchor classes, and changes37/64 final intervals. Native class mass is preserved to 6.66e-16. Thus the original quantization ambiguity is measurably reduced; discrimination alone is not correct ranking.

Offline posterior-expected tIoU increases +.1956pp versus native (CI[+.0566,+.3375]) and +.2671pp versus the shifted signal (CI[+.0513,+.5039]). This indicates useful local information in this development panel. But final MBR tIoU gains only+.0817pp and its CI crosses zero. The new signal is neither sufficiently reliable nor clearly superior to RawCell. Do not call expectation gains final localization gains.

| GT-duration group, diagnostic only | Queries | Primary Δt vs G6 pp | RawCell Δt vs G6 pp |
|---|---:|---:|---:|
| short_le5s | 13 | +1.1898 | -0.5572 |
| long_gt5s | 51 | -0.2008 | +0.5328 |

GT duration never selects a branch or parameter. Complementary group averages do not authorize a GT-conditioned mixture.

## Concrete corrections and failures

| Query | Δt vs G6 pp | Δv vs G6 pp | Start MAE change, frames | End MAE change, frames |
|---|---:|---:|---:|---:|
| RC009 | +6.4099 | +4.4959 | +0 | -8 |
| RC016 | -6.2500 | -4.3619 | +0 | +8 |
| RC025 | +4.0541 | +3.0976 | -15 | +0 |
| RC029 | +0.6364 | +0.4372 | +0 | -9 |
| RC030 | +4.3599 | +2.5742 | +0 | -7 |
| RC039 | +6.1728 | +4.1569 | -15 | +0 |
| RC042 | -11.7188 | -3.1914 | +8 | +7 |

RC025/RC029 were harmed by the previous raw-query fit and are improved here without query optimization. RC009 improves by moving the end closer. RC042 contracts both ends toward stronger feature-change evidence but increases both boundary errors; RC016 similarly stops too early. These are measured feature/annotation misalignments, not a visual assertion about the exact action semantics.

Across all17 temporal harms,10 intervals shorten and7 lengthen;13 move to a higher boundary score. The issue is not only systematic shrinking. The current change score has no learned start-versus-end or specific-event criterion, and separate endpoint peaks can be internal changes. Rank normalization also does not certify evidential strength. The original time posterior anchors semantics only imperfectly.

Primary complete vIoU improves17/declines15/ties32, with0 losses>5pp. tIoU improves19/declines17/ties28, with2 losses>5pp. All arms and Source retain signed rows and severe declines. RawCell has a larger overall temporal mean but still harms short events; it is not silently selected as the new method.

## Cost, validation and disposition

Cached CPU feedback averages 2.617ms/query (p95 2.807ms). All64 five-arm readouts/spatial paths/files took 2.505s total. This excludes creation/loading of the backbone cache and is not end-to-end deployment acceleration.

All64 predictions seal before new labels.768 independent official/DenseTube checks agree to6.66e-16. Native64 interval/box controls are bitwise exact, all48 paper runtime and434 provider files match their locked hashes. The scorer first failed only at final NumPy integer JSON serialization; original logs/code retained, an independent CPU serializer adapter completes the same scoring without changing any prediction, plan or fingerprint.

**Decision:** this bounded candidate adds real dense boundary information and produces actual corrections, but has not established a reliable net boundary improvement. Do not register it, increase query steps, scan weights, or resume full. The unresolved mechanism is semantic validation and joint evaluation of the two event boundaries, not missing optimizer iterations. This result does not rule out trained/appropriately aligned boundary features or all cache-based temporal feedback.

Reproduce: `python -B research/boundary_feedback_v1/test_core.py`; `run.py prepare`, `run.py predict`, then independent CPU `score_json_adapter.py` after the seal. Assets and prior receipt-bound captures are required. Public files contain no raw captions, media, GT coordinates, weights or features.

[Protocol](protocol.json) · [All results](report.json) · [Signed rows](signed.tsv) · [Severe declines](severe_declines.tsv) · [Failure analysis](failure_analysis.json) · [Verification](verification.json)
