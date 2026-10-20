# Dataset-tuned single-ReferDINO runtime, 2026-10-20

One frozen ReferDINO supplies the same eight selected-track boxes for initialization and OPD supervision. Frozen TA-STVG uses its original source checkpoint per transfer direction; Strong time and physical MBR remain unchanged. No independent GroundingDINO model, detector stage or detector cache is needed. Network weights are frozen. Each query resets8×256 content states and Adam.

`tuned_config_20261020.json` is the unique selected configuration for this revision. `config.json` retains the previous single-expert evaluation recipe. CURRENT/RESEARCH/MAIN/PAPER registries are historical and are not promoted by this parameter search.

| Target dataset | initialization gamma | Adam lr | teacher tau | steps | final output alpha |
|---|---:|---:|---:|---:|---:|
| hc2 | 0.5 | 0.0141421356 | 0.141421356 | 40 | 1 |
| vidstg | 1 | 0.01 | 0.5 | 30 | 1 |

Gamma scales the original Source-to-consensus residual in center/log-size **before** adaptation. Each real OPD rollout uses this actual initialization. Alpha scales only the resulting deterministic adaptation residual around the same E_gamma **after** fitting. The official student output remains a decoded full trajectory, not a sampled candidate or teacher box. Both parameters operate on normalized centers and log width/height, not raw xyxy interpolation.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_single_referdino_opd_v1.tuned_gamma_cli \
  --request /absolute/path/request.json --output /absolute/path/prediction.npz
```

The JSON request schema is the original single-expert CLI: `dataset` (`hc2` or `vidstg`), deterministic sampling `key` and `metadata` containing video/text/physical frame geometry, without GT. The code decodes RGB, runs Source224 and448, performs one native ReferDINO eight-frame query, initializes E_gamma, executes the selected actual number of state updates and writes Source/E_Ref/Live_raw/Live plus physical frame IDs and both intervals. The `.cost.json` records actual capture and adaptation time. It does not import a scorer or read GT.

The class `InitializationTunedSingleReferDINOOPDPredictor` can also consume valid existing caches via `initialize_cached` and `adapt`. Initialization always reconstructs the unscaled consensus from Source and the same teacher boxes before applying gamma once; repeated calls do not compound the strength. `output_alpha` does not change sampling or pretend to be a separately run optimizer path.

The selected values were chosen on all128 exposed parents per dataset using complete vIoU subject to no increase in the count of adaptation losses worse than5pp relative to the original baseline (HC0, Vid1). Within0.1pp of the best protected complete mean, revision003 prioritizes the number of parents gaining at least0.1pp, then fewer parents losing0.1pp, then higher mean. Both mean maxima remain public. This disclosed development-selection revision follows the harmful-video analysis and user coverage preference; it is not a per-video gate. The unconstrained mean winner and all of its harmful outcomes are retained separately in the selection record. Each phase's predictions and actual updates were sealed and independently audited before GT scoring. This is development selection, not independent confirmation or a claim of global optimality. All positive and negative parameter outcomes remain in the public report, including no-adaptation E_gamma+Strong controls. Initialization improvements are separated from subsequent OPD improvements. Source, Strong, experts, observations and pretrained network weights are unchanged.

See [full report](../../results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/REPORT.md), [five-row results](../../results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/SELECTED_FIVE_ROW_RESULTS_PUBLIC.json), and [complete signed parameter outcomes](../../results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/all_parameter_outcomes/INDEX.json).
