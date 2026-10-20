# STVG: final short-decimal single ReferDINO OPD + Strong

Final tested research recipe: [runtime guide](methods/stvg_single_referdino_opd_v1/ROUNDED_README_20261020.md), [configuration](methods/stvg_single_referdino_opd_v1/rounded_config_20261020.json), [final report](results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/rounding/REPORT.md), [five-row results](results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/rounding/FINAL_FIVE_ROW_RESULTS_PUBLIC.json).

HC uses learning rate0.014, temperature0.14,40 updates, initialization gamma0.5 and output alpha1. Vid uses0.01,0.5,30 updates,gamma1 and alpha1. One frozen ReferDINO supplies the same eight observations for initialization and supervision. TA-STVG weights and Strong time remain fixed. Each dataset uses a single fixed configuration on all128 exposed development parents, seed41; these are not independent full-dataset results. Total method gains, initialization gains, OPD increments and remaining harmful cases are reported separately.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_single_referdino_opd_v1.rounded_cli \
  --request /absolute/path/request.json --output /absolute/path/prediction.npz
```

## Pre-rounding development selection and previous recipes (retained)

# STVG: dataset-tuned single ReferDINO spatial OPD + Strong time

Current selected research recipe: [runtime guide](methods/stvg_single_referdino_opd_v1/TUNED_README_20261020.md), [dataset configuration](methods/stvg_single_referdino_opd_v1/tuned_config_20261020.json), [joint parameter report](results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/REPORT.md), [complete five-row results](results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/SELECTED_FIVE_ROW_RESULTS_PUBLIC.json).

One frozen ReferDINO supplies the same eight boxes to initialization and OPD. Frozen TA-STVG provides Source and Strong time; no independent detector is loaded. The finite HC128/Vid128 search selects one configuration per dataset, with seed41. The report separates initialization tuning, subsequent adaptation, total method gains, signed tails and full-input costs. These exposed development results do not establish independent generalization or a new Live-versus-Fixed advantage. Pretrained model weights and historical CURRENT/RESEARCH/MAIN/PAPER registries remain unchanged.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_single_referdino_opd_v1.tuned_gamma_cli \
  --request /absolute/path/request.json --output /absolute/path/prediction.npz
```

## Previous single-expert recipe and earlier evidence (historical)

# STVG: single ReferDINO initialization and spatial OPD

The user-selected research design is **`stvg_single_referdino_opd_v1`**: one frozen ReferDINO expert supplies the **same eight native selected-track boxes** for spatial initialization **E_Ref** and OPD supervision. [One-expert configuration](methods/stvg_single_referdino_opd_v1/config.json), [runtime and implementation guide](methods/stvg_single_referdino_opd_v1/README.md), [paired bidirectional protocol](protocols/stvg_single_referdino_bidirectional256_v1.md), [new evaluation](results/stvg_error_state/2026-10-20/single_referdino_bidirectional256/REPORT.md).

Video + original text → frozen TA-STVG Source → one native ReferDINO8 prediction → robust E_Ref initialization → current-student dual-scale OPD → deterministic full trajectory. Strong time uses only TA-STVG224/448 and its six temporal layers, followed by the original physical MBR. **No independent GroundingDINO loading, four-frame detection stage or detector observation cache is required.** ReferDINO retains its official internal architecture; its complete checkpoint covers all learned parameters without a separate detector-checkpoint preload.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_single_referdino_opd_v1.cli \
  --request /absolute/path/request.json --output /absolute/path/prediction.npz
```

HC uses40 steps and Vid30; lr0.01/tau0.2/scales(0.05,0.25)/32 candidates/seed41 remain fixed. Each query resets eight256-dimensional temporary content states and Adam. No pretrained network is trained. The inherited source-domain robust geometric calibration is disclosed; it was not recalibrated using target outcomes. Vid30 is the completed **old-initialization** development-grid winner, not a result of the interrupted lr/tau search or an independent optimum.

The original exposed HC128 and Vid128 panels are reevaluated in five matched rows: Source, spatial only, temporal only, complete method, and E_Ref+Strong without spatial adaptation. New one-expert results are separate from historical two-expert scores. Initialization gains and subsequent adaptation gains must be assessed separately; this experiment does not independently establish Live over Fixed.

## Historical version identities

The pre-existing CURRENT/RESEARCH/MAIN/PAPER registries retain their archived versions and are not automatically promoted by this design replacement. The new design is defined by the explicit package/configuration above. Historical results and failed configurations retain their original paths. The previous research method used **GroundingDINO-T plus ReferDINO**; it is not the one-expert method and its scores cannot be relabeled.

## Previous two-expert research entry (historical)

# STVG: frozen ReferDINO-supervised spatial OPD research version

Previous two-expert research entry: **`stvg_referdino_opd_v1`**, HC primary. [Authoritative research manifest](methods/RESEARCH_METHOD.json), [single frozen configuration](methods/stvg_referdino_opd_v1/config.json), [method and execution guide](methods/stvg_referdino_opd_v1/README.md), [final results and full costs](results/stvg_error_state/2026-10-19/referdino_final/REPORT.md).

Video + text → frozen TA-STVG Source → four GroundingDINO-T observations and E initialization with Source finalMBR → native ReferDINO eight-frame evidence → dual-scale output-box OPD → full student trajectory. Both expert stages are real dependencies and are charged. E retains the original budget of at most four planned-position calls, including the existing skip/rejection behavior; actual counts are reported and all queries remain in evaluation. All pretrained weights stay frozen. Each query independently resets **8×256=2048 temporary content states** and Adam; no online LayerNorm carry.

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_referdino_opd_v1.run \
  --input request.json --output prediction.pt --seed 41
```

Input metadata specifies video path, caption, original width/height and physical frame IDs. The [guide](methods/stvg_referdino_opd_v1/README.md) documents input validation, optional cache acceleration and separate sealed-prediction scoring. Runtime weights, local model packages and exact dependency hashes are listed in the [provenance record](results/stvg_error_state/2026-10-19/referdino_final/PROVENANCE_PUBLIC.json). Private weights, media, labels, captions and raw caches are excluded from this repository.

The final round compares only temperatures .05/.10/.20 on exposed HC128 with seeds41/42/43; the selected recipe is evaluated on32 prelocked, disjoint parent videos. Earlier Source/old-method exposure remains disclosed. Configuration freeze is not a claim of a stable2–3pp OPD gain. See the report for signed results, confidence intervals, tails, Direct/fresh Fixed/Live comparisons and total spatial versus incremental OPD effects. Three seeds are experimental repeats; default inference is one fixed seed, not an ensemble.

The frozen RF32 comparison did not reproduce the development gain: Live−E full vIoU **−0.8672pp** (95%CI[−3.2597,+0.7790]); Live−Direct **−0.2719pp**, Live−freshFixed **−0.4222pp**, with intervals crossing zero. The configuration stays frozen; this is not a verified replacement for E or a stable2–3pp OPD gain. Severe cases remain in the published mean.

## Version identities

- The archived `methods/RESEARCH_METHOD.json` records the previous two-expert research definition; its references are preserved for provenance.
- [CURRENT_METHOD.json](methods/CURRENT_METHOD.json) preserves the historical deployed P7-A algorithm and is explicitly marked historical deployment. It is not the current paper recipe.
- [Historical P7-A](methods/stvg_opd_p7a_v1/README.md) uses1792 query/LN parameters and online fractional LN carry; the new method does not.
- Historical T2+B, P7-A, protocols, saved predictions and negative results retain their original paths. Old paused queues are not resumed.
