# Final short-decimal single-ReferDINO recipe

The final tested configuration uses one frozen ReferDINO for the same eight-box initialization and OPD supervision, frozen Source/Strong, physical MBR, per-query reset8×256 content states and seed41. The user requested shorter learning-rate and temperature numbers. Exactly one full HC128 trial verified lr0.014 and tau0.14; no precision grid was searched. Vid0.01/0.5 is unchanged and its exact saved predictions are reused.

| Dataset | learning rate | temperature | steps | initialization gamma | output alpha |
|---|---:|---:|---:|---:|---:|
| hc2 | 0.014 | 0.14 | 40 | 0.5 | 1.0 |
| vidstg | 0.01 | 0.5 | 30 | 1.0 | 1.0 |

Use `rounded_config_20261020.json` and the matching executable:

```bash
.conda/tubedetr/bin/python -B -m methods.stvg_single_referdino_opd_v1.rounded_cli \
  --request /absolute/path/request.json --output /absolute/path/prediction.npz
```

The request schema, frozen weights and exact pipeline are the same as [the detailed runtime guide](TUNED_README_20261020.md). The older `tuned_config_20261020.json` and `tuned_gamma_cli` preserve the pre-rounding selected recipe; their results were not relabeled. The final entry uses the explicit shorter-decimal configuration above.

[Final five-row evaluation](../../results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/rounding/FINAL_FIVE_ROW_RESULTS_PUBLIC.json), [rounding report](../../results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/rounding/REPORT.md), and [retained full search](../../results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/REPORT.md) contain all signed outcomes and costs. These are exposed development panels, not full-dataset confirmation. Vid coverage preference retains BV030's serious loss; the alternative mean-first configuration is disclosed. CURRENT/MAIN/PAPER registries are not promoted.
