# Source-only TTFT calibration review

All72 source-validation queries/576 adapted states and all720 official/manual scalar checks passed. No target scores selected these configurations. The validation videos were historically exposed in the reverse transfer direction; HC source queries have34 scorer parents, not36. These are calibration outcomes and selected-setting CIs are descriptive, not independent validation.

## Target hc2: calibrate only on source vidstg

Selected Config6: {"lr": 0.0001, "lambda_l1": 1, "lambda_giou": 1, "steps": 5, "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0, "foreach": false, "clip_norm": 0.1}
SourceTR vIoU 14.168325 -> selected 17.156247; delta 2.987923 pp, descriptive95%CI [1.1220812809656844, 5.051085087856583]. Temporal delta 2.106954, CI [-0.24303371030729523, 5.846105052463606]. All8 candidates, harms/severe/short-long and actual update traces remain in report/ROOT_diagnostics/signed files.

## Target vidstg: calibrate only on source hc2

Selected Config7: {"lr": 0.0001, "lambda_l1": 1, "lambda_giou": 1, "steps": 20, "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0, "foreach": false, "clip_norm": 0.1}
SourceTR vIoU 29.748369 -> selected 36.298899; delta 6.550530 pp, descriptive95%CI [2.647458447707878, 10.379772857327644]. Temporal delta -0.793022, CI [-1.627795875434844, 0.0021137770111128206]. All8 candidates, harms/severe/short-long and actual update traces remain in report/ROOT_diagnostics/signed files.

The learned baseline is functional in both directions. Accept the source-selected configurations for the original128-query target pilot only. Target scores cannot change lr, loss weights or steps. No shared correction, transport or DP enters the TTFT baseline. Source/acquisition/update/checkpoint costs are retained; conditional cached wall is not deployment timing.
