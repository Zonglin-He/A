# Paper48 P0: matched Raw-RKL component ablation

16 exposed VidSTG sources, five locked orders, clean and five 5% conditions. Frozen Final unchanged. Each row is mean ± sample SD over the five shared-source orders.

| Arm | Whole Δv pp | Future Δs pp | Future Δv pp | Future arm−Final pp |
|---|---:|---:|---:|---:|
| Frozen | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | -0.025505 ± 0.020209 |
| Fast-only | +1.496239 ± 0.863637 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | -0.025505 ± 0.020209 |
| Slow-only | +0.021456 ± 0.015196 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +0.000000 ± 0.000000 |
| Final | +1.517681 ± 0.859943 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +0.000000 ± 0.000000 |
| raw_rkl | +1.496393 ± 0.863629 | +0.000614 ± 0.000127 | +0.000180 ± 0.000042 | -0.025324 ± 0.020191 |

Raw-RKL reproduces the original S1 raw-IoU softmax teacher (temperature 1). Pairwise was cancelled by the user and was not run. Raw-RKL keeps SGD .005 and all other frozen settings. Loss scale was not tuned or norm matched; conclusions apply to this configuration. Clean, every order, harm counts and source effects are retained in the companion JSON. On-policy superiority remains unsupported by A1.
