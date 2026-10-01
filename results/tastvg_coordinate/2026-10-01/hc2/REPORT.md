# hc2: learning rate then temperature

Selected parameters: {'lr': 0.006097133675874025, 'rho': 0.05, 'teacher_temperature': 1.0}. Source-disjoint confirmation is historically exposed data. No confirmation reselection.

| Arm | Corruption all ΔvIoU pp [95% CI] | Nonexpert ΔvIoU pp [95% CI] |
|---|---:|---:|
| default | +1.51701 [+0.01674, +3.48873] | +0.03125 [-0.07531, +0.13919] |
| lr_only | +1.52631 [+0.03745, +3.49332] | +0.04090 [-0.09558, +0.17584] |
| selected | +1.52631 [+0.03745, +3.49332] | +0.04090 [-0.09558, +0.17584] |

All scheduled points, numerical failures and clean/expert/nonexpert/harm outcomes are retained. A single coordinate pass is not a global optimum. No production promotion or full-data rerun. Root publication and parameter curves remain required.
