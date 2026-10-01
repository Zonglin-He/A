# vidstg: learning rate then temperature

Selected parameters: {'lr': 0.033761698432507946, 'rho': 0.05, 'teacher_temperature': 0.34902548789596055}. Source-disjoint confirmation is historically exposed data. No confirmation reselection.

| Arm | Corruption all ΔvIoU pp [95% CI] | Nonexpert ΔvIoU pp [95% CI] |
|---|---:|---:|
| default | -1.39911 [-3.90616, +0.20026] | +0.05092 [-0.00330, +0.11181] |
| lr_only | -1.18660 [-3.78095, +0.53205] | +0.26771 [-0.07965, +0.65600] |
| selected | -0.91493 [-3.80692, +1.16787] | +0.53318 [-0.42459, +1.60665] |

All scheduled points, numerical failures and clean/expert/nonexpert/harm outcomes are retained. A single coordinate pass is not a global optimum. No production promotion or full-data rerun. Root publication and parameter curves remain required.
