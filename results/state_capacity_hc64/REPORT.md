# HC64 state_capacity_hc64

Counterfactual decoder interventions on the exact saved-input replay of original geometric adaptation. Full is the original geometric state; FullR4 and PCA4 variants are state projections, not newly optimized mask-reward methods. Same64 parents and fixed Strong intervals.

| Method | vIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|
| ERefStrong | 31.0918 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 53.1654 | 51.2469 |
| Full | 31.3868 | 53.1654 | 51.2469 |
| FullR4 | 31.1539 | 52.7142 | 50.8975 |
| PCA4 | 31.0910 | 52.2730 | 50.7119 |
| PCA4R4 | 31.0910 | 52.2730 | 50.7119 |

Original geometric states were not saved by the old worker. Exact cached-decoder replay recovers them only after all64 final trajectories match the saved geometric baseline bitwise. Across64, content L2 median is13.9068 (range4.1317–18.0518), and PCA4 retained state energy median is2.3466% (range1.0637–5.6278%). Every original state exceeds norm4.

The R4 intervention radially projects to4*(1-1e-6), within the feasible open ball of the old smooth map; PCA4 is the orthogonal state projection. These are nearest-state transfer tests, not a search for the best output inside each constrained space. They demonstrate failure to preserve the original geometric corrections, not mathematical impossibility of every constrained solution. All predictions were sealed before this offline scoring.
