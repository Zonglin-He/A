# Episodic Rank-RKL and functional correction scope

Both experiments completed on the fixed, historically exposed 32 development + 16 confirmation sources per dataset. 1,152 arrivals, 288 scheduled expert writes; two orders, clean plus five 5% corruptions. The unchanged same-domain A bundles are Vid K1 and HC2 K8. No new specialist inference, loss, hyperparameter search, online GT rule or method promotion.

This experiment separates complete query-wise source resets from the transfer scope of an isolated A write. Episodic_after uses the current query’s update before its output; A_before is the original predict-before-update persistent output. A_after is a matched post-update control, not the formal online output. Primary spatial attribution fixes the source-native interval for every lifecycle arm; native and before-update Fast intervals are reported separately.

## Lifecycle: spatial attribution on a common fixed interval

| Dataset / panel | Frozen vIoU % | A_before | A_after | E_after | E_after−A_after, pp [95% CI] | E_after−A_before |
|---|---:|---:|---:|---:|---|---|
| vidstg / search | 16.215 | 16.787 | 17.007 | 16.275 | -0.733 [-2.468, +0.415] | -0.513 [-2.221, +0.599] |
| vidstg / confirm | 32.953 | 33.027 | 33.386 | 33.207 | -0.178 [-0.690, +0.224] | +0.180 [-0.128, +0.495] |
| hc2 / search | 29.867 | 31.151 | 31.235 | 29.941 | -1.294 [-4.527, +0.847] | -1.211 [-4.446, +0.937] |
| hc2 / confirm | 26.478 | 26.436 | 26.591 | 26.565 | -0.025 [-0.175, +0.142] | +0.130 [-0.120, +0.438] |

A_before−Frozen describes online history; A_after−A_before describes immediate current-query execution. E_after−A_after changes only the state lifecycle at matched output timing. E_after−A_before changes both lifecycle and timing and cannot uniquely attribute an episodic advantage. Episodic nonexpert outputs are source Frozen by construction.

## Transfer scope of the same sealed A write

| Dataset / panel | Scope | Donor sources | Target cells | Full write vIoU change, pp [95% CI] |
|---|---|---:|---:|---|
| vidstg / search | self | 16 | 80 | +0.879 [+0.110, +1.784] |
| vidstg / search | same_video_other_query | 16 | 80 | +0.992 [-0.102, +2.192] |
| vidstg / search | different_video_same_corruption | 16 | 1080 | +0.298 [-0.052, +0.687] |
| vidstg / search | semantic_near | 16 | 80 | +0.173 [-0.786, +1.296] |
| vidstg / search | semantic_far | 16 | 80 | +0.613 [-0.279, +1.676] |
| vidstg / search | different_corruption | 16 | 80 | +0.165 [-0.602, +1.147] |
| vidstg / confirm | self | 8 | 40 | +1.435 [+0.215, +2.762] |
| vidstg / confirm | same_video_other_query | 8 | 40 | +0.695 [-0.341, +2.257] |
| vidstg / confirm | different_video_same_corruption | 8 | 300 | -0.042 [-0.363, +0.301] |
| vidstg / confirm | semantic_near | 8 | 40 | +0.007 [-0.617, +0.626] |
| vidstg / confirm | semantic_far | 8 | 40 | -0.694 [-1.545, -0.003] |
| vidstg / confirm | different_corruption | 8 | 40 | +0.074 [-0.216, +0.361] |
| hc2 / search | self | 14 | 80 | +0.442 [+0.033, +0.922] |
| hc2 / search | same_video_other_query | 0 | 0 | N/A |
| hc2 / search | different_video_same_corruption | 14 | 1080 | +0.273 [-0.037, +0.636] |
| hc2 / search | semantic_near | 14 | 80 | +0.148 [-0.094, +0.363] |
| hc2 / search | semantic_far | 14 | 80 | +0.211 [-0.025, +0.439] |
| hc2 / search | different_corruption | 14 | 80 | +0.029 [-0.218, +0.266] |
| hc2 / confirm | self | 7 | 40 | +0.561 [-0.697, +2.112] |
| hc2 / confirm | same_video_other_query | 0 | 0 | N/A |
| hc2 / confirm | different_video_same_corruption | 7 | 300 | +0.047 [-0.108, +0.201] |
| hc2 / confirm | semantic_near | 7 | 40 | -0.070 [-0.204, +0.057] |
| hc2 / confirm | semantic_far | 7 | 40 | +0.054 [-0.326, +0.379] |
| hc2 / confirm | different_corruption | 7 | 40 | +0.210 [+0.016, +0.400] |

The same source donor’s whole K1/K8 write is evaluated under a common donor-pre state. Targets include every eligible future nonexpert query in the original order/split/corruption, frozen text-near/far, and a fixed-hash same target tested in two corruptions. Near/far roles can alias existing targets; they are not additional independent samples. Same-video other-query uses real identical-media alternative captions and their own original sampling grids. HC2 has no such alternative for these sources and is explicitly N/A.

Target effects are averaged within donor/scope, then within donor source across condition/order; paired 10,000 donor-source bootstrap is primary. Target-source bootstrap sensitivity is saved independently. Single-write transfer does not establish long-stream retention or justify an online scope router.

## Block interventions and negative tails

| Dataset / panel | Scope | Query residual | norm1 | norm3 | norm4 | Full |
|---|---|---|---|---|---|---|
| vidstg / search | self | -0.047 [-0.504, +0.359] | +0.129 [+0.013, +0.271] | +0.413 [+0.029, +0.892] | +0.863 [+0.089, +1.795] | +0.879 [+0.110, +1.784] |
| vidstg / search | same_video_other_query | -0.123 [-0.505, +0.113] | +0.140 [+0.018, +0.306] | +0.453 [+0.064, +0.988] | +0.902 [+0.228, +1.803] | +0.992 [-0.102, +2.192] |
| vidstg / search | different_video_same_corruption | +0.244 [+0.036, +0.503] | +0.015 [+0.003, +0.029] | +0.051 [+0.011, +0.094] | +0.082 [-0.062, +0.231] | +0.298 [-0.052, +0.687] |
| vidstg / search | semantic_near | +0.643 [-0.087, +1.663] | +0.004 [-0.045, +0.056] | +0.000 [-0.168, +0.176] | +0.075 [-0.533, +0.747] | +0.173 [-0.786, +1.296] |
| vidstg / search | semantic_far | +0.032 [-0.002, +0.075] | +0.030 [-0.019, +0.082] | +0.105 [-0.049, +0.272] | +0.525 [-0.273, +1.484] | +0.613 [-0.279, +1.676] |
| vidstg / search | different_corruption | +0.244 [-0.345, +1.092] | +0.012 [+0.001, +0.024] | +0.031 [-0.002, +0.071] | -0.079 [-0.413, +0.170] | +0.165 [-0.602, +1.147] |
| vidstg / confirm | self | -0.010 [-0.096, +0.080] | +0.090 [+0.012, +0.180] | +0.315 [+0.040, +0.621] | +1.242 [+0.216, +2.382] | +1.435 [+0.215, +2.762] |
| vidstg / confirm | same_video_other_query | -0.033 [-0.227, +0.111] | +0.032 [-0.014, +0.105] | +0.095 [-0.051, +0.318] | +0.605 [-0.153, +1.793] | +0.695 [-0.341, +2.257] |
| vidstg / confirm | different_video_same_corruption | -0.014 [-0.065, +0.023] | +0.008 [-0.005, +0.024] | +0.029 [-0.024, +0.096] | -0.020 [-0.312, +0.294] | -0.042 [-0.363, +0.301] |
| vidstg / confirm | semantic_near | +0.004 [+0.001, +0.008] | +0.016 [-0.005, +0.040] | +0.079 [-0.005, +0.184] | -0.051 [-0.588, +0.485] | +0.007 [-0.617, +0.626] |
| vidstg / confirm | semantic_far | -0.023 [-0.070, +0.006] | -0.003 [-0.023, +0.017] | -0.003 [-0.082, +0.075] | -0.636 [-1.417, -0.059] | -0.694 [-1.545, -0.003] |
| vidstg / confirm | different_corruption | -0.032 [-0.098, +0.008] | +0.013 [+0.002, +0.027] | +0.047 [+0.001, +0.102] | +0.085 [-0.218, +0.391] | +0.074 [-0.216, +0.361] |
| hc2 / search | self | +0.001 [-0.151, +0.113] | +0.032 [+0.004, +0.065] | +0.096 [+0.022, +0.181] | +0.341 [+0.060, +0.697] | +0.442 [+0.033, +0.922] |
| hc2 / search | different_video_same_corruption | +0.164 [-0.103, +0.482] | +0.011 [+0.003, +0.018] | +0.037 [+0.015, +0.059] | +0.078 [+0.022, +0.131] | +0.273 [-0.037, +0.636] |
| hc2 / search | semantic_near | -0.017 [-0.178, +0.103] | +0.014 [+0.003, +0.026] | +0.040 [+0.010, +0.073] | +0.140 [+0.060, +0.225] | +0.148 [-0.094, +0.363] |
| hc2 / search | semantic_far | +0.010 [-0.004, +0.025] | +0.012 [-0.007, +0.032] | +0.044 [-0.014, +0.101] | +0.155 [-0.006, +0.313] | +0.211 [-0.025, +0.439] |
| hc2 / search | different_corruption | -0.070 [-0.245, +0.114] | +0.009 [+0.001, +0.016] | +0.034 [+0.008, +0.057] | +0.068 [-0.012, +0.152] | +0.029 [-0.218, +0.266] |
| hc2 / confirm | self | -0.247 [-0.837, +0.094] | +0.092 [-0.004, +0.240] | +0.280 [-0.002, +0.712] | +0.663 [-0.093, +1.812] | +0.561 [-0.697, +2.112] |
| hc2 / confirm | different_video_same_corruption | -0.044 [-0.077, -0.012] | +0.007 [-0.001, +0.016] | +0.028 [+0.003, +0.052] | +0.068 [-0.062, +0.204] | +0.047 [-0.108, +0.201] |
| hc2 / confirm | semantic_near | -0.040 [-0.143, +0.020] | +0.000 [-0.007, +0.007] | +0.008 [-0.008, +0.024] | -0.038 [-0.148, +0.056] | -0.070 [-0.204, +0.057] |
| hc2 / confirm | semantic_far | -0.007 [-0.034, +0.010] | +0.004 [-0.017, +0.021] | +0.020 [-0.033, +0.067] | +0.048 [-0.239, +0.303] | +0.054 [-0.326, +0.379] |
| hc2 / confirm | different_corruption | -0.013 [-0.068, +0.019] | +0.015 [+0.003, +0.027] | +0.050 [+0.015, +0.085] | +0.177 [-0.005, +0.382] | +0.210 [+0.016, +0.400] |

These finite state-replacement effects are nonadditive. A sum of individual LayerNorm effects is not the Full effect; they are not a gradient decomposition. They show which saved parameter block causes useful or harmful readout changes under this intervention.

| Dataset / confirmation scope | Improved / harmed cells | >5 / >20pp harm | Correct .3 destroyed / rescued | Correct .5 destroyed / rescued |
|---|---|---|---|---|
| vidstg / self | 25 / 12 | 0 / 0 | 0 / 0 | 0 / 0 |
| vidstg / same_video_other_query | 19 / 9 | 0 / 0 | 0 / 0 | 0 / 0 |
| vidstg / different_video_same_corruption | 145 / 149 | 0 / 0 | 1 / 0 | 1 / 1 |
| vidstg / semantic_near | 20 / 20 | 0 / 0 | 0 / 0 | 0 / 0 |
| vidstg / semantic_far | 15 / 25 | 0 / 0 | 1 / 0 | 0 / 0 |
| vidstg / different_corruption | 25 / 15 | 0 / 0 | 1 / 0 | 0 / 0 |
| hc2 / self | 24 / 16 | 0 / 0 | 0 / 3 | 0 / 1 |
| hc2 / same_video_other_query | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| hc2 / different_video_same_corruption | 148 / 137 | 0 / 0 | 0 / 0 | 0 / 0 |
| hc2 / semantic_near | 17 / 23 | 0 / 0 | 0 / 0 | 0 / 0 |
| hc2 / semantic_far | 23 / 17 | 0 / 0 | 0 / 0 | 0 / 0 |
| hc2 / different_corruption | 25 / 14 | 0 / 0 | 0 / 0 | 0 / 0 |

Gross gains/losses, clean controls, expert/nonexpert lifecycle slices, every order, native-free and source-fixed interval controls are included in SUMMARY.json and all anonymous rows. Cell-level tails are counts, not independent source counts. Each scope’s positive-gain concentration is disclosed below.

- vidstg / self: 5/8 positive donor sources; largest donor share of positive source gain 34.46%.
- vidstg / same_video_other_query: 4/8 positive donor sources; largest donor share of positive source gain 80.48%.
- vidstg / different_video_same_corruption: 3/8 positive donor sources; largest donor share of positive source gain 50.19%.
- vidstg / semantic_near: 4/8 positive donor sources; largest donor share of positive source gain 40.44%.
- vidstg / semantic_far: 3/8 positive donor sources; largest donor share of positive source gain 49.25%.
- vidstg / different_corruption: 5/8 positive donor sources; largest donor share of positive source gain 41.07%.
- hc2 / self: 4/7 positive donor sources; largest donor share of positive source gain 71.46%.
- hc2 / different_video_same_corruption: 4/7 positive donor sources; largest donor share of positive source gain 45.76%.
- hc2 / semantic_near: 3/7 positive donor sources; largest donor share of positive source gain 51.63%.
- hc2 / semantic_far: 4/7 positive donor sources; largest donor share of positive source gain 41.48%.
- hc2 / different_corruption: 6/7 positive donor sources; largest donor share of positive source gain 38.21%.

## Matched cross-corruption intervention

| Dataset / panel | Full cross-corruption−same-corruption transfer, pp [95% CI] |
|---|---|
| vidstg / search | -0.130 [-0.335, +0.023] |
| vidstg / confirm | +0.058 [+0.003, +0.125] |
| hc2 / search | +0.029 [-0.021, +0.100] |
| hc2 / confirm | +0.117 [-0.014, +0.347] |

A donor write is applied to the exact same target caption/media in donor corruption and the next prelocked corruption. This controls target identity; it does not prove natural temporal continuity between adjacent independent videos. No text-similarity threshold, learned router or memory is derived from GT.

## Actual execution, limitations and decision scope

Actual unique target cells: 4032; donor/scope aggregate rows: 1584. Vid alternative-query frozen captures: 144; new specialist calls: 0. RESOURCES.json records suffix replays and worker wall time, not pure GPU kernel latency.

All new episode and matrix predictions seal globally before diagnostic labels. Root audit checks full source resets, refreshed K-step candidates, original donor state parity, exact block masks, frozen text target selection, Rank-RKL/SGD arithmetic and every dense readout. Public scalar audit independently reconstructs paired source bootstrap, tails and donor balancing. Full network Jacobians are not recomputed on CPU.

The results qualify local execution and isolated transfer scopes. They do not automatically authorize a new state router, a universal episodic replacement, dataset-specific rules or deployment changes.

The confirmation writer omitted direct post-update predictions. A_after there uses the sealed matched rank_native replay’s final-step raw output; original pre/post parameter states, pre-output, and search post-output boxes/intervals are independently identical. The replay’s top-level Fast-fixed interval is explicitly excluded from the native-free control. Nonexpert no-op uses the unchanged slow output. Schema/type recovery preserved the failed CPU attempts and changed no GPU inference or scientific setting.

On both confirmation panels, complete episodic reset did not establish an advantage over the matched persistent A_after control: both paired intervals cross zero. The isolated Full write improves Vid self readout, but confirmation transfer to other videos under the same corruption is inconclusive on both datasets. The Vid text-far contrast is negative in this small diagnostic panel; it is not a validated deployment router or a universal rule about semantic distance. Parameter-block effects are exploratory matched interventions, and many scope contrasts have not been adjusted for multiple testing.
