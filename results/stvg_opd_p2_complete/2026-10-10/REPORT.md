# P2: fixed OPD paper stage, actual postseal results

All locked deployment arms and both directions sealed before this phase scoring. Intervals are 10000 paired parent bootstrap resamples after averaging orders within query and queries within parent. All cohorts have historical exposure; no hyperparameters or main budget are selected here.

| Stage / condition | Arm | parent m_vIoU (%) | ΔvIoU (pp), 95% CI | Current / inherited (pp) | Parents harm >5 / >20 pp |
|---|---|---:|---|---|---|
| P2_hc2_cross_clean / clean | direct_L1_GIoU | 22.9258 | +0.7698 [-0.7821, +2.1866] | -0.7363 / +1.5061 | 17 / 3 |
| P2_hc2_cross_clean / clean | shuffled_feedback | 19.6650 | -2.4910 [-3.5271, -1.4702] | -2.5336 / +0.0426 | 31 / 2 |
| P2_hc2_cross_clean / clean | frozen_rollout | 22.3778 | +0.2218 [-0.0047, +0.4526] | +0.0803 / +0.1415 | 2 / 0 |
| P2_hc2_cross_clean / clean | on_policy | 23.5020 | +1.3461 [-0.0460, +2.5255] | -0.4301 / +1.7761 | 11 / 2 |
| P2_hc2_same_5percent / frame_drop_5 | direct_L1_GIoU | 32.4914 | +3.7870 [+1.6589, +5.8658] | +0.0411 / +3.7459 | 14 / 4 |
| P2_hc2_same_5percent / frame_drop_5 | shuffled_feedback | 26.5758 | -2.1286 [-3.3214, -0.9843] | -2.4619 / +0.3333 | 35 / 1 |
| P2_hc2_same_5percent / frame_drop_5 | frozen_rollout | 29.3550 | +0.6506 [+0.4320, +0.9060] | +0.3447 / +0.3059 | 0 / 0 |
| P2_hc2_same_5percent / frame_drop_5 | on_policy | 32.8238 | +4.1194 [+2.6353, +5.6122] | +0.2247 / +3.8947 | 11 / 3 |
| P2_hc2_same_5percent / frame_freeze_5 | direct_L1_GIoU | 32.6486 | +3.3300 [+1.1124, +5.4560] | -0.5878 / +3.9178 | 13 / 7 |
| P2_hc2_same_5percent / frame_freeze_5 | shuffled_feedback | 26.7264 | -2.5922 [-3.6362, -1.5856] | -3.8933 / +1.3011 | 38 / 3 |
| P2_hc2_same_5percent / frame_freeze_5 | frozen_rollout | 29.6919 | +0.3733 [+0.0645, +0.6996] | +0.1429 / +0.2305 | 2 / 0 |
| P2_hc2_same_5percent / frame_freeze_5 | on_policy | 33.2199 | +3.9013 [+2.3441, +5.3095] | -0.0048 / +3.9061 | 8 / 3 |
| P2_hc2_same_5percent / motion_blur_5 | direct_L1_GIoU | 33.0774 | +3.7118 [+1.6473, +5.7677] | -0.0330 / +3.7448 | 15 / 4 |
| P2_hc2_same_5percent / motion_blur_5 | shuffled_feedback | 26.0619 | -3.3037 [-4.5423, -2.0696] | -2.6334 / -0.6703 | 41 / 3 |
| P2_hc2_same_5percent / motion_blur_5 | frozen_rollout | 30.0776 | +0.7120 [+0.4475, +1.0214] | +0.4118 / +0.3001 | 0 / 0 |
| P2_hc2_same_5percent / motion_blur_5 | on_policy | 33.4182 | +4.0526 [+2.5421, +5.4930] | +0.1677 / +3.8849 | 7 / 4 |
| P2_hc2_same_5percent / occlusion_5 | direct_L1_GIoU | 31.7603 | +2.9660 [+0.8850, +4.9490] | -0.9718 / +3.9378 | 15 / 5 |
| P2_hc2_same_5percent / occlusion_5 | shuffled_feedback | 26.5827 | -2.2115 [-3.4838, -0.9749] | -2.9578 / +0.7463 | 32 / 5 |
| P2_hc2_same_5percent / occlusion_5 | frozen_rollout | 29.3125 | +0.5183 [+0.2484, +0.8205] | +0.2945 / +0.2238 | 0 / 0 |
| P2_hc2_same_5percent / occlusion_5 | on_policy | 32.3715 | +3.5772 [+2.0167, +5.0709] | -0.2116 / +3.7888 | 13 / 4 |
| P2_hc2_same_5percent / exposure_5 | direct_L1_GIoU | 32.1255 | +2.8407 [+0.5010, +5.0408] | -1.1646 / +4.0053 | 15 / 8 |
| P2_hc2_same_5percent / exposure_5 | shuffled_feedback | 27.5754 | -1.7094 [-2.7181, -0.7011] | -3.4073 / +1.6979 | 31 / 1 |
| P2_hc2_same_5percent / exposure_5 | frozen_rollout | 29.8961 | +0.6113 [+0.3480, +0.9253] | +0.3061 / +0.3051 | 0 / 0 |
| P2_hc2_same_5percent / exposure_5 | on_policy | 33.2869 | +4.0021 [+2.3283, +5.6119] | +0.0514 / +3.9506 | 11 / 4 |
| P2_vidstg_cross_clean / clean | direct_L1_GIoU | 18.5498 | +4.0839 [+2.5085, +5.6644] | +2.3538 / +1.7301 | 7 / 1 |
| P2_vidstg_cross_clean / clean | shuffled_feedback | 9.5165 | -4.9494 [-6.3095, -3.6919] | -2.4186 / -2.5308 | 42 / 8 |
| P2_vidstg_cross_clean / clean | frozen_rollout | 14.4769 | +0.0110 [-0.4115, +0.4426] | +0.1270 / -0.1160 | 4 / 0 |
| P2_vidstg_cross_clean / clean | on_policy | 18.3366 | +3.8706 [+2.4299, +5.2901] | +2.0370 / +1.8336 | 6 / 1 |
| P2_vidstg_same_5percent / frame_drop_5 | direct_L1_GIoU | 22.6912 | +1.0746 [-1.5068, +3.4764] | +0.4185 / +0.6561 | 17 / 7 |
| P2_vidstg_same_5percent / frame_drop_5 | shuffled_feedback | 15.0162 | -6.6005 [-8.6613, -4.6478] | -3.5754 / -3.0251 | 47 / 17 |
| P2_vidstg_same_5percent / frame_drop_5 | frozen_rollout | 21.5462 | -0.0704 [-0.9167, +0.7683] | +0.1402 / -0.2106 | 9 / 1 |
| P2_vidstg_same_5percent / frame_drop_5 | on_policy | 23.5810 | +1.9644 [-0.3022, +4.1185] | +0.3029 / +1.6616 | 14 / 6 |
| P2_vidstg_same_5percent / frame_freeze_5 | direct_L1_GIoU | 25.6360 | +1.3103 [-1.3331, +3.8478] | +0.2851 / +1.0252 | 17 / 10 |
| P2_vidstg_same_5percent / frame_freeze_5 | shuffled_feedback | 16.7674 | -7.5582 [-9.8676, -5.3143] | -4.9532 / -2.6050 | 52 / 21 |
| P2_vidstg_same_5percent / frame_freeze_5 | frozen_rollout | 23.9184 | -0.4073 [-1.3570, +0.4457] | +0.1899 / -0.5972 | 14 / 1 |
| P2_vidstg_same_5percent / frame_freeze_5 | on_policy | 26.6905 | +2.3648 [+0.1278, +4.5331] | +0.1656 / +2.1992 | 14 / 7 |
| P2_vidstg_same_5percent / motion_blur_5 | direct_L1_GIoU | 25.4772 | +1.4891 [-1.2069, +4.0976] | +0.1635 / +1.3256 | 18 / 8 |
| P2_vidstg_same_5percent / motion_blur_5 | shuffled_feedback | 14.7863 | -9.2018 [-11.7114, -6.8338] | -3.9754 / -5.2264 | 56 / 24 |
| P2_vidstg_same_5percent / motion_blur_5 | frozen_rollout | 23.4610 | -0.5272 [-1.5059, +0.4141] | +0.2686 / -0.7958 | 18 / 1 |
| P2_vidstg_same_5percent / motion_blur_5 | on_policy | 25.9974 | +2.0093 [-0.4390, +4.2994] | +0.1204 / +1.8888 | 15 / 6 |
| P2_vidstg_same_5percent / occlusion_5 | direct_L1_GIoU | 26.2665 | +2.4747 [-0.3618, +5.2037] | +1.5383 / +0.9364 | 16 / 8 |
| P2_vidstg_same_5percent / occlusion_5 | shuffled_feedback | 16.1370 | -7.6548 [-9.8870, -5.4756] | -2.9845 / -4.6703 | 59 / 17 |
| P2_vidstg_same_5percent / occlusion_5 | frozen_rollout | 23.3350 | -0.4568 [-1.5686, +0.6237] | +0.4004 / -0.8572 | 19 / 2 |
| P2_vidstg_same_5percent / occlusion_5 | on_policy | 26.4847 | +2.6929 [+0.2117, +5.0943] | +1.2414 / +1.4515 | 14 / 6 |
| P2_vidstg_same_5percent / exposure_5 | direct_L1_GIoU | 26.1590 | +1.9113 [-0.8715, +4.5933] | +1.0765 / +0.8348 | 17 / 8 |
| P2_vidstg_same_5percent / exposure_5 | shuffled_feedback | 15.0283 | -9.2194 [-11.6151, -6.9424] | -3.3680 / -5.8514 | 61 / 23 |
| P2_vidstg_same_5percent / exposure_5 | frozen_rollout | 23.6886 | -0.5592 [-1.5881, +0.4102] | +0.1157 / -0.6749 | 16 / 1 |
| P2_vidstg_same_5percent / exposure_5 | on_policy | 26.0907 | +1.8430 [-0.6377, +4.2439] | +0.2194 / +1.6236 | 18 / 8 |

Direct uses standard 5 L1 + 2 GIoU and identical admitted support/parameter interface/Adam and fixed last-step budget. Query-only, LN-only and alpha-zero reconstruct their own state from source. Full before/current/inherited decomposition is descriptive and does not replace these interventions. Identical Full/Shuffled/Fixed or K4 streams are reused only with the complete original input, configuration, order and saved-state chain bound by SHA256.

GT expert-quality/duration/motion/query-type strata are descriptive offline associations, never filters or online admission decisions. Corruption coverage denotes the physical time fraction, not equal pixel strength. The main Uniform4 and target configurations remain unchanged. Cached expert cost is separated from actual new DINO calls, decoder fit and CPU audit; reused measurements do not establish cold end-to-end latency.

Root must actually inspect the generated figures and publish actual anonymous code/config/results and negative findings before closing this phase.
