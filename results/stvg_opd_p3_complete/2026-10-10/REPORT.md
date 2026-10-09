# P3: fixed OPD paper stage, actual postseal results

All locked deployment arms and both directions sealed before this phase scoring. Intervals are 10000 paired parent bootstrap resamples after averaging orders within query and queries within parent. All cohorts have historical exposure; no hyperparameters or main budget are selected here.

| Stage / condition | Arm | parent m_vIoU (%) | ΔvIoU (pp), 95% CI | Current / inherited (pp) | Parents harm >5 / >20 pp |
|---|---|---:|---|---|---|
| P3_hc2_cross_clean / clean | query_only | 22.3203 | +0.1643 [-0.5774, +0.8286] | +0.1643 / +0.0000 | 5 / 2 |
| P3_hc2_cross_clean / clean | LN_only | 23.6725 | +1.5165 [+0.2392, +2.5242] | -0.2624 / +1.7789 | 8 / 1 |
| P3_hc2_cross_clean / clean | joint_alpha0 | 23.2638 | +1.1078 [-0.3028, +2.2971] | +1.1078 / +0.0000 | 11 / 2 |
| P3_hc2_cross_clean / clean | on_policy | 23.5020 | +1.3461 [-0.0460, +2.5255] | -0.4301 / +1.7761 | 11 / 2 |
| P3_hc2_same_5percent / frame_drop_5 | query_only | 28.9184 | +0.2141 [-0.2457, +0.6812] | +0.2141 / +0.0000 | 5 / 0 |
| P3_hc2_same_5percent / frame_drop_5 | LN_only | 32.7118 | +4.0074 [+2.6155, +5.3938] | +0.1454 / +3.8621 | 10 / 3 |
| P3_hc2_same_5percent / frame_drop_5 | joint_alpha0 | 32.5915 | +3.8871 [+2.3615, +5.4330] | +3.8871 / +0.0000 | 11 / 3 |
| P3_hc2_same_5percent / frame_drop_5 | on_policy | 32.8238 | +4.1194 [+2.6353, +5.6122] | +0.2247 / +3.8947 | 11 / 3 |
| P3_hc2_same_5percent / frame_freeze_5 | query_only | 29.1753 | -0.1433 [-1.1147, +0.7065] | -0.1433 / +0.0000 | 7 / 2 |
| P3_hc2_same_5percent / frame_freeze_5 | LN_only | 33.5934 | +4.2749 [+2.9791, +5.4728] | +0.4047 / +3.8701 | 4 / 2 |
| P3_hc2_same_5percent / frame_freeze_5 | joint_alpha0 | 32.8429 | +3.5243 [+1.9087, +5.0198] | +3.5243 / +0.0000 | 8 / 3 |
| P3_hc2_same_5percent / frame_freeze_5 | on_policy | 33.2199 | +3.9013 [+2.3441, +5.3095] | -0.0048 / +3.9061 | 8 / 3 |
| P3_hc2_same_5percent / motion_blur_5 | query_only | 29.1584 | -0.2073 [-1.0519, +0.4651] | -0.2073 / +0.0000 | 5 / 1 |
| P3_hc2_same_5percent / motion_blur_5 | LN_only | 33.2554 | +3.8898 [+2.5694, +5.1823] | +0.1710 / +3.7188 | 8 / 3 |
| P3_hc2_same_5percent / motion_blur_5 | joint_alpha0 | 32.9564 | +3.5908 [+2.0836, +5.0360] | +3.5908 / +0.0000 | 9 / 4 |
| P3_hc2_same_5percent / motion_blur_5 | on_policy | 33.4182 | +4.0526 [+2.5421, +5.4930] | +0.1677 / +3.8849 | 7 / 4 |
| P3_hc2_same_5percent / occlusion_5 | query_only | 28.6810 | -0.1132 [-1.0798, +0.7829] | -0.1132 / +0.0000 | 7 / 1 |
| P3_hc2_same_5percent / occlusion_5 | LN_only | 32.4548 | +3.6605 [+2.2356, +5.0343] | -0.1267 / +3.7872 | 10 / 4 |
| P3_hc2_same_5percent / occlusion_5 | joint_alpha0 | 32.2453 | +3.4510 [+1.9494, +4.9087] | +3.4510 / +0.0000 | 12 / 4 |
| P3_hc2_same_5percent / occlusion_5 | on_policy | 32.3715 | +3.5772 [+2.0167, +5.0709] | -0.2116 / +3.7888 | 13 / 4 |
| P3_hc2_same_5percent / exposure_5 | query_only | 29.4782 | +0.1933 [-0.7035, +1.0103] | +0.1933 / +0.0000 | 5 / 1 |
| P3_hc2_same_5percent / exposure_5 | LN_only | 33.0578 | +3.7729 [+2.3231, +5.1610] | +0.0929 / +3.6800 | 9 / 3 |
| P3_hc2_same_5percent / exposure_5 | joint_alpha0 | 33.4012 | +4.1163 [+2.2457, +6.0534] | +4.1163 / +0.0000 | 11 / 4 |
| P3_hc2_same_5percent / exposure_5 | on_policy | 33.2869 | +4.0021 [+2.3283, +5.6119] | +0.0514 / +3.9506 | 11 / 4 |
| P3_vidstg_cross_clean / clean | query_only | 15.0147 | +0.5487 [+0.1139, +1.0253] | +0.5487 / +0.0000 | 3 / 0 |
| P3_vidstg_cross_clean / clean | LN_only | 17.8731 | +3.4071 [+2.0431, +4.7504] | +1.6157 / +1.7914 | 7 / 1 |
| P3_vidstg_cross_clean / clean | joint_alpha0 | 18.0787 | +3.6128 [+2.2191, +4.9687] | +3.6128 / +0.0000 | 5 / 1 |
| P3_vidstg_cross_clean / clean | on_policy | 18.3366 | +3.8706 [+2.4299, +5.2901] | +2.0370 / +1.8336 | 6 / 1 |
| P3_vidstg_same_5percent / frame_drop_5 | query_only | 22.3266 | +0.7100 [-0.0492, +1.6305] | +0.7100 / +0.0000 | 3 / 0 |
| P3_vidstg_same_5percent / frame_drop_5 | LN_only | 23.0153 | +1.3987 [-0.8034, +3.4217] | +0.0852 / +1.3135 | 13 / 5 |
| P3_vidstg_same_5percent / frame_drop_5 | joint_alpha0 | 23.2107 | +1.5941 [-0.5371, +3.6180] | +1.5941 / +0.0000 | 12 / 5 |
| P3_vidstg_same_5percent / frame_drop_5 | on_policy | 23.5810 | +1.9644 [-0.3022, +4.1185] | +0.3029 / +1.6616 | 14 / 6 |
| P3_vidstg_same_5percent / frame_freeze_5 | query_only | 24.8950 | +0.5694 [-0.1123, +1.3615] | +0.5694 / +0.0000 | 3 / 0 |
| P3_vidstg_same_5percent / frame_freeze_5 | LN_only | 26.1121 | +1.7864 [-0.5265, +3.9290] | -0.0065 / +1.7929 | 14 / 7 |
| P3_vidstg_same_5percent / frame_freeze_5 | joint_alpha0 | 25.6061 | +1.2804 [-0.9373, +3.3352] | +1.2804 / +0.0000 | 13 / 7 |
| P3_vidstg_same_5percent / frame_freeze_5 | on_policy | 26.6905 | +2.3648 [+0.1278, +4.5331] | +0.1656 / +2.1992 | 14 / 7 |
| P3_vidstg_same_5percent / motion_blur_5 | query_only | 24.8239 | +0.8358 [+0.1337, +1.7025] | +0.8358 / +0.0000 | 2 / 0 |
| P3_vidstg_same_5percent / motion_blur_5 | LN_only | 25.9384 | +1.9503 [-0.2840, +4.0491] | +0.5983 / +1.3520 | 13 / 5 |
| P3_vidstg_same_5percent / motion_blur_5 | joint_alpha0 | 25.8973 | +1.9092 [-0.3534, +4.0735] | +1.9092 / +0.0000 | 12 / 6 |
| P3_vidstg_same_5percent / motion_blur_5 | on_policy | 25.9974 | +2.0093 [-0.4390, +4.2994] | +0.1204 / +1.8888 | 15 / 6 |
| P3_vidstg_same_5percent / occlusion_5 | query_only | 24.4949 | +0.7030 [+0.0769, +1.4489] | +0.7030 / +0.0000 | 1 / 0 |
| P3_vidstg_same_5percent / occlusion_5 | LN_only | 25.9279 | +2.1361 [-0.1701, +4.3421] | +0.9691 / +1.1670 | 13 / 6 |
| P3_vidstg_same_5percent / occlusion_5 | joint_alpha0 | 25.7318 | +1.9400 [-0.3395, +4.1672] | +1.9400 / +0.0000 | 12 / 7 |
| P3_vidstg_same_5percent / occlusion_5 | on_policy | 26.4847 | +2.6929 [+0.2117, +5.0943] | +1.2414 / +1.4515 | 14 / 6 |
| P3_vidstg_same_5percent / exposure_5 | query_only | 25.1151 | +0.8674 [+0.1196, +1.7972] | +0.8674 / +0.0000 | 2 / 0 |
| P3_vidstg_same_5percent / exposure_5 | LN_only | 25.7492 | +1.5014 [-0.8374, +3.7704] | +0.0464 / +1.4551 | 17 / 7 |
| P3_vidstg_same_5percent / exposure_5 | joint_alpha0 | 26.0006 | +1.7528 [-0.5441, +4.0034] | +1.7528 / +0.0000 | 13 / 8 |
| P3_vidstg_same_5percent / exposure_5 | on_policy | 26.0907 | +1.8430 [-0.6377, +4.2439] | +0.2194 / +1.6236 | 18 / 8 |

Direct uses standard 5 L1 + 2 GIoU and identical admitted support/parameter interface/Adam and fixed last-step budget. Query-only, LN-only and alpha-zero reconstruct their own state from source. Full before/current/inherited decomposition is descriptive and does not replace these interventions. Identical Full/Shuffled/Fixed or K4 streams are reused only with the complete original input, configuration, order and saved-state chain bound by SHA256.

GT expert-quality/duration/motion/query-type strata are descriptive offline associations, never filters or online admission decisions. Corruption coverage denotes the physical time fraction, not equal pixel strength. The main Uniform4 and target configurations remain unchanged. Cached expert cost is separated from actual new DINO calls, decoder fit and CPU audit; reused measurements do not establish cold end-to-end latency.

Root must actually inspect the generated figures and publish actual anonymous code/config/results and negative findings before closing this phase.
