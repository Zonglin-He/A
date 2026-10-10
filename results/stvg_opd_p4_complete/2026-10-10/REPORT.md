# P4: fixed OPD paper stage, actual postseal results

All locked deployment arms and both directions sealed before this phase scoring. Intervals are 10000 paired parent bootstrap resamples after averaging orders within query and queries within parent. All cohorts have historical exposure; no hyperparameters or main budget are selected here.

| Stage / condition | Arm | parent m_vIoU (%) | ΔvIoU (pp), 95% CI | Current / inherited (pp) | Parents harm >5 / >20 pp |
|---|---|---:|---|---|---|
| P4_hc2 / clean | on_policy | 34.4502 | +3.7659 [+2.5784, +4.9006] | -0.7224 / +4.4883 | 18 / 7 |
| P4_hc2 / frame_drop_2.5 | on_policy | 34.3248 | +3.7763 [+2.5947, +4.9297] | -0.6315 / +4.4078 | 20 / 6 |
| P4_hc2 / frame_freeze_2.5 | on_policy | 34.0221 | +3.4055 [+2.1378, +4.6077] | -1.0720 / +4.4775 | 19 / 9 |
| P4_hc2 / motion_blur_2.5 | on_policy | 34.1483 | +3.3066 [+2.0718, +4.4905] | -1.0503 / +4.3570 | 21 / 9 |
| P4_hc2 / occlusion_2.5 | on_policy | 34.1426 | +3.3707 [+2.0968, +4.5993] | -1.0940 / +4.4647 | 23 / 10 |
| P4_hc2 / exposure_2.5 | on_policy | 34.5097 | +3.7997 [+2.5100, +5.0202] | -0.6450 / +4.4447 | 16 / 9 |
| P4_hc2 / frame_drop_5 | on_policy | 33.4899 | +3.5292 [+2.3485, +4.6697] | -0.7395 / +4.2686 | 20 / 7 |
| P4_hc2 / frame_freeze_5 | on_policy | 34.5555 | +3.9248 [+2.7793, +5.0230] | -0.7082 / +4.6331 | 17 / 7 |
| P4_hc2 / motion_blur_5 | on_policy | 33.6169 | +3.0926 [+1.8801, +4.2673] | -1.1903 / +4.2829 | 23 / 9 |
| P4_hc2 / occlusion_5 | on_policy | 33.4323 | +3.0326 [+1.7718, +4.2476] | -1.3754 / +4.4080 | 25 / 9 |
| P4_hc2 / exposure_5 | on_policy | 34.3468 | +3.8109 [+2.5387, +5.0622] | -0.6395 / +4.4504 | 22 / 9 |
| P4_hc2 / frame_drop_10 | on_policy | 32.4387 | +3.6831 [+2.5613, +4.7659] | -0.4334 / +4.1165 | 16 / 5 |
| P4_hc2 / frame_freeze_10 | on_policy | 33.8305 | +3.4924 [+2.3612, +4.5697] | -1.0023 / +4.4947 | 20 / 6 |
| P4_hc2 / motion_blur_10 | on_policy | 32.7950 | +3.1190 [+1.9957, +4.2031] | -0.9317 / +4.0507 | 23 / 5 |
| P4_hc2 / occlusion_10 | on_policy | 33.5186 | +3.6655 [+2.4865, +4.8178] | -0.6067 / +4.2722 | 20 / 6 |
| P4_hc2 / exposure_10 | on_policy | 34.1511 | +3.7409 [+2.5350, +4.8852] | -0.7634 / +4.5044 | 19 / 8 |
| P4_vidstg / clean | on_policy | 23.6794 | +2.4557 [+1.6663, +3.2533] | +0.7987 / +1.6570 | 70 / 25 |
| P4_vidstg / frame_drop_2.5 | on_policy | 22.5425 | +2.2806 [+1.5156, +3.0411] | +0.7144 / +1.5661 | 70 / 23 |
| P4_vidstg / frame_freeze_2.5 | on_policy | 23.5218 | +2.2937 [+1.5159, +3.0694] | +0.6263 / +1.6674 | 75 / 27 |
| P4_vidstg / motion_blur_2.5 | on_policy | 23.3453 | +2.2413 [+1.4948, +2.9768] | +0.7493 / +1.4920 | 68 / 23 |
| P4_vidstg / occlusion_2.5 | on_policy | 23.1437 | +2.1956 [+1.4275, +2.9501] | +0.5641 / +1.6314 | 71 / 24 |
| P4_vidstg / exposure_2.5 | on_policy | 23.5209 | +2.4076 [+1.6405, +3.1828] | +0.7076 / +1.7000 | 67 / 24 |
| P4_vidstg / frame_drop_5 | on_policy | 21.2954 | +2.0388 [+1.2806, +2.7922] | +0.5750 / +1.4638 | 65 / 24 |
| P4_vidstg / frame_freeze_5 | on_policy | 23.3570 | +2.0878 [+1.3469, +2.8428] | +0.6741 / +1.4137 | 72 / 24 |
| P4_vidstg / motion_blur_5 | on_policy | 23.0093 | +2.0303 [+1.2737, +2.7758] | +0.4947 / +1.5356 | 73 / 23 |
| P4_vidstg / occlusion_5 | on_policy | 23.0767 | +2.2014 [+1.4289, +2.9701] | +0.5761 / +1.6253 | 70 / 25 |
| P4_vidstg / exposure_5 | on_policy | 23.3448 | +2.1371 [+1.3708, +2.9089] | +0.7459 / +1.3912 | 74 / 24 |
| P4_vidstg / frame_drop_10 | on_policy | 19.5592 | +1.7418 [+1.0811, +2.4168] | +0.4259 / +1.3160 | 68 / 23 |
| P4_vidstg / frame_freeze_10 | on_policy | 23.3471 | +2.5309 [+1.7625, +3.3085] | +0.7364 / +1.7945 | 63 / 22 |
| P4_vidstg / motion_blur_10 | on_policy | 22.7144 | +2.1321 [+1.3905, +2.8666] | +0.5600 / +1.5721 | 75 / 23 |
| P4_vidstg / occlusion_10 | on_policy | 22.7652 | +2.2266 [+1.4668, +2.9805] | +0.3084 / +1.9182 | 63 / 24 |
| P4_vidstg / exposure_10 | on_policy | 23.4212 | +2.2586 [+1.4952, +3.0208] | +0.5120 / +1.7466 | 68 / 23 |

Direct uses standard 5 L1 + 2 GIoU and identical admitted support/parameter interface/Adam and fixed last-step budget. Query-only, LN-only and alpha-zero reconstruct their own state from source. Full before/current/inherited decomposition is descriptive and does not replace these interventions. Identical Full/Shuffled/Fixed or K4 streams are reused only with the complete original input, configuration, order and saved-state chain bound by SHA256.

GT expert-quality/duration/motion/query-type strata are descriptive offline associations, never filters or online admission decisions. Corruption coverage denotes the physical time fraction, not equal pixel strength. The main Uniform4 and target configurations remain unchanged. Cached expert cost is separated from actual new DINO calls, decoder fit and CPU audit; reused measurements do not establish cold end-to-end latency.

Root must actually inspect the generated figures and publish actual anonymous code/config/results and negative findings before closing this phase.
