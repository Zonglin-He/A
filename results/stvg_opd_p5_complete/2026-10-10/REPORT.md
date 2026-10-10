# P5: fixed OPD paper stage, actual postseal results

All locked deployment arms and both directions sealed before this phase scoring. Intervals are 10000 paired parent bootstrap resamples after averaging orders within query and queries within parent. All cohorts have historical exposure; no hyperparameters or main budget are selected here.

| Stage / condition | Arm | parent m_vIoU (%) | ΔvIoU (pp), 95% CI | Current / inherited (pp) | Parents harm >5 / >20 pp |
|---|---|---:|---|---|---|
| P5_hc2_K1 / clean | on_policy | 22.8113 | +0.6554 [-0.7915, +1.9284] | -0.8618 / +1.5172 | 12 / 3 |
| P5_hc2_K2 / clean | on_policy | 22.8195 | +0.6636 [-0.7390, +1.8587] | -0.8096 / +1.4732 | 10 / 3 |
| P5_hc2_K4 / clean | on_policy | 23.5020 | +1.3461 [-0.0460, +2.5255] | -0.4301 / +1.7761 | 11 / 2 |
| P5_hc2_K8 / clean | on_policy | 22.4975 | +0.3416 [-1.3935, +1.8837] | -1.3839 / +1.7255 | 15 / 5 |
| P5_vidstg_K1 / clean | on_policy | 16.7147 | +2.2488 [+0.9852, +3.4830] | +0.6838 / +1.5651 | 8 / 1 |
| P5_vidstg_K2 / clean | on_policy | 18.1149 | +3.6490 [+2.2051, +5.0809] | +2.0972 / +1.5518 | 5 / 1 |
| P5_vidstg_K4 / clean | on_policy | 18.3366 | +3.8706 [+2.4299, +5.2901] | +2.0370 / +1.8336 | 6 / 1 |
| P5_vidstg_K8 / clean | on_policy | 18.9238 | +4.4579 [+2.8465, +6.0781] | +2.7379 / +1.7200 | 7 / 2 |
| P5_unified_hc2 / clean | on_policy | 23.0540 | +0.8980 [-0.3351, +2.0181] | -0.9709 / +1.8689 | 15 / 2 |
| P5_unified_vidstg / clean | on_policy | 18.3863 | +3.9204 [+2.4294, +5.3642] | +2.1273 / +1.7930 | 6 / 1 |

Direct uses standard 5 L1 + 2 GIoU and identical admitted support/parameter interface/Adam and fixed last-step budget. Query-only, LN-only and alpha-zero reconstruct their own state from source. Full before/current/inherited decomposition is descriptive and does not replace these interventions. Identical Full/Shuffled/Fixed or K4 streams are reused only with the complete original input, configuration, order and saved-state chain bound by SHA256.

GT expert-quality/duration/motion/query-type strata are descriptive offline associations, never filters or online admission decisions. Corruption coverage denotes the physical time fraction, not equal pixel strength. The main Uniform4 and target configurations remain unchanged. Cached expert cost is separated from actual new DINO calls, decoder fit and CPU audit; reused measurements do not establish cold end-to-end latency.

Root must actually inspect the generated figures and publish actual anonymous code/config/results and negative findings before closing this phase.
