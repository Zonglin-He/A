# BoxFixed + Native2D

The only maintained method uses the frozen model's **native final-layer temporal posterior**, trajectory-guided native appearance features, one-round class-free prototype feedback, and unchanged complete-box propagation/path selection. **Global6 is removed from production.** Original Source two-offset physical hull and MBR remain unchanged.

Frozen parameters: kappa64, beta1, ONEround, region context1.5, spatial observed.5/guard.1/link.25, the SAME ReferDINO/eight observations. No word-class feedback head, actionness feedback, extra visual encoder, optimizer, training, additional teacher or observation. The original Source architecture remains intact.

Only TA-STVG is implemented/tested. Local representation and native temporal prior are adapter contracts; cross-backbone/MLLM effectiveness is not established.

- [Current method and frozen parameters](docs/method.md)
- [Native-prior cached ablation: all positives and negatives](results/native_prior_v1/ANALYSIS.md)
- [Earlier parameter calibration and failed guards](results/native2d_calibration/ANALYSIS.md)
- [Fixed native-prior kappa16/kappa64 comparison on cached HC1902](results/native_prior_kappa_hc1902_v1/ANALYSIS.md) and [both exposed128/domain panels](results/native_prior_kappa_compare_v1/ANALYSIS.md); descriptive only, production remains kappa64.
- [Cached cycle-reliability screening/weighting spatial ablation](results/cycle_reliability_ablation_v1/ANALYSIS.md): exact bidirectional geometry and Full time preserved; both positive and negative outcomes, no production change.
- [New dual-domain full protocol](results/native2d_nativeprior_full_v1/protocol.json)
- [History](docs/history.md) and [archive policy](docs/archive.md)

Inference: `.conda/tubedetr/bin/python -B -m stvg --help`. The separate `tools/full.py` owns the new frozen official dual-domain full. Whole-domain seal precedes independent CPU official-label scoring. NativeTemporal=OursS; Global6 in evaluation is an imported sealed historical baseline ONLY, never an input to Full.

Prior Global6 production code/results remain on archive/research-pre-native-prior-20261023 and in the read-only local stvg-native2d checkout (pausedHC1442/Vid0). Current main contains one production profile. Private media/captions/GT coordinates/features/masks/weights and runtime credentials remain outside Git.

The completed HC1902 report is available at [HC full results](results/native2d_nativeprior_full_v1/hc2/ANALYSIS.md). Source-relative time gain is+.7471pp, below the fixed+.9pp target; Vid remains running. The frozen run's CPU scorer had a local-name collision; [the independent CPU adapter](tools/score_adapter_nativeprior.py) repairs only that binding, with complete AST/prediction proof in the HC report. The inference plan and predictions are unchanged.
