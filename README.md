## Current selected method: no shrinkage (2026-10-25)

Use [CURRENT_METHOD.json](CURRENT_METHOD.json) and `python -m stvg.current predict --request REQUEST.json --output OUTPUT.npz`. The selected profile uses **kappa0, final jointMAP and actual interval-conditioned exact spatial DP**. [Current main values, tails and dependent recalculation status](results/no_shrink_main_v1/README.md). Legacy configs/default.json and commands below remain immutable evidence interfaces.

# SimplePath + Native2D

**Source naming correction:** official TA-STVG uses native per-offset MAP plus physical hull, not MBR. Historical local `Source` result keys mean **Source + our MBR decoder**. Read the [explicit three-row comparison and correction](results/source_semantics_v1/README.md); native Source, Source+MBR and Full must be reported separately.
The maintained method combines **bidirectional complete-box propagation and exact path selection** with **one video-wide Native2D prototype pair**. It uses the frozen model's native final-layer temporal posterior and pre-text appearance grid. Global6, cycle-reliability scoring, extent RBF, initialization guard and mask-transport links are absent from the production algorithm.

The spatial score contains available-anchor appearance similarity and adjacent complete-box IoU. The temporal module pools the original local appearance grid along a full-video SimplePath trajectory, estimates one event/background pair from all frames using native soft membership, applies common-mean shrinkage, and revises the native interval posterior once. There is **no offset-wise prototype fitting**. The source network retains its two offsets; native MAP-hull readout and our added MBR decoder are distinct, with explicit controls.

Fixed parameters: kappa64, beta1, one round, region context1.5, spatial .5 observed/.5 unobserved/.25 link. The same single ReferDINO supplies the same eight observations. No extra encoder, teacher, observations, word-class feedback, actionness feedback, training or optimizer is introduced.

- [Complete HC1902 results](results/simplepath_full_v2/hc2/ANALYSIS.md) and [Vid10303 results](results/simplepath_full_v2/vidstg/ANALYSIS.md)
- [Supplemental paper-review protocol](results/paper_review_suite_v1/protocol.json), [full cached temporal controls](results/paper_review_suite_v1/cached_protocol.json), and [matched TTA384 results](results/paper_review_suite_v1/consistency_v2/ANALYSIS.md)
- [Completed CG-STVG full evaluation](results/cgstvg_full_v1/README.md) and [TA/CG complete MAP+DP comparisons](results/map_full_now_v3/README.md): original Source/readout controls, all signed results and audit receipts. The earlier [interface-preparation record](results/paper_review_suite_v1/CGSTVG_ADAPTER_PREPARATION.md) is preserved.
- [Fig. 1: real paired entropy-minimization TTA experiment](results/entropy_tta_v2/ANALYSIS.md)
- [Earlier sample-level surrogate correlation diagnostic](results/surrogate_alignment_v1/README.md)
- [Current algorithm and contracts](docs/method.md)
- [New independent dual-domain full protocol](results/simplepath_full_v2/protocol.json)
- [Modular adoption evidence, including negative results](results/algorithm_pruning_v2/ANALYSIS.md)
- [Actual-grid coupled bridge](results/simplepath_full_v2/BRIDGE.json) and [cached adoption checks](results/simplepath_full_v2/REGRESSION.json)
- [Historical native-prior HC1902 full](results/native2d_nativeprior_full_v1/hc2/ANALYSIS.md): time +.7471pp, below the fixed +.9pp target; preserved unchanged.
- [History](docs/history.md) and [archive policy](docs/archive.md)

The main dual-domain TA-STVG full evaluation is complete and published. Its original per-direction +0.9pp criterion remains a historical record: HC +0.9295pp, Vid +0.7998pp, joint target_not_met; the user accepts the measured method and removed that threshold as a work blocker. The independently reopened supplemental suite does not retune production. Only TA-STVG has been evaluated. Dimension-free local-representation and temporal-evidence interfaces do not establish effectiveness on other backbones or MLLMs.

Inference: `.conda/tubedetr/bin/python -B -m stvg --help`. The independent `tools/full.py` queue owns `outputs/simplepath_full_v2`. Complete domain seal precedes CPU official-label scoring. Source boxes/native heads/eight positions retain the saved native-input contract; stored `Source` intervals use our added MBR, not official native decoding; Global6 is only an imported historical baseline. Full/NoLocal/Independent are recomputed under this new method. NativeTemporal=OursS.

The old full is safely paused at HC1902/Vid2388. Its code, logs and all4290 receipts/predictions/captures remain immutable in the old local checkout and archived Git history. Old pooled Native2D vectors cannot be reused after the auxiliary path changes; current collection recomputes the original appearance prefix when necessary and pools on the new path. This acquisition cost is recorded separately from historical reuse and from deployment timing.

Public code includes the current method, protocol, complete signed positive/negative results and engineering evidence. Prior executable research variants are retained in the archive branch. Videos, captions, GT coordinates, raw grids/masks, checkpoints and credentials remain private.


## Final current-MAP supplemental evaluation

[All four TA/CG full-domain MAP+DP comparisons](results/map_full_now_v3/README.md), [original CG full results](results/cgstvg_full_v1/README.md), [TA progressive components/final-DP controls](results/map_components_v1/README.md) and [own-K fixed-cohort budgets with preserved numeric recovery](results/map_budget_restore_v3/README.md) are complete and audited. The complete source/readout comparisons, positive/negative results, severe tails, parent-cluster CIs, exposure limits and measured cost scopes are preserved; the latest closure records supersede historical queue-status prose. No full-result parameter selection or new automatic experiment is performed.


Latest newly authorized experiments (all review items except TubeDETR) are independently running/prepared in [MAP paper completion](results/map_paper_completion_v1/README.md). Existing compatible completed results are reused; launch protocols do not assert new measurements complete.
