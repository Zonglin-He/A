# SimplePath + Native2D

The maintained method combines **bidirectional complete-box propagation and exact path selection** with **one video-wide Native2D prototype pair**. It uses the frozen model's native final-layer temporal posterior and pre-text appearance grid. Global6, cycle-reliability scoring, extent RBF, initialization guard and mask-transport links are absent from the production algorithm.

The spatial score contains available-anchor appearance similarity and adjacent complete-box IoU. The temporal module pools the original local appearance grid along a full-video SimplePath trajectory, estimates one event/background pair from all frames using native soft membership, applies common-mean shrinkage, and revises the native interval posterior once. There is **no offset-wise prototype fitting**. The original Source still runs its unchanged two offsets; its physical hull and final MBR remain.

Fixed parameters: kappa64, beta1, one round, region context1.5, spatial .5 observed/.5 unobserved/.25 link. The same single ReferDINO supplies the same eight observations. No extra encoder, teacher, observations, word-class feedback, actionness feedback, training or optimizer is introduced.

- [Fig. 1: real paired entropy-minimization TTA experiment](results/entropy_tta_v2/ANALYSIS.md)
- [Earlier sample-level surrogate correlation diagnostic](results/surrogate_alignment_v1/README.md)
- [Current algorithm and contracts](docs/method.md)
- [New independent dual-domain full protocol](results/simplepath_full_v2/protocol.json)
- [Modular adoption evidence, including negative results](results/algorithm_pruning_v2/ANALYSIS.md)
- [Actual-grid coupled bridge](results/simplepath_full_v2/BRIDGE.json) and [cached adoption checks](results/simplepath_full_v2/REGRESSION.json)
- [Historical native-prior HC1902 full](results/native2d_nativeprior_full_v1/hc2/ANALYSIS.md): time +.7471pp, below the fixed +.9pp target; preserved unchanged.
- [History](docs/history.md) and [archive policy](docs/archive.md)

Only TA-STVG has been evaluated. Dimension-free local-representation and temporal-evidence interfaces do not establish effectiveness on other backbones or MLLMs.

Inference: `.conda/tubedetr/bin/python -B -m stvg --help`. The independent `tools/full.py` queue owns `outputs/simplepath_full_v2`. Complete domain seal precedes CPU official-label scoring. Source/SharedInit/eight positions must reproduce the original official outputs; Global6 is only an imported historical baseline. Full/NoLocal/Independent are recomputed under this new method. NativeTemporal=OursS.

The old full is safely paused at HC1902/Vid2388. Its code, logs and all4290 receipts/predictions/captures remain immutable in the old local checkout and archived Git history. Old pooled Native2D vectors cannot be reused after the auxiliary path changes; current collection recomputes the original appearance prefix when necessary and pools on the new path. This acquisition cost is recorded separately from historical reuse and from deployment timing.

Public code includes the current method, protocol, complete signed positive/negative results and engineering evidence. Prior executable research variants are retained in the archive branch. Videos, captions, GT coordinates, raw grids/masks, checkpoints and credentials remain private.
