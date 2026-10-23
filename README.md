# BoxFixed + Native2D

The single maintained method uses eight observations from one ReferDINO, shared geometric initialization, complete-box propagation and path selection. Its temporal feedback aggregates native local appearance BEFORE textfusion, constructs one-round class-free region-kernel prototypes from the frozen native temporal prior, and applies calibrated evidence with the original physical-time MBR.

Only TA-STVG is implemented and tested. Native local representation is an adapter contract; other backbones/MLLM effectiveness is untested. The original Source architecture remains intact. There is no word-class feedback head, external visual encoder, query optimizer, extra expert or training.

- [Method and exact frozen parameters](docs/method.md)
- [Calibration, all signs and severe harms](results/native2d_calibration/ANALYSIS.md)
- [New complete-evaluation protocol](results/native2d_full_v2/protocol.json)
- [Archive and evidence policy](docs/archive.md)
- [History and current execution](docs/history.md)

Run `.conda/tubedetr/bin/python -B -m stvg --help` for the only inference entry. `tools/full.py` manages the separately frozen full protocol. `tools/full_score.py` reads fixed official labels only after the entire domain seals. New Full outputs are never aliases of archived fused/word-head methods. NativeTemporal=OursS is explicit.

Old executable methods, tuning launchers and their broad result trees have been removed from this maintained tree and retained with all signs on `archive/research-pre-native2d-20261023`. Local old checkouts are immutable evidence; only `stvg-native2d` is maintained. Private media/captions/GT coordinates/features/weights stay outside Git.
