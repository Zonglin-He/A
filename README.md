# BoxFixed + Global6

A frozen TA-STVG inference method for cross-domain spatio-temporal video grounding.
The main pipeline uses one ReferDINO with eight observations, complete native-box
propagation, six-layer native224 temporal posteriors and the original physical MBR
and exact BoxFixed dynamic program. There is no network training, OPD update,
448 branch or conditional decoder replay in this release.

## Pipeline

1. Run the frozen source model at height 224 with the original two offsets.
2. Select the original eight observations and obtain native ReferDINO boxes and masks.
3. Keep complete teacher box extents. Use soft masks and source-feature correspondence
   to estimate forward/backward box transforms; always retain the initialized source candidate.
4. Read all six native temporal layers through the original frozen time head. Build
   the original two-offset interval hull separately per layer, then average posteriors.
5. Decode the time interval with the original physical MBR and select a spatial path
   with the fixed BoxFixed score and exact DP.

See [method](docs/method.md), [evaluation](docs/evaluation.md),
[private assets](docs/assets.md) and [verification](results/verification.json).

## Installation

Use a compatible PyTorch/CUDA environment and the dependencies in
[environment.yml](environment.yml), [requirements.txt](requirements.txt) and
[requirements-teacher.txt](requirements-teacher.txt). Place the frozen assets at
[the documented paths](docs/assets.md). They are not bundled or downloaded automatically.

```bash
python tools/setup.py --prepare --tokenizer /path/to/existing/bert-tokenizer
python tools/setup.py --build
python tools/setup.py --check
python tools/check.py
```

## Inference and scoring

The sole public method is the same for both directions. `hc2` uses the VidSTG-trained
source checkpoint; `vidstg` uses the HC2-trained checkpoint.

```bash
python -m stvg predict --request /path/to/request.json --output outputs/example.npz
python -m stvg evaluate --manifest /path/to/label-free-manifest.jsonl --output outputs/evaluation
python -m stvg score --manifest /path/to/label-free-manifest.jsonl --labels /path/to/offline-labels.jsonl --predictions outputs/evaluation --output outputs/metrics.json
```

[examples/request.json](examples/request.json) documents the request schema. A manifest
contains `id`, `dataset`, `key`, `metadata`; metadata is limited to raw input support,
video properties and caption. No boxes, event labels or GT-derived online choices are accepted.
Evaluation seals predictions before independent CPU scoring. Resume requires the exact
manifest, runtime fingerprint and saved prediction bindings; code cleanup does not
waive those checks. The public resident evaluator reuses the preceding decoded RGB grid and a 4 GiB
actual-tensor visual cache by default. The existing full controller has its separately
verified 512 MiB RGB LRU. These affect reuse, not model inputs or precision.

## Results and history

The complete official evaluation is sealed and scored: HC2 has 1,902 queries from
227 parents and VidSTG10,303 queries from732. [Full results and signed changes](results/boxfixed_full_v1/REPORT.md)
show vIoU24.9948→26.7597 on HC2 and11.7777→14.2685 on VidSTG. The development
panels have prior exposure; these complete splits are not pristine independent confirmations.

This is a source cleanup, not a new experimental result. The completed full job retains
its original immutable source checkout and fingerprint; it was not restarted with this
refactor. Cached compatibility covers 192 saved captures and is described in the
verification receipt; no new GPU run was performed for this cleanup.

All previous methods, Strong/448 references, unsuccessful trials and signed results
remain in the [research archive](https://github.com/Zonglin-He/A/tree/archive/research-pre-mainline-20261022) and Git history. They are deliberately
absent from the current inference tree. See [archive policy](docs/archive.md) and
[history](docs/history.md). Upstream source and revisions are retained under
[external/](external) and [third_party.json](docs/third_party.json).

User-authorized HC2 temporal calibration is recorded separately in the
[bounded tuning report](results/temporal_hc_v2/REPORT.md). Its selected profile failed
the fixed validation; it is not part of the maintained inference method.

[Broader HC temporal readout study](results/temporal_hc_v3/REPORT.md) reports all16
additional configurations and a preserved failed Source validation gate. Its frozen
start-temperature full comparison was explicitly paused by the user at338 queries
because Source superiority was not established. It has not replaced the main method.

[Source-primary calibration](results/temporal_hc_v4/REPORT.md) and
[strict short/long temporal study](results/temporal_hc_v5/REPORT.md) preserve the next
negative results. The selected MBR3 control did not keep long-event tIoU above Source
on the fixed development recheck. No full restart or inference-default change follows.
