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

The current official full evaluation is still running. HC2 has 1,902 queries from
227 parent sources; VidSTG has 10,303 queries from 732. Full results will be imported
only after both domains seal and score under the frozen protocol. The development
panels have prior exposure and are not independent test confirmations.

This is a source cleanup, not a new experimental result. The existing full job keeps
its original immutable source checkout and fingerprint; it is not restarted with this
refactor. Cached compatibility covers 192 saved captures and is described in the
verification receipt; no new GPU run was performed for this cleanup.

All previous methods, Strong/448 references, unsuccessful trials and signed results
remain in the [research archive](https://github.com/Zonglin-He/A/tree/archive/research-pre-mainline-20261022) and Git history. They are deliberately
absent from the current inference tree. See [archive policy](docs/archive.md) and
[history](docs/history.md). Upstream source and revisions are retained under
[external/](external) and [third_party.json](docs/third_party.json).
