# External dependencies

Third-party source code, model weights and datasets are not vendored in this snapshot. Obtain them from their respective projects and observe their licenses and dataset access conditions.

| Component | Upstream | Local code revision used |
|---|---|---|
| ParallelTubeDecoding | https://github.com/mbzuai-oryx/ParallelTubeDecoding | `d0bf40bff60524bdd3608d601cc8e37465193d18` |
| TA-STVG | https://github.com/HengLan/TA-STVG | `904ad0c344eb12fe34513f97a33f93118f6d3be4` |

For the PTD integration, the original directory is `external/ParallelTubeDecoding/`, and the local official 4B checkpoint is `checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/`. Source files use strict expected layouts and hashes. Installing the upstream repository alone does not supply the private runtime manifests or model files.

Observed PTD environment: Python environment with torch 2.7.0+cu128, torchvision 0.22.0+cu128, transformers 5.12.1, accelerate 1.10.1, qwen-vl-utils 0.0.14, av 18.1.0, Pillow 12.3.0, numpy 1.26.4 and safetensors 0.8.0. This is a record of the actual research runtime, not a tested fresh-install lockfile. Full GPU integration was not repeated for this publication.

DeCoTA uses a separate TA-STVG environment, source model checkpoints, local Grounding DINO weights and text/vision resource caches. Its own README and config specify that interface. Do not treat the PTD environment as a compatible replacement for the TA-STVG environment.

Some local loading helpers were extracted from the prior research runtime; their provenance is recorded in `methods/decota_final_simplified_v1/SOURCE_MAP.json`. `RELEASE.json` retains integrity pins, including external dependencies not shipped here. Missing dependencies should be prepared, not bypassed by disabling checks.

