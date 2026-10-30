# R8 independent CPU evaluation commands

All commands use one CPU thread and do not invoke a model/GPU. Do not execute the
scoring commands until root writes the matching true release after both prediction
stages are sealed. `FULL_SEAL_SHA`, `RELEASE_SHA` and `RESULT_SHA` below are explicit
placeholders supplied from the immutable corresponding receipts; not shell variables.
The program root is the workspace root because R8 producer seals use workspace-relative
program paths. The scoring index remains separate from model inputs.

## Full unlabelled gate (no GT release needed after root authorizes this audit)

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <workspace>/crossplatform/env/bin/python <workspace>/research/round8/evaluation/audit_unlabelled_predictions.py --predictions <workspace>/crossplatform/round8/full5912_attempt001 --manifest <workspace>/research/round8/experiment/full5912_gpu_inputs.json --workspace-root <workspace> --expected-seal-sha256 FULL_SEAL_SHA --output <workspace>/research/round8/evaluation/FULL5912_UNLABELLED_AUDIT.json
```

## Full post-seal task scoring (only after true release)

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <workspace>/crossplatform/env/bin/python <workspace>/research/round8/evaluation/score_crossdomain.py --predictions <workspace>/crossplatform/round8/full5912_attempt001 --manifest <workspace>/research/round8/experiment/full5912_gpu_inputs.json --scoring-index <workspace>/crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json --source-checkpoint <workspace>/crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth --release <workspace>/research/round8/experiment/FULL_SCORE_RELEASE.json --release-sha256 RELEASE_SHA --program-root <workspace> --output <workspace>/research/round8/experiment/scored_full5912_attempt001
```

For smoke scoring use `crossplatform/round8/smoke40_attempt001`,
`research/round8/experiment/smoke40_gpu_inputs.json`, `SMOKE_SCORE_RELEASE.json`,
and a new `scored_smoke40_attempt001` output. It still waits for the full prediction
seal and its own true release; smoke task labels are not used to tune full.

## Independent score-row reduction audit

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <workspace>/crossplatform/env/bin/python <workspace>/research/round8/evaluation/audit_scored_results.py --results <workspace>/research/round8/experiment/scored_full5912_attempt001/RESULTS_PRIVATE.json --expected-results-sha256 RESULT_SHA --output <workspace>/research/round8/evaluation/FULL5912_SCORE_REDUCTION_AUDIT.json
```

The scorer performs independent general3D versus batch geometry for selected boxes,
while this separate audit recomputes aggregation and paired contrasts from sealed
private score rows. It does not claim a third full256 independent geometry pass.
Private rows and raw scene names must not be copied wholesale to public artifacts.
