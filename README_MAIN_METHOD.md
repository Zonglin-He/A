# Query–Caption Temporal + Direct Spatial OPD

The single research method is registered in `methods/MAIN_METHOD.json`.
One frozen Qwen3.5-9B provides both event descriptions/preferences and sparse
Direct geometry. Temporal uses the original 512-parameter merged-interval KL
adapter and marginal MAP. Spatial uses the original 1792-parameter episodic
Gaussian OPD, complete response control and rollback. No external Actionness,
DINO, cross-query state inheritance or tuning is part of this entry point.

```bash
.venv/stvg_qwen35_geometry_v1/bin/python -B -m main_method.run prepare
.venv/stvg_qwen35_geometry_v1/bin/python -B -m main_method.qualification cpu
.venv/stvg_qwen35_geometry_v1/bin/python -B -m main_method.qualification processor
```

The serial supervisor is `python -B -m main_method.controller`. It refuses to
run new models without a remotely verified cleanup-publication receipt and a
frozen code lock. Individual stage commands are exposed by
`python -m main_method.run --help`; source/Teacher/readout stages own the common
GPU lease. Private datasets, checkpoints and environment installations are not
published; place the existing verified assets at their recorded local paths.
The compatible Qwen environment is isolated from the Student/R1 environments.

Code cleanup materializes a dependency-only tree under `main_method_release/code`;
all included source bytes and pure pairwise-function extraction are checked.
Historical source has a local recoverable tar snapshot and remains recoverable
from the parent Git commit. Existing results and negative findings are retained.
Unrelated running research services and their working source dependencies are
not removed. Legacy names in retained primitive dependencies do not activate
historical launchers; only `main_method` is the current execution entry.

The official population is HC2 validation 3482 queries/237 videos and VidSTG
test 10303 queries/732 videos. A fixed old development32 first checks actual
end-to-end integration and cost; it is not independent confirmation and cannot
select a new recipe. The official, three component rows, original corruption
families/severities, sparse propagation and K=1/2/4/8 analyses remain distinct
measured groups. Every group requires actual prediction seals, independent
readback, post-seal GT evaluation, privacy/pixel review and verified publication.
Preparation, engineering qualification and complete scientific measurement are
reported separately. Figure 1 and manuscript writing are not part of this run.

See `protocols/stvg_main_qwen_v1.md`. Existing QC results have not established
an extra benefit over the same marginal decoder without updates; that control
and the historical failed confirmation remain in the research record.
