# HC64 state-candidate OPD

The original 128-query plan was reduced by the user, without outcome selection,
to the first 64 distinct parents in the original manifest. The original worker
configuration and fingerprint remain intact; `scope64.py` stops it after the
selected 64 complete artifacts, seals that prefix, then invokes CPU scoring.

This isolated exposed-development experiment is specified in `config.json` and
`docs/state_opd.md`. It does not change the registered method or resume formal
evaluation. It reuses the sealed HC128 Source inputs, ERef/geometry predictions,
Strong intervals and native ReferDINO masks. Dataset files, labels, checkpoints
and raw caches are private and are not distributed with the code.

`run.py` defaults to a no-inference preparation and identity check. An explicit
`--run` executes the four locked methods. `--resume` only accepts the identical
configuration, source, checkpoint and input lock, and preserves completed query
artifacts. Every query resets the temporary state and optimizer. No image or
expert forward is needed; frozen decoder evaluation is the only model work.

Start the worker in terminal A:

```bash
python -B scripts/state_opd/run.py --run
```

While that worker is running, start the selected-scope controller in terminal B.
Do not wait for the original 128-query worker to finish before starting it:

```bash
python -B scripts/state_opd/scope64.py --watch --score
```

All selected 64 predictions must be sealed and the owned worker exited before the separate CPU evaluator may read
GT. It cross-checks the reference predictions, traces, configuration and lock,
then uses the existing official and independent dense metrics.

The scope controller invokes CPU scoring automatically after that seal; do not
start a second scorer while it is already running.

The four new arms share the same fixed 8×4 latent space. Geometry uses the old
registered full state and is a method reference, not a matched optimization
ablation. The common allowance is at most 1321 complete-trajectory forward
evaluations per arm/query, with actual backwards, time and memory reported
separately. Backtracking consumes evaluations, and deterministic rejection can
stop Direct early; unused budget is not filled with duplicate evaluations.
All positive and negative results must be retained. This is one fixed
configuration, not independent confirmation or a method-specific tuning study.
