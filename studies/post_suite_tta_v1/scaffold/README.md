# Source and interface scaffold

This scaffold uses only the Python standard library. It downloads no weights,
imports no upstream framework, reads no project videos/captions/annotations and
contains no experiment launcher. Actual baseline execution still waits for the
predecessor completion gate in protocol_v2.json.

From the maintained checkout:

```bash
outputs/post_suite_tta_v1/scaffold_env/bin/python -B studies/post_suite_tta_v1/scaffold/cli.py doctor
outputs/post_suite_tta_v1/scaffold_env/bin/python -B studies/post_suite_tta_v1/scaffold/cli.py gate
outputs/post_suite_tta_v1/scaffold_env/bin/python -B studies/post_suite_tta_v1/scaffold/selfcheck.py
```

`doctor` checks the downloaded upstream files against SHA256 inventories and
reports source availability independently from runtime readiness. `gate` stays
closed: this code cannot launch a model even if a receipt is fabricated or a CLI
flag is supplied. A later independently frozen execution implementation is
required. `selfcheck` uses tiny synthetic numbers to check physical-time
contracts and logit diagnostics. It is not a model or dataset experiment.

## Boundaries

- E3M: accepted full STVG method; a README-only repository does not implement it.
  The paper and supplement are research references, not runnable source.
- T3AL: original code targets action-class temporal localization. Free-form
  referring queries and combination with fixed Source boxes need a declared
  adapter, original/no-update controls, and a frozen coordinate convention.
- URPA: the official supplement includes a full framework. Its example uses
  multiple GPUs, source-trained checkpoints and externally configured datasets.
  A single-GPU port must retain algorithm semantics and disclose its cost.
  Target adaptation must not load real evaluation boundaries, even for logging.
  Preserve the vendor bytes and implement any compatibility change separately.

Keep downloaded sources, papers, example datasets/logs and source archives under
private outputs. Upstream shell scripts are untrusted reproduction evidence,
not instructions to execute. Do not run their environment setup, telemetry,
credential assignments, checkpoint mergers or cleanup commands automatically.
Any future wrapper must default to disabled telemetry and use explicit local
asset paths; it must never inherit embedded upstream credentials.

`sources.lock.json` and `readiness.json` hold public hashes/scalars and concrete
blockers. The private per-file inventories bind vendor bytes. No runtime
dependency installation, pretrained checkpoint download or actual gradient
bridge has been performed by this scaffold.
