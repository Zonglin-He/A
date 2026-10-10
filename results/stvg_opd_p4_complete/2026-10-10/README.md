# Fixed OPD P4: complete same-domain robustness evidence

Read [the actual root review](ACTUAL_ROOT_REVIEW.md), all sixteen conditions in
[ROOT_STATISTICS.json](ROOT_STATISTICS.json), [all parent effects](ALL_PARENT_EFFECTS.json),
[negative attribution](FAILURE_STRATA.json), [actual costs](COST.json), and
[current/inherited and observed/unobserved readback](ACTUAL_ROOT_ALL_CONDITION_MECHANISM_COST.json).
All 15,504 anonymous rows are under `stages`: 237 HC2 parents and 732 VidSTG parents,
one query per parent, clean plus five physical-burst families at 2.5/5/10% coverage.
Every condition resets to its same-domain source. All are new formal fits; no aliases
or qualification predictions are scored. All deployment predictions sealed before GT.
Large anonymous root JSON files use deterministic gzip; CODE_BINDING records original
uncompressed bytes/SHA and public compressed bytes/SHA, both independently checked.

```bash
python -B scripts/audit_stvg_opd_p4_export_v2.py results/stvg_opd_p4_complete/2026-10-10
```

Complete parent confidence intervals, negative tails, strata and actual cost are retained.
Current/inherited decomposition describes the Full trajectory, not an alpha0 causal contrast.
All saved mathematical/state/dense records and source statistics were actually checked;
five report plot pairs and six private RGB sheets were actually viewed. Private case RGB
must match the original physical-corruption pixel hash and corruption specification.
Stored timings include synchronous numerical recording and are not cold service latency.
Public arithmetic is reproducible; this package does not reproduce private inference,
implement the full decoder Jacobian or prove CUDA transcendental kernels.
No private RGB/query/caption/GT geometry/box/action/weights/fit/gradient/Adam payload is
exported. Original P5/P6 still require actual root/view/public/archive closing;
EATA and historical paused queues remain paused. No retuning or method promotion.
