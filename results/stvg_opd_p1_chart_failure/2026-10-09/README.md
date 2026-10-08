# P1 native-box chart failure: diagnosed, revision decision pending

See docs/STVG_OPD_P1_CHART_FAILURE_DIAGNOSIS.md for the actual failure and scope.
HC2 has sealed all 10,446 arrivals; the opaque integrity proof is included here.
VidSTG failed after 1,882 outputs. The global P1 seal and P1 GT scores are absent.
Original code, scientific locks and all receipted predictions remain preserved.

One actual GPU replay from the full saved prefix reproduced a finite native
logit whose float32 sigmoid output rounds to 1.0. The original inverse-chart
guard correctly rejects the endpoint under the locked protocol. The dead
worker did not serialize its fit; this is a reproduction, not a comparison
with its lost memory. The first eight completed rounds and all nine Adam
updates were independently checked on CPU, with no claim of a complete fit
or an independent decoder Jacobian. No GT or new DINO observation was used.

The finite-logit endpoint representation is a concrete DRAFT, not a deployed
repair or a new qualification. It changes the protocol's endpoint-failure
behavior and is awaiting explicit human authorization. Portable CPU contracts:

```bash
python -B scripts/test_stvg_opd_boundary_chart_proposal003.py
```

The preserved_initial_integrity_auditor.py file is an exact archived source
snapshot, not an executable entrypoint from this results directory. The current
integrity auditor writes immutable receipts in the private research workspace.
Its synthetic contract checker also writes a new private receipt; it is not a
GT evaluation or a model run.

Private frames, captions, annotations, source weights, raw predictions, live
optimizer/gradient tensors and the reproduced binary fit are excluded. Scalar
numerical diagnostics and aggregate byte digests are provided instead. The
reproduced GPU fit was not separately timed; no total recovery GPU time or
completed-run throughput is inferred. EATA remains explicitly user paused;
P1 scientific completion and P2-P6 remain pending.
