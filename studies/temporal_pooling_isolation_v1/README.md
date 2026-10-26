# Temporal pooling region isolation

The latest direct user request isolates the temporal evidence region while fixing
the entire spatial candidate construction and final decoding parameters. The two
arms use original Source boxes and shared-corrected boxes, respectively. Each arm
fits its own centered/normalized event and background prototypes, updates the same
native prior, selects its own legal joint-MAP interval, and runs the same final DP
on that interval. Same eight experts, native pretext features, context scale1.5,
kappa0/beta1 and one update. No first auxiliary DP or method/default changes.

The exact experiment already exists as **AuxSourceROI vs FullMAP**, with the MAIN
geometry retained in both, in the complete selected structural experiments. It
must not be confused with NoSharedCorrection, which also replaces spatial geometry.
Both complete target domains have official scores and independent all-array/SHA
audits. This namespace performs a fresh full two-arm numerical-chain verification
and dedicated scalar/paired-CI export, reusing those exact official scores rather
than spending new Source/expert/GT calls.

Private frozen science plan is `outputs/temporal_pooling_isolation_v1/plan.json`.
Public protocol omits private input paths. Four actual retained grids verify
Source/E pooling by the same `extract` function, repeated prototypes/posteriors,
their own intervals and final DP. Full-domain verification uses the actual
audited pooled sufficient features; no claim that previously disposed raw grids
remain available.

`run.py` is the preserved original version. Its first four-case preflight stopped
after both computed arms already matched the historical predictions because it
hashed a Python metadata dictionary via NumPy object-pointer bytes. No model,
Source, expert, GT or full-domain worker was started. Independently frozen
`runtime_v2.py` replaces only metadata hashing with canonical dictionary content;
all numerical computation and old science files remain unchanged. Original
source/plan/diagnosis are preserved privately; failure wall-clock was not recorded
and is not invented as zero.

Commands use the registered original Python runtime, hidden CUDA, CPU14-15,
nice19/idle I/O and one thread:

```bash
python -B studies/temporal_pooling_isolation_v1/runtime_v2.py preflight
python -B studies/temporal_pooling_isolation_v1/runtime_v2.py audit --dataset hc2
python -B studies/temporal_pooling_isolation_v1/runtime_v2.py audit --dataset vidstg
python -B studies/temporal_pooling_isolation_v1/runtime_v2.py export --dataset hc2
python -B studies/temporal_pooling_isolation_v1/runtime_v2.py export --dataset vidstg
```

Do not rerun a completed audit: each owns a unique worker/receipt namespace. Root
exports require the corresponding worker's successful completion/actual exit.
Primary tIoU, vIoU, R.3/R.5, query-equal paired parent-video cluster10000bootstrap
pointwise95CI, short/long splits, all signed losses/rescues and severe t/v union
are retained. sIoU stays an internal check outside the compact table. Both target
domains were historically exposed; no tuning or independent-confirmation claim.

Current CG GPU and gamma queue are untouched. Corruption remains paused by the
user. No manuscript/Fig1 edits, new backbone, new experts or optimizer updates.
