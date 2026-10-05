# Frozen DeCoTA paper experiments: registered and running

These files record the actual new formal evaluation scope and engineering
checks. They contain no efficacy scores, target labels, raw media, predictions
or private caches. Ours VidSTG-source -> HC2 completed its first clean cross-direction job seal
(3482 official validation queries, three orders, 10446 arrivals). Root verified
all file bytes/receipts without unpacking predictions or reading GT. The existing
controller now runs HC2-source -> VidSTG; baseline continuation still waits for
both Ours directions. Every requested remaining arm/table stays tracked.

See [execution](../../../docs/DECOTA_PAPER_EXPERIMENTS_EXECUTION.md),
[protocol](../../../protocols/decota_paper_experiments_v1.md) and
[baseline port protocol](../../../protocols/decota_paper_baselines_20261005_v1.md).
`OURS_SMOKE_SUMMARY` is four real-query no-GT replay/math qualification.
`PORT_CPU_CONTRACTS` contains 32 synthetic CPU checks, not live baseline
qualification or performance. Registration is a dated snapshot, not a
self-updating claim that future stages already executed.

`HC2_OURS_STAGE_SEAL_READBACK` is only a byte/hash, receipt-binding and seal-time
check. No efficacy results or complete independent state/math/dense audit are
claimed. GT scoring waits for all Table1 deployment arms to seal.

The original combined smoke leaked HC output-timing decoder binding into its
two Vid inputs. The formal fresh Vid worker used canonical original-frame
decoding and correctly stopped on cached-pixel/signature mismatch. Revision001
locally isolates decoder routing/cache scope; all2283 saved formal predictions
remain unchanged and the original queue has resumed. Four actual no-GT query
requalifications and CPU decoder contracts passed. Original wrong inputs and
failure evidence are retained privately; the current smoke summary explicitly
supersedes those two original Vid input qualifications. No efficacy scoring.
