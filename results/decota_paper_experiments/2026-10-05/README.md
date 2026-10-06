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

## 2026-10-06 actual Ours global seal and serial continuation

Both clean cross-domain Ours directions now completed all three fixed orders:
HC2 validation10446 and VidSTG test30909,41355 arrivals. Source Only and the
shared DINO-Refine control have been derived and sealed (13785 unique payloads,
41355 logical arrivals each). The existing serial continuation actually started
target-trained frozen reference inference after the Ours controller exited.
No duplicate GPU controller was launched.

OURS_CONTROLS_ROOT_BYTE_READBACK.json verifies all55140 corresponding files:
Ours1351776038 bytes plus stateless44453740 bytes, SHA256, exact coverage and
receipt size/runtime/seal times. Four synthetic auditor tests pass. This is
stage metadata and opaque-byte verification only; it reads neither GT nor
prediction arrays, and publishes no efficacy score. State/optimizer/dense
scoring and GT pipeline diagnosis remain required after the complete Table1
all-deployable-arm barrier. TENT/EATA/SAR live qualification and predictions
are not claimed complete by this receipt. HC source Fisher media and the HC
Table2 parent-movie/clip-unit answer remain unresolved.

This is not all Table1 or paper completion. No method retuning or promotion,
old queue resumption, or GT-selected prediction is authorized by this handoff.

