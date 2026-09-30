# Paper48 P1: completed 670-source main evaluation

The frozen method improves over Frozen on the complete locked P1 panel. Under the five5% deployment corruptions, source-macro dense m_vIoU rises from21.0217% to21.4929%, a gain of **+0.4712 percentage points** (source-bootstrap95% CI **[+0.1905,+0.7756]**). The two fixed orders have gains +0.2720 and +0.6704pp. This is a completed result; P2–P5 remain separate queued experiments.

On **future arrivals with no current expert call**, the corruption gain is **+0.3787pp [ +0.3100,+0.4523 ]**. This subset contains5020 cells and626 distinct sources:44 of the670 sources are expert-scheduled in both fixed orders. The two order-specific gains are +0.3580/+0.4103pp. The final aggregate first averages each source over its eligible conditions and orders, then weights sources equally; consequently it need not equal the arithmetic mean of those two subset order means. Temporal IoU is unchanged on these arrivals, while dense spatial IoU improves by +0.8101pp. Current specialist reranking cannot account for an improvement on arrivals without a current call; this supports useful inherited spatial state under this fixed protocol, without establishing the ranking-loss mechanism by itself.

| Condition/readout | Frozen m_vIoU % | Ours m_vIoU % | Paired gain pp | Source-bootstrap95% CI pp |
|---|---:|---:|---:|---|
| Clean, all |21.7122|22.1230|+0.4108|[+0.1100,+0.7429]|
| Corruption, all |21.0217|21.4929|+0.4712|[+0.1905,+0.7756]|
| Clean, nonexpert |21.9154|22.2998|+0.3845|[+0.2950,+0.4860]|
| Corruption, nonexpert |21.1727|21.5514|+0.3787|[+0.3100,+0.4523]|

All five corruption-family mean gains are positive, both for the full stream and the nonexpert readout. The paired **corruption-minus-clean gain is +0.0604pp, CI[-0.0454,+0.1665]**. These results establish improvement under corruption; they do not establish a gain specific to corruption, since clean also improves and the excess interval crosses zero. The independent clean streams do not measure clean retention from a corrupted terminal state.

|5% family|All-stream gain pp|Nonexpert gain pp|
|---|---:|---:|
|Frame drop|+0.5314|+0.2962|
|Frame freeze|+0.4331|+0.4109|
|Motion blur|+0.4730|+0.3998|
|Occlusion|+0.4267|+0.3700|
|Exposure|+0.4919|+0.4163|

Negative tails are retained. For corruption over the full stream,15/670 source means lose more than5pp and2 lose more than20pp; at the cell level the counts are210/6700 and44/6700. For corruption nonexpert arrivals, no source mean loses more than5pp, but11/5020 cells lose more than5pp and1 loses more than20pp. Positive averages do not imply every arrival improves. Only two prespecified orders were run; source bootstrap intervals are conditional on those orders, not a many-seed robustness estimate.

The panel uses670 official VidSTG-test sources outside this TA-STVG route's development-source union, one query/source by hash, two fixed orders, six independent streams per order (clean and five5% corruptions),8040 arrivals. Historical project exposure remains disclosed. The Vid-trained checkpoint is5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83. The J0.1 method remains1792 spatial parameters, SGD.005/one step,9 current-student probes, relative radius.05, every fourth arrival has two specialists, current spatial output before update. State persists within each670-query stream and resets between conditions/orders. No method or deployment registration was changed after these outcomes.

All8040 predictions sealed before GT scoring. The saved audit passes8040 state links,1763 SGD updates,2016 temporal critic checks and16080 comparisons between official dense evaluator functions and an independent geometry kernel, with zero maximum discrepancy. Root readback verifies all8040 receipt hashes and raw prediction-payload hashes, scalar identity alignment, and24 full spatial reinsertion endpoint receipts without new inference or GT scoring. An independent public-scalar auditor passes170971 checks, including source/order means,10000-resample bootstrap intervals, recalls, negative tails, paired clean/corruption excess and quartiles. Primary dense metrics are kept distinct from historical sampled-grid continuity metrics.

Online worker allocations total8419.322 seconds (2.339 hours), including the preserved user-requested execution-policy interruption; this excludes previously generated/reused expert evidence and is not uncached deployment latency. P4 measures the latter separately. P2 is continuing, followed by P3/P4 and required P5; no total deadline is enforced.

Public artifacts include anonymous scalar SCALARS.csv, all aggregates, original runtime AUDIT, root readback and public scalar audit. No captions, private media, GT boxes/spans, weights, raw prediction tubes or learned states are exported. Run `python scripts/audit_tastvg_paper48_public_v1.py results/tastvg_paper48/2026-09-30/P1` with NumPy to independently reproduce scalar checks.
