# O1: Sparse-Critic Online Transfer

This fixed first implementation does not show a positive non-expert transfer gain. The result is specific to raw temporal features, LR .001, eight one-step writes and this stream; it does not identify representation failure or establish that transferable critic information is absent.

32 previously exposed VidSTG development sources, one query and one source-hashed existing transient condition each, one separately hashed order. Expert positions: 1,5,9,13,17,21,25,29. The primary comparison is on the other 24 arrivals, where neither Budgeted Rerank nor Online Slow-Fast reads the current expert.

| Arm | Logical expert calls | Non-expert tIoU (%) | Non-expert vIoU (%) | All32 tIoU (%) | All32 vIoU (%) |
|---|---:|---:|---:|---:|---:|
| Frozen | 0 | 46.2867 | 17.9538 | 45.0910 | 16.2558 |
| Budgeted Rerank | 8 | 46.2867 | 17.9538 | 43.9290 | 16.8264 |
| Online Slow-Fast | 8 | 46.2867 | 17.9538 | 43.9290 | 16.8264 |
| Full Rerank | 32 | 50.0614 | 18.6286 | 46.7600 | 17.3325 |

## Primary: inherited state at non-expert arrivals

Online minus Budgeted Rerank: tIoU +0.0000 [+0.0000, +0.0000] pp; vIoU +0.0000 [+0.0000, +0.0000] pp.

**The intervals are descriptive paired bootstraps of sealed arrival outcomes, conditional on this one realized stream. They do not rerun the online trajectory and do not estimate robustness over stream orders or independent online runs.** There are 24 unique non-expert sources but their predictions share an inherited state.

Only 1/24 non-expert selections differ from Frozen. vIoU improves on 0, worsens on 0, and is unchanged on 24. >5pp vIoU harms: 0.

| Changed non-expert arrival | Parent | Condition | Frozen/Budgeted vIoU (%) | Online vIoU (%) | Full Rerank vIoU (%) |
|---|---|---|---:|---:|---:|
| 6 | Q21 | frame_freeze_5 | 0.0000 | 0.0000 | 0.0000 |

## What was adapted

Only a zero-initialized 768D weight and a common scalar bias persist. Raw features concatenate the final temporal hidden state at candidate start/end and its within-interval mean. The native candidate base score is the exact maximum legal two-offset log-probability score consistent with that physical envelope; zero residual reproduces native selection on 32/32 arrivals. No hand-added native bonus or score calibration.

At each expert arrival, the current output is direct expert reranking and is fixed BEFORE the update. One plain SGD step (LR .001, no momentum or weight decay) fits all strict expert preference pairs. The common bias cancels and stays exactly zero. No H, temporal head, decoder or backbone parameters are changed. No normalization tuning, gate, EMA, replay buffer, prototype, low rank, extra step, learning-rate sweep or extra stream.

All 8/8 expert updates decreased their pairwise training loss. Final weight norm is 0.01139664. Loss reduction on expert arrivals is not itself evidence of future-query task improvement.

The 8 expert outputs are shared exactly between Budgeted Rerank and Online. Online output/state was sealed and independently reconstructed before reading/computing the additional 24 Full Rerank scores. Full Rerank is a higher-cost reference, not an oracle or guaranteed performance ceiling. All four arms were sealed before retaining the original 32 GT records for offline scoring.

## Chronology, accounting and limits

STATE_AUDIT.json reconstructs all32 arrival states and eight SGD writes in NumPy, verifies prior-state and record hashes, and confirms zero teacher reads at non-expert arrivals. UTILITY_AUDIT.json checks two task metric implementations, all expert scores and matched controls. PUBLIC_AUDIT.json independently rebuilds published aggregates and chronology. Source/condition hashes never use GT, query or outcome.

PROGRESSION.json reports eight chronological blocks and cumulative non-expert differences. Blocks have different sources; a rising or falling curve alone does not identify state pollution. No margin write gate or prototype was added. Spatial and C0.6 remain unrun. Production CURRENT is unchanged.

Logical online expert budget is 8/32 (25%). Actual online evidence: 4 new, 4 reused exactly; full-reference extras: 12 new, 12 reused. These cache-reuse savings do not change the logical expert budgets or establish fresh-input end-to-end latency.

GPU process allocation (including loading, decoding and smoke): 112.706s; stage times {"capture": 47.77418395198765, "teacher_online": 30.04820169499726, "teacher_full": 34.8837103539845}. CPU online loop: 0.050s. No run failures. CPU development/scoring and historical cache creation are excluded.

This is a fixed first implementation on an exposed 32-source development stream. A small number of changed predictions limits what can be inferred about transferable representations. Negative results do not isolate feature quality, learning scale and discrete candidate readout; positive results do not establish broad generalization. No follow-on experiment was launched.

Reproduction: protocols/tastvg_sparse_online_o1_v1.md; capture -> teacher online -> online stream -> state audit -> teacher full -> seal full -> analysis. Public scalar reproduction uses scripts/audit_tastvg_sparse_online_public_v1.py on this result directory. Raw media, labels, hidden features, learned state tensors and predictions remain local.
