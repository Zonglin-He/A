# Paper48 P2: completed corruption severity panel

The frozen method has positive mean dense vIoU gains in all15 family/severity cells on the prespecified128-source subset. Across the five families and1/5/10% severities, overall gain is **+0.4706pp**, with source-bootstrap95% CI **[+0.0728,+0.8993]**. Future arrivals without a current specialist call gain **+0.1380pp [+0.0472,+0.2170]** (121 sources/2880 cells). This is a completed panel; subsequent budget and efficiency/HC2 panels remain separate.

| Corrupted physical frames | Frozen m_vIoU % | Ours m_vIoU % | All gain pp [95% CI] | Nonexpert gain pp [95% CI] |
|---|---:|---:|---|---|
| 1% | 19.7725 | 20.2505 | +0.4780 [+0.0747,+0.9073] | +0.1366 [+0.0351,+0.2229] |
| 5% | 19.2695 | 19.6694 | +0.4000 [-0.0231,+0.8400] | +0.1360 [+0.0447,+0.2146] |
| 10% | 18.9123 | 19.4461 | +0.5338 [+0.1097,+1.0152] | +0.1415 [+0.0599,+0.2186] |

The5% all-arrival CI crosses zero even though its mean is positive; the three nonexpert severity CIs are positive. The gain is not monotonic in corruption strength. The15 positive means do not establish that every family-level effect is statistically resolved.

Clean all-arrival gain is+0.4789pp [+0.0030,+0.9831]; clean nonexpert gain is+0.1351pp [+0.0170,+0.2300]. The paired corruption-minus-clean excess is−0.0082pp [−0.3048,+0.2698], again not evidence of corruption-specific improvement. Independent clean streams do not test retention after a corrupted stream.

Negative tails are retained: across3840 corrupted arrival cells,60 lose>5pp and17 lose>20pp; source-averaged counts are1/128 and0/128. For2880 nonexpert cells,2 lose>5pp and none lose>20pp; no nonexpert source mean loses>5pp. Two prespecified orders and source-level bootstrap are reported; there is no claim of many-seed robustness.

Configuration:128 sources from the original670-source hash roster, one query/source, two restricted orders, clean+five corruption families at1/5/10%,4096 arrivals. Vid-trained TA checkpoint5ab12c86, J0.1 fixed1792-parameter spatial SGD.005/one step,9 current probes,25% specialist schedule, pre-update current output. Every128-query stream resets independently. This subset overlaps P1 and has different inherited-state/schedule histories; it is neither an independent cohort nor a controlled stream-length ablation. Historical project exposure remains disclosed.

All4096 predictions sealed before GT. Runtime audit passes4096 state links,950 updates,1024 temporal critic checks and8192 official/independent dense metric comparisons, maximum discrepancy0. Root verifies every receipt and prediction-payload SHA, scalar identities,64 full spatial reinsertion endpoint receipts. Independent scalar reaggregation passes91799 checks including10000 source-bootstrap replicates and the added severity aggregates. The generalized public auditor also reproduces the already published P1. No new inference or GT scoring was used for this publication audit.

Online worker time3543.232 seconds (59.05minutes); expert evidence is reused, so this is not uncached deployment latency. All outputs, anonymous scalar rows, negative tails, original per-family aggregates and severity readouts are published. No labels/media/weights/raw tubes/states are exported. Frozen method and production registry remain unchanged.
