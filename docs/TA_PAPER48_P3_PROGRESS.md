# Paper48 P3: completed 0% and25% budget arms;100% pending

The prespecified64-source panel has completed two of its three expert budgets. This is an interim publication of completed arms, not a completed budget curve. The remaining100% arm runs with the original frozen configuration; these results do not trigger tuning.

| Expert availability | Corruption whole-stream Δdense vIoU pp [95% source-bootstrap CI] | Corruption nonexpert Δdense vIoU pp [95% CI] |
|---|---|---|
|0%|0.0000 [0.0000,0.0000]|0.0000 [0.0000,0.0000]|
|25%|−0.0477 [−0.5050,+0.3118]|+0.0356 [−0.0006,+0.0746]|
|100%|Pending|No nonexpert arrivals by definition|

At0%, Ours equals Frozen exactly in the runtime prediction checks and all768 scored arrivals, with0 specialist calls and0 updates. This verifies the no-expert behavior, not an effectiveness gain.

At25%, the overall corruption mean is slightly negative and its interval crosses zero. The two order-specific whole-stream gains are−0.2355/+0.1402pp. The nonexpert readout has480 cells and61 sources, with order-specific gains+0.0214/+0.0565pp; its source-weighted overall CI still crosses zero. Clean whole-stream gain is−0.2466pp [−0.9099,+0.1663], and clean nonexpert gain is+0.0363pp [−0.0020,+0.0767]. Thus this64-source25% arm does not establish a positive overall or future-transfer effect.

The positive larger P1/P2 results and this smaller-panel result are all retained. P3 is a nested64-source subset with independent64-query state resets and restricted order/schedule; changing the subset also changes inherited-state histories. Cross-panel differences do not isolate stream length, and the100% comparison must finish before discussing the complete budget response. No global monotonic-budget claim is made.

At25% on corrupted arrivals,1/64 source means loses>5pp and none loses>20pp;8/640 cells lose>5pp and3 lose>20pp. No nonexpert cell loses>5pp. The full negative tails and both orders are public, along with clean and paired excess readouts.

Both completed arms use64 hash-selected sources/one query per source, two fixed orders, clean plus five5% corruption families,768 arrivals per arm. Vid-trained TA checkpoint5ab12c86, J0.1 spatial1792 parameters, SGD.005/one step,9 current probes, pre-update current output and persistent state within a stream remain fixed. Each arm and condition/order resets independently. At25%, both specialists are scheduled every fourth arrival. The sources overlap P1/P2 and have historical project exposure.

Each arm sealed all predictions before GT scoring. Runtime audits, root verification of all768 receipt/payload hashes and24 spatial reinsertion endpoints, and independent public scalar/source/bootstrap checks pass. Full precision results are in their respective directories. No new GPU inference or GT scoring is performed during publication auditing. All raw predictions, GT coordinates, media, weights and learned states remain private. P4 and required P5 still follow the budget arms; no total deadline is imposed.
