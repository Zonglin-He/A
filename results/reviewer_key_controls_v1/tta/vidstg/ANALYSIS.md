# Actual saved Tent/SAR optimization diagnostics: vidstg

Complete existing full-domain runs, 10303 queries. No new model, optimizer or GT scoring. Every saved native/adapted logit and receipt SHA was checked during derivation; independent audit checks all aggregate means, principal CIs and exact update/interval counts against the original full-domain audit.

|Method|Entropy before|Entropy after|Delta entropy [95% CI]|Mean saved parameter-update L2|Mean logit-change L2|Changed native intervals % [95% CI]|
|---|---:|---:|---|---:|---:|---|
|Tent|2.633853164|2.633493455|-0.000360 [-0.000384, -0.000336]|0.000576653|0.015604327|+0.067941 [+0.020196, +0.122149]|
|SAR|2.633853164|2.633518113|-0.000335 [-0.000358, -0.000313]|0.000573329|0.015052851|+0.048530 [+0.009832, +0.094331]|

|Method|Native delta t pp [95% CI]|Native delta v pp [95% CI]|Matched joint-MAP delta t pp [95% CI]|Matched joint-MAP delta v pp [95% CI]|
|---|---|---|---|---|
|Tent|-0.000384 [-0.001762, +0.000599]|-0.000260 [-0.000840, +0.000056]|-0.000204 [-0.002035, +0.001681]|+0.000349 [-0.000468, +0.001445]|
|SAR|-0.000098 [-0.000303, +0.000000]|-0.000038 [-0.000116, +0.000000]|-0.000009 [-0.001791, +0.001833]|+0.000349 [-0.000468, +0.001445]|

Entropy uses the four offset/endpoint distributions, before adaptation and saved logits from a real final forward AFTER all five updates. The last pre-update training trace is not substituted for the actual after state. Parameters are temporal-decoder LayerNorm affine only; saved state-delta L2 is the original actual trace. Source boxes stay fixed, episodic resets are exact. Centered logit changes and per-query entropy/update details are retained in all_saved_optimizer_diagnostics.tsv. SAR may have no reliable samples on some steps; original audited first/second mask counts remain authoritative.

Changes to continuous distributions/parameters do not imply an argmax interval change. Tiny or null localization effects on this adapter cannot establish a universal failure of Tent/SAR, canonical BatchNorm/full-network or continual adaptation. Joint-MAP versus native readout gains must not be attributed to the optimizer: each adapter is compared to Source with the SAME readout. Current method uses eight expert observations and zero optimizer steps; these baselines use zero expert observations and five optimizer iterations. Cached decoder/adaptation runtime is not matched end-to-end deployment latency; original acquisition costs remain nonzero.

Query-equal statistics and 10000 parent-cluster pointwise95%CIs; no familywise or unseen-data claim, and no full-target retuning. All signed per-query changes and original failures/costs remain in the old and new reports.
