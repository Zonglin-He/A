# Actual saved Tent/SAR optimization diagnostics: hc2

Complete existing full-domain runs, 1902 queries. No new model, optimizer or GT scoring. Every saved native/adapted logit and receipt SHA was checked during derivation; independent audit checks all aggregate means, principal CIs and exact update/interval counts against the original full-domain audit.

|Method|Entropy before|Entropy after|Delta entropy [95% CI]|Mean saved parameter-update L2|Mean logit-change L2|Changed native intervals % [95% CI]|
|---|---:|---:|---|---:|---:|---|
|Tent|2.571947549|2.571703821|-0.000244 [-0.000254, -0.000234]|0.000536065|0.008001565|+0.157729 [+0.000000, +0.354974]|
|SAR|2.571947549|2.571742196|-0.000205 [-0.000216, -0.000195]|0.000492724|0.007268987|+0.105152 [+0.000000, +0.268244]|

|Method|Native delta t pp [95% CI]|Native delta v pp [95% CI]|Matched joint-MAP delta t pp [95% CI]|Matched joint-MAP delta v pp [95% CI]|
|---|---|---|---|---|
|Tent|+0.003824 [+0.000000, +0.012258]|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|
|SAR|+0.003824 [+0.000000, +0.012258]|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|

Entropy uses the four offset/endpoint distributions, before adaptation and saved logits from a real final forward AFTER all five updates. The last pre-update training trace is not substituted for the actual after state. Parameters are temporal-decoder LayerNorm affine only; saved state-delta L2 is the original actual trace. Source boxes stay fixed, episodic resets are exact. Centered logit changes and per-query entropy/update details are retained in all_saved_optimizer_diagnostics.tsv. SAR may have no reliable samples on some steps; original audited first/second mask counts remain authoritative.

Changes to continuous distributions/parameters do not imply an argmax interval change. Tiny or null localization effects on this adapter cannot establish a universal failure of Tent/SAR, canonical BatchNorm/full-network or continual adaptation. Joint-MAP versus native readout gains must not be attributed to the optimizer: each adapter is compared to Source with the SAME readout. Current method uses eight expert observations and zero optimizer steps; these baselines use zero expert observations and five optimizer iterations. Cached decoder/adaptation runtime is not matched end-to-end deployment latency; original acquisition costs remain nonzero.

Query-equal statistics and 10000 parent-cluster pointwise95%CIs; no familywise or unseen-data claim, and no full-target retuning. All signed per-query changes and original failures/costs remain in the old and new reports.
