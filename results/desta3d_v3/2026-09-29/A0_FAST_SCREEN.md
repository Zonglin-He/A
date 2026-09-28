# A0 fast-screen completed

Dev direction median is below0.1 and train below0.3. The fixed A0 does not fit the oracle directions sufficiently at this budget; capacity/optimization is the next screening hypothesis, not an established cause. No native inference or expansion was run.

|Set|Queries / parents|Defined directions|Mean cosine|Median cosine|
|---|---:|---:|---:|---:|
|train|128 / 95|128|0.029865917|0.007693602|
|dev|64 / 16|64|0.024222450|0.002401304|

Existing StateAwareDirectionMixer hidden128/state33/union256 and fixed radius. One seed,200 actual AdamW steps,800 query occurrences, no best-step/CE/KL/gate. Source GT supplies privilege and analytic oracle targets. Dev is exposed diagnosis; fresh31/388 and target remain untouched. CPU controls do not establish native utility.

Measured GPU allocation and wrapper seconds: 1813.737310228; cumulative 72405.619553776, cap=null. Prior failures remain counted.

Neither screening classification nor a development pass proves a general mechanism, fresh generalization, teacher qualification or target TTA. All missing/invalid cases and negative outcomes remain retained.

The registered-to-completion pipeline took1908.29seconds (31.8minutes). Cache1794.69seconds; cached fit11.95seconds; wrappers7.09seconds. Full cache CPU audit63.69seconds and terminal cosine audit1.57seconds are separately measured. All8 trainable tensors changed, gradients were finite and nonzero, clip0, frozen union unchanged and integer Adam/live counters200 verified. This excludes a completely disconnected update, but does not distinguish limited capacity from optimization/conditioning at this fixed budget.

The only proposed next test is internal mixer width128→256 while keeping frozen feature width128, state33, cache, loss, radius and200steps unchanged. The current constructor ties input-feature width to hidden width, so a future capacity implementation must preserve its input dimensionality. This variant has not been registered or trained. No native A0 metrics are claimed.
