# K16 comparison adapter type failure

CUDA visibility was restored correctly. On the first query, original Source/ERef/
Global6/native time/eight-position outputs passed the strict official comparison.
The additional fixed K16 comparison then received a NumPy auxiliary trajectory,
while its existing frozen helper requires a CPU tensor. It failed at `.clone()`.
There were zero saved candidate predictions and no new label evaluation.

Original source, plan, log and failure state are retained. Worker553721/start228286197
exited and the shared GPU lock was verified free before V3. V3 has a separate plan
and fingerprint, asserts the exact same64 selected inputs, and only converts the
same auxiliary values back to a CPU tensor for the K16 comparison. Raw Source
captures are saved before diagnostic readout. The primary algorithm and all
numerical/protocol choices are unchanged. No completed prediction is rerun,
no old hash is waived, and K16 full remains paused.
