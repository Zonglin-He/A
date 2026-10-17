# Qwen3.5 versus R1, unchanged B3 temporal adaptation32

Same exposed HC16/Vid16, exact original B3 two hypothesis intervals and marked8frame NPY observations. Qwen3.5-9B BF16 replaces frozen R1 only. Same caption text task, original A/B/C support/contradiction/uncertain alignment, r=P(A)-P(B),160token cap, candidate prior/target/likelihood merged interval masses, tau1/full-grid trust1/512parameter originalLBFGS50/Armijo, actual FP32 head reinsertion and complete marginal MAP. Native non-thinking Qwen processor/template are architecture-specific, so this compares the whole caption+judge Teacher interface, not only the text judge. No new Actionness, hypotheses, Prompt or parameter search, no Spatial updates or deployment change. All outputs sealed before GT.

|Domain|Arm|vIoU %|tIoU %|dense sIoU %|
|---|---|---:|---:|---:|
|hc2|Frozen|28.116723|52.754115|50.497117|
|hc2|MARGINAL_NO_UPDATE|24.576445|48.654998|50.497117|
|hc2|R1_B3|24.713955|49.022568|50.497117|
|hc2|QWEN_B3|24.390435|48.521602|50.497117|
|hc2|QWEN_TARGET_HARD|24.338231|48.455425|50.497117|
|vidstg|Frozen|12.185607|35.923576|32.870500|
|vidstg|MARGINAL_NO_UPDATE|11.914150|35.236262|32.870500|
|vidstg|R1_B3|11.925210|35.215982|32.870500|
|vidstg|QWEN_B3|11.914150|35.236262|32.870500|
|vidstg|QWEN_TARGET_HARD|11.914150|35.236262|32.870500|

All-source paired t95/negative tails/unknowns and original R1/Qwen rewards/posteriors, allowed-label mass, target changes, fit KL and steps are in RESULTS.json. No-information sources retain Source marginal MAP and remain in the denominator. Confidence is not calibrated accuracy; GT interval-order diagnostics are not new semantic labels. Existing semantic panel remains44 model-assisted/468pending. No independent confirmation or cross-backbone evidence. No mixed Full is automatically launched.

hc2: informative Qwen 16/16 versus R1 16/16; uncertain posterior mean 0.6806856361217797; accepted optimizer steps 800. These are interface/transfer diagnostics, not semantic correctness.

vidstg: informative Qwen 14/16 versus R1 13/16; uncertain posterior mean 0.6822284234804101; accepted optimizer steps 700. These are interface/transfer diagnostics, not semantic correctness.


Processor correction002: Qwen3-VL interprets longest_edge as total video pixels. Copying the R1 per-frame448^2 limit had resized8frames to128 each (64 vision tokens/clip). That pilot remains preserved and its calls/cost included. Correcting total budget to8*448^2 preserves8x448, actual native grid[4,28,28]/784vision tokens verified before and during every model request. Hypotheses/NPY/clock/Prompt/reward/Student unchanged; this is an input contract correction, not an outcome retry or resolution sweep. Original4h deadline inherited. Executable entry is stvg_qwen_b3_temporal32_448_v2.py.


Current numerical conclusion: Qwen temporal adaptation fails to add a positive mean over Source marginal MAP (-0.186010pp HC/0pp Vid). Paired source intervals and every negative/neutral source remain. Uncertainty did not drop. GT interval-order proxy agreement is8/16HC and7/12 comparable Vid for Qwen, versus8/16 and4/10 for R1; different validity/tie sets and small counts prevent interpreting these as independent semantic accuracy or a sufficient quality guarantee. The method-lock shorthand "rewards unchanged" means the reward formula/task only, not identical numerical rewards after correcting pixels; the original private lock is preserved and explicitly clarified.
