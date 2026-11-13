# R95 bounded repair

R94's secondary solver hit a fixed 5%-weight-norm cap on all 336 inputs and implemented only approximately 16% of the requested logit correction. R95 fixes this execution limitation without changing the sealed dense binding-contrast target, underlying R91 architecture, source model, or A256.

Before new task scoring, eight fixed real inputs passed independent native replay of all 439 attempted states and 144 endpoint readouts, including an unfitted observation. The 336-input protocol and code hashes were then locked. The primary is `lm_FC`; factual-only fitting and held-observation-only prediction remain secondary diagnostics. All 26 arms and all ordered negative comparisons are retained. No task GT determines targets, step acceptance, stopping, weights, or outputs. All inputs are historically exposed development data; no independent-confirmation claim is made.

The main LM procedure uses ten relinearizations, six damping trials per iteration, and an actual/predicted decrease threshold of 0.1. A source-mixture target, shuffled target, and per-step 5%-norm cap provide matched controls. The same sealed R94 target is reused; no external visual model, source training, or FINAL access is introduced.
