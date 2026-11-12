# R80 fixed uniform physical-recovery TTA extension

Claim: R76 e6 one-step uniform SGD helps grounding beyond its frozen model and same-observation direct averaging. No query weighting, source training or new architecture. Known issue: DEV336 +0.348480pp is exposed and not confirmed.

Use fixed R69 VAL1280/TRAIN1024 rosters, same A256 and verified geometry; supplement uniform wrong-binding control. Reuse DEV336 completed core endpoints and source128 diagnostics. All predictions sealed and independently replayed before saved-IoU join. Preserve all negatives, failures and recording-level results. No CI based on falsely independent frames.

Primary evidence is paired task gain per cohort; no requirement to outrank R59. Wrong control shares physical targets but corrupts logical bindings. Correct correspondences need not make control collapse. Noncomplementary/single-modality ablation is conditional on positive gain vs own frozen and direct3 in both extension cohorts. No tuning or FINAL access.
