# 3EED official frozen full validation baseline

See [report](REPORT.md), [qualitative review without private media](QUALITATIVE_REVIEW.md),
[runner schema](crossplatform/round5/baseline_engineering/SCHEMA.md), and scalar
results under results/. Original model is frozen: no training or TTA updates.

All available official VAL expressions are forwarded under B1 deterministic
sampling, with full text and a text-derived first noun on identical points.
Official GT-class-assisted eligible scores and all-raw text-only scores are
separate. Axis-aligned source boxes are the primary; GT geometry replacements
are post-seal offline diagnostics, never deployable methods. Author legacy
seven-head mIoU and last-head mIoU are separate. Float32 original evaluator
and independently audited float64 geometry are distinguished.

This evidence export requires legally accessible official data, weights,
upstream3EED at the recorded commit, and the existing compatibility environment.
Private media, raw captions/labels, model weights, points, embeddings and input
path mappings are excluded. Reproduction must create a new registration with
new local paths, not rewrite these historical immutable locks. The relative
path launch failure and independent-auditor dtype v1 are retained in provenance.
Per-query scalar CSV retains every positive and negative language delta; scene
identifiers are hashed. It contains no annotation coordinates or captions.
