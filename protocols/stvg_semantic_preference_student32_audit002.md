# Root-only Adam numerical audit correction002

GEOM produced all32 sealed predictions (23 actual fits) before the original root
failed its fixed5e-6 absolute moment/variance comparison. No target GT was read.
Fixed-support atoms can give large likelihood gradients; native float32 moments
then legitimately round above that constant absolute threshold. Examples of
saved variance error were up to1.73e-4, within0.157 of an8-operation float32 bound.
Actual proposed parameter-delta error remained below the original5e-6 threshold.

No model or update is modified. Preserve original source/runtime/failure and all
Teacher/GEOM predictions. Replace only the independent root's Adam moment check:
float64 recurrence against actual native float32 moments, with gamma8=8u/(1-8u),
u=float32 eps, pointwise bounds gamma8*(|old_mean|+|gradient|) for native stable
lerp, gamma8*(beta2*|old_variance|+(1-beta2)*gradient^2) for multiply/addcmul/FMA,
plus the float32 tiny underflow bound. Require actual moment dtypefloat32, finite
nonnegative variance, exact step increment and parameter ordering. Keep the old
strict5e-6 absolute proposal-delta check using the independently verified moments.
A deliberate0.1%/at-least.001 variance mutation is rejected; the real saved GPU
state passes. This is operation-based float32 audit, not lower precision or an
arbitrary task/acceptance threshold change. Remaining full audit stays unchanged.

Continue at GEOM root only, then first PREF worker/root, D2, all-seal GT score.
No Teacher/GEOM model rerun, original2h deadline unchanged. Identity-checked Qwen
CPU handoff/resume is preserved. Recovery source/hashes and failed root wall are
published with actual results; no adverse measurement is deleted or rescored.
