# Joint component attribution: local complementarity, native causality unresolved

Completed / independently audited, CPU only. Same frozen B1/PTD4B saved PANEL16
shared-F gradients and Joint directions; all16 exposed Vid training parents.
No new model/forward/backward/optimizer, target data, or label pool. Old source
supervision remains in the stored gradients. GEOMETRY also contains previously
saved source-action targets deserialized by torch.load but unused by this audit.

The principal-method hypothesis "correction must be decomposed" is withdrawn.
The current candidate is **Decomposed Evidence, Joint Correction**. A Joint
mixer, TVG/SVG evidence providers, and joint OPD remain unimplemented/unqualified.

## Subspaces are independent but not orthogonal

T and S each have128 dimensions; their union has256 and the numerical
intersection is0. Principal angles range30.877080–89.985328 degrees, median
78.297775 degrees; direct-sum condition number3.620962. Independence does not
mean orthogonality. Naively adding PT J and PS J has relative reconstruction
error53.1292–70.3068%. This is double allocation of nonorthogonal directions,
not an unmodelled component or an old inference bug.

Both exact orthogonal orderings were retained:

- T-first: PT J + (PU−PT) J + (I−PU) J.
- S-first: PS J + (PU−PS) J + (I−PU) J.

The added S-given-T/T-given-S spaces are residualized spaces, not pure original
S/T components. The order-independent direct sum QT cT + QS cS was additionally
solved; because the intersection is0, original-span membership is unique here,
but energies require the signed cross term. All reconstructions include the
small outside-union residual (maximum relative L2 4.203e−8).

## Added span capacity and local contributions

All entries below are parent-equal means of per-query fractions. Positive means
predicted CE descent at the saved initial point, not observed finite task gain.

|Question|Mean fraction|Range|Positive parents|
|---|---:|---:|---:|
|Additional union gradient energy beyond T's own span, relative to union gT|10.7638%|7.2339–14.2383%|16/16|
|Additional union gradient energy beyond S's own span, relative to union gS|35.0432%|26.7858–41.2767%|16/16|
|S-given-T contribution to Joint's local T descent|10.5877%|6.7994–14.6777%|16/16|
|T-given-S contribution to Joint's local S descent|34.0970%|24.7294–41.3403%|16/16|
|Original S-span component contribution to local T descent (direct sum)|24.3973%|17.8706–30.0532%|16/16|
|Original T-span component contribution to local S descent (direct sum)|60.7913%|39.5353–72.4929%|16/16|

The union contains49.5584% of full gT squared energy and35.2535% of full gS
squared energy on average. The fractions in the first two rows are relative to
those union energies, not to the full gradients. Do not mix the two denominators.
Both orders and direct-sum conventions have positive self and cross-component
local dots16/16; magnitudes/ownership differ. Stored J_pass agrees up to recorded
FP32 rounding, and is not an independent sample of evidence.

## Subspace origin is not task-gradient origin

Joint explicitly combines normalized supervised gT and gS inside the union.
Splitting by gradient INPUT instead of by output SPAN produces a different result:

|Gradient input -> objective|Local-positive / negative parents|Mean local descent|
|---|---:|---:|
|gT input -> T|16 / 0|1045.486531|
|gS input -> T|9 / 7|−22.703664|
|gT input -> S|9 / 7|+0.318450|
|gS input -> S|16 / 0|79.082834|

Thus the all-positive cross-SPAN contributions do NOT mean another task's
supervision always helps. A spatial-span component can carry temporal-gradient
information. Each task gradient was already supplied to the Joint oracle by
construction; this is not independent evidence for external-expert correctness,
necessary two-expert use, or learnable cross-task compensation.

## What this does and does not explain

The saved Joint exploits locally useful directions unavailable to the objective's
own128-dimensional span. This supports local complementary capacity as one
plausible contributor. It does not isolate the cause of the prior native gain:
union access, gradient balancing, direction allocation and shared application
changed together. The original finite correction also changes later temporal
conditioning and goes through nonlinear/BF16/readout operations.

Old whole-correction evidence is unchanged: Decomposed−Joint Δv−4.401181pp,
95% descriptive CI[−10.411040,−.404300]; versus pass-matched Joint−5.941072pp
[−12.342213,−1.730258]. Joint's own advantage over Base has CI crossing0.
All96 old predictions, format failures, severe losses, positive cases, original-
good retention and fixed-support CE are retained at the original result entry.
The machine-readable component report includes inherited signed native metrics
for every P00–P15, without re-evaluation or new outcomes.

**No isolated-component finite/native forward has been run.** Therefore we cannot
say how much of Joint's t/s/v improvement was CAUSED by its T or S component.
The next causal gap is a locked finite component intervention at the same native
support, not a new mixer or an amplitude search. That intervention has not been
registered or launched in this bounded CPU audit.

## Validation and resources

4 synthetic CPU contracts (nonorthogonal/intersection/zero/cross-sign/rotation
and energy cases) passed.48 consumed old episode files are checked against their
original seal; basis and original report have separate verified seals. NumPy
FP64 SVD/union-coordinate attribution is independently checked with Torch FP64
QR/pseudoinverse/full2560-dimensional projection:1570 scalar checks, maximum
absolute difference6.00413e−8 under atol1e−8+rtol2e−7. The largest absolute error
is an energy-scale reduction, not a new predictive uncertainty estimate.

Primary CPU21.382100s (prior input hash precheck excluded); independent
readback19.332128s (its input hash checks included). New GPU0; cumulative GPU remains
43107.875212573104s, cap=null. No source tensors copied to public results.
No old raw/predictions/checkpoints were deleted. Monitor remains deleted because
there is no active GPU/download stage. Full-source/scalar/external/target queues
remain stopped or deferred.
