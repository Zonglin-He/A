# Source native actuation chain, v1

User attachment 4775cbaa requests a working source diagnostic before further OPD.
The old oracle is completed and preserved: correct support attenuation did not
show preferential utility over matched wrong support. This does not establish
universal latent failure or a confirmed suppression/context/weak-actuator cause.

## Fixed question and inputs

Use only existing exposed source PANEL16 records. The full16 CPU case/margin
audit is separate and uses old sealed predictions. The two explicitly
outcome-selected optimization cases are temporal19368 (sole correct-T interval
flip) and spatial28546 (largest correct-minus-wrong sIoU deficit). Spatial473
(largest spatial sIoU gain) stays in the existing-evidence good-case audit.
No replacement, extra label pool, target input, external teacher or OPD.

Official frozen PTD4B and exact B1, original observed pixels, physical frame IDs,
query, dtype, cached decoder and backbone weights. Model and all adapter
parameters stay frozen. Source GT supplies only the diagnostic loss and scoring;
semantic prefix/interval/anchors are generated natively, never forced from GT.

## Sequential factors

1. Free FP32 zero-initialized delta at the existing merger injection before the
   unchanged BF16 cast, added to the existing B1 updated_tokens for one branch.
   Same delta reused at every prefill of that branch; other branch untouched.
2. Only if both fixed step30 primary controls improve: same cases/loss/schedule,
   restrict delta to the frozen branch output-projection column span. QR basis
   Q times sqrt(C/d) puts coefficients in comparable token-RMS units. This is
   exactly a reparameterization of g W delta_z (nonzero stored gate), not a gate
   increase or weight change. It tests unrestricted span reachability, not the
   efficiency of small reader updates. Adam is not invariant to parameterization;
   a failed finite run cannot prove the span is mathematically insufficient.
3. Existing multiplicative oracle results are retained, no retraining. Compare
   its margins and native outputs with the two controls. Even successful free
   controls would not make GT-supervised high-dimensional updates comparable to
   a no-update mask or prove that attenuation alone is the root cause.
4. Earlier conditioning and context-preserving masks are conditional proposals;
   require a separate norm-matched protocol, not bundled with this first test.

Each case/mode uses fresh AdamW lr .01 in token-space units, wd0, clip1, betas
(.9,.999), eps1e-8, exactly30steps, seed20260927, no scheduler or best-step
selection. This is an intentionally strong supervised feasibility control,
not a continuation of the old 3-step66816 TTA comparison or a hyperparameter grid.
Record every native state0..30; only step30 is the registered final endpoint.

Each step captures the actual current native cache schedule and replays it
through the previously audited differentiable memory-v7 path. Event CE is the
mean over the two actual endpoints and all observed time classes. Spatial CE is
the mean over four coordinate positions and all1001 classes on current native
anchors with valid referent annotation, regardless of event activity. Unknown
boxes excluded from supervised loss and explicitly counted; native evaluation
still retains all frames and invalid zero boxes. Quantize normalized box coords
by round(1000*x), consistent with existing source serialization. GT endpoint
classes come from the registered source response, not newly optimized intervals.

Every replay must equal its native logits exactly before backward; fail closed
and save both on mismatch. Independent fresh branch prefills, no shared event KV.
Native semantic reference may change in event control; report it. Spatial pass
must preserve original reference/time/anchors. No full-tube KL claim.

## Evidence and decision

Save native outputs, complete restricted logits/trace/support, targets, every
gradient and actual Adam delta/counter, exact baseline replay, final parameter
and optimizer, frozen-state hashes. Seal before independent scalar/tensor
geometry scoring (source GT already exposed by diagnostic supervision).
Compare margins, CE, endpoint and coordinate argmax, s/t/v, injection cast changes,
and all intermediate outcomes without selecting a best state. Step30 temporal
success requires tIoU improvement and both GT endpoints as native argmax;
spatial success requires fixed-support sIoU improvement and lower same-support CE.
Do not call CE decrease alone tube improvement or assume fit is guaranteed.

GPU serial; each allocation3600s engineering limit, minimum8GiB disk, at most
10GiB new artifacts, memory-v7 16GiB offload/6GiB host reserve. All actual
imports/load/update/native/failure/finalization seconds accumulate cap=null via
latent_oracle_v1/*/RECEIPT.json. No silent reruns or scientific hot edits.

Scope controls: zero delta native identity, frozen PTD/adapter gradient absence,
CPU QR-span equivalence, loss denominator/missing support, hook restoration.
CPU checks are not actual GPU feasibility or utility.
