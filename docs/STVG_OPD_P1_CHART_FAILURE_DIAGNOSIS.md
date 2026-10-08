# P1 native-box chart failure: actual diagnosis, continuation decision pending

The fixed VidSTG P1 run failed after 1,882 receipted arrivals in its first order.
The first missing arrival, 1882 (zero based), is query ordinal 10220. The failure
occurs after the ninth Adam update, before an accepted output for that query.
The original controller and worker exited. The previously sealed HC2 direction
is retained; the global P1 deployment barrier remains absent and no P1 GT was
opened. This is a numerical failure report, not an effectiveness result.

## Actual reproduced cause

One admitted frame's height coordinate has a finite native bbox-head logit of
16.950244903564453. Applying sigmoid in float64 yields 0.9999999564886907;
the original float32 sigmoid output is exactly 1.0. The original inverse-sigmoid
chart rejects this endpoint. Boxes, native logits and parameter tensors are
finite, so the observed failure is not a NaN or an OOM. No box was clipped or
substituted, no query was skipped, and no LR or other selected parameter changed.

The complete 1,882-output VidSTG prefix contains 547,699,038 opaque bytes. Each
payload and original JSON receipt was independently hash/byte/runtime verified.
Original code, locks, config, logs and tracebacks are preserved. One GPU replay
reconstructed the full committed prefix and reproduced the first missing fit.
A return-neutral bbox-head hook captured the original pre-sigmoid outputs; an
exception hook serialized live parameter state, Adam moments and the partial
fit before restoration. There were zero new DINO calls and no accepted new
prediction. The original dead worker did not serialize its failing fit; this
capture is a reproduction and does not establish equality with dead memory.

CPU readback independently verified the first eight completed rounds' Gaussian
gradient, reward, Adam and state arithmetic. Maximum errors were 1.362493e-7,
3.634839e-7 and 1.193473e-7 for mean gradient, reward and Adam, respectively.
All nine Adam updates also agree with independent arithmetic, maximum
1.199205e-7; the actual ninth parameter delta exactly matches the saved update.
This does not verify a completed ten-round fit or an independent decoder
Jacobian. The reproduced GPU fit was not separately timed, so no total recovery
GPU cost or completed-run throughput is claimed.

## Concrete draft, not a deployed repair

`vg_tta/decota_spatial_opd_boundary_chart_proposal003.py` retains the exact
original `logit(box)` value and gradient for all interior coordinates. Its
proposed endpoint branch would use the finite native logit from the same bbox
head evaluation, after verifying identical float32 sigmoid output. It introduces
no epsilon clamp, box change or arbitrary coordinate cap. Local CPU tests cover
the actual reproduced coordinate and finite endpoint gradients; portable tests
pass four valid regimes and reject seven invalid inputs. These are not GPU
qualification, model-Jacobian tests, or a formal resumed run.

The locked protocol explicitly specifies that exact 0/1 coordinates fail chart
qualification. The draft therefore changes that failure behavior and cannot be
silently installed as a Softmax audit supplement. Explicit human authorization
is pending. Original fitting, configuration, selected parameters, code pins and
predictions remain untouched. No representation revision is registered in
CURRENT or imported by the current controller.

## Scope and retained stage evidence

HC2 has sealed all 10,446 arrivals (3,482 validation clip/query inputs from
237 parent movies, three orders). Root separately read all 13,928 opaque files,
13,053,097,040 bytes: 10,446 outputs/13,025,335,506 bytes and 3,482 inputs/
27,761,534 bytes. Exact roster coverage, SHA256, receipt byte counts, runtime and
receipt-before-seal timing passed; the supplementary receipt also verifies an
unchanged barrier snapshot. Four valid integrity contracts and 25 deliberately
invalid coverage/receipt cases passed. This is not HC2 GT scoring or dense,
mathematical, effectiveness or state-chain completion.

The full P1 target remains 41,355 adapted arrivals. Both-direction global seal,
GT scoring, root audits, report figures and subsequent P2–P6 are incomplete.
EATA and its media/Fisher preparation remain paused. The public diagnosis export
contains code, protocol, safe scalar diagnostics and aggregate integrity metadata;
private media, annotations, weights, predictions, gradients, Adam states and
the reproduced tensor capture are excluded.
