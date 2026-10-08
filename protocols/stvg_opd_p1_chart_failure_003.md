# P1 VidSTG native-box chart failure: bounded root recovery

The current revision002 controller and VidSTG worker exited after 1,882 saved
arrivals in order1. The first missing arrival is 1882 (zero based), official
query ordinal 10220. The original assertion is in the inverse-sigmoid chart of
the native cxcywh output, called after an Adam update. It is distinct from the
previous cross-precision Softmax audit failure.

Before new execution, preserve the original code, scientific configuration and
runtime, controller/worker traceback and log, every complete prefix payload and
JSON receipt digest, and the already captured input. The original dead worker
did not serialize this failing fit. A later replay cannot be described as a
comparison with its inaccessible memory.

The initial diagnostic is exactly one first-missing fit after reconstructing the
complete receipted prefix. It reuses the captured input/expert and original
random draws, source checkpoint, 1792 update interface, Adam, fixed parameters,
output and GT barrier. A process-local forward hook records the original bbox
head's pre-sigmoid outputs without changing the forward return. An exception
hook saves the live state, Adam moments, completed rounds, partial path, boxes
and raw logits before the fitter's finally block restores the origin. It then
stops without accepting a prediction. No GT, scores, new expert observations,
sample skipping, clipping, LR change or additional parameter search is allowed.

Diagnosis must distinguish nonfinite computation, a genuinely unsupported
coordinate, and finite pre-sigmoid logits rounded to a box boundary. Numerical
representation work requires separately pinned code and an independent check;
it cannot silently change the Gaussian policy or erase the original failure.
Original fit/prediction/runtime bytes remain protected. A new scientific rule
would require explicit human authorization rather than being called an
engineering audit repair.

The HC2 direction has already sealed all 10,446 predictions and passed the
separate opaque integrity readback. It is not rerun. The global P1 deployment
seal remains absent; no P1 GT scoring is permitted by either the HC2 seal or
this diagnostic. EATA and its preparation remain paused.
