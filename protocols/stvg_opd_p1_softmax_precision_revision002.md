# P1 audit-only softmax precision recovery 002

The actual revised HC2 P1 failed at order2 arrival148/query2960 after 3630
receipted predictions. The original audit's float64 softmax compared with saved
GPU float32 weights exceeded the original 3e-7 absolute threshold; the old
revision001 correctly refused to supplement an unexpected failure site.
Original code, runtime locks, logs, stage metadata, and all 3630 opaque
predictions (4524043386 bytes) are preserved and byte-verified.

The old wrapper did not serialize the original failing fit. The new capture is
explicitly a deterministic reproduction from the full saved prefix, not recovered
dead-process memory. It re-established the same original softmax assertion,
captured all 40 rounds, and stopped before accepting a new prediction.

## Evidence and exact implementation

The installed runtime is PyTorch 2.7.0+cu128, git commit
134179474539648ba7dee1317959529fbd0e7f89. Its CUDA scalar division uses float32
reciprocal followed by multiplication, while the CPU/direct float64 division
used in the old reference has different intermediate rounding. Primary source:
https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu#L32

Reproduction: original mixed-precision max error 3.477426682718665e-7, with the
failed round9. An unmatched CPU32 division also differs by 3.2782554626464844e-7.
CPU32 with the exact pinned CUDA scalar-reciprocal operation differs by at most
2.9802322387695312e-8; independent float64 normalization of its rounded logits
differs by at most 1.968405555219377e-8. These are math-audit quantities, not GT
task measurements.

## Supplement, without changing scientific behavior

New process-local revision002 changes only the independent auditing path.
Original scientific fitter, loss, probabilities, sampling, gradients, optimizer,
state consolidation, readout, admission, source checkpoints, order/roster,
selected configuration, and all original runtime-lock bytes remain unchanged.
HC2 remains .01/.025/.05/40/LN1/16/M32 and Vid .03/.1/.25/10/LN1/8/M32.

The original mixed-precision check remains a recorded failure. The supplemental
check separately validates the actual rounded CUDA logits with gamma3 for the
three input/conversion/reciprocal/multiplication rounding effects, checks CPU32
and independently normalized probabilities against saved GPU weights using the
unchanged 3e-7 threshold, and bounds the cross-precision discrepancy using exact
componentwise softmax sensitivity intervals. Gradient/IoU/Adam/1792-chain and
final-round checks remain unchanged. Fallback is limited to original lines25/28;
any other assertion remains an actual failure and now captures its full fit.

CPU validation covers 16 synthetic precision regimes with 16 rejected wrong,
normalized probability distributions and all 40 rounds of the reproduced fit.
GPU_requal repeats the complete fit, checks all tensor/scalar values against the
serialized first reproduction, runs the independent supplement, and stops before
saving a prediction. Only after its actual pass can the finite controller resume.

The first formal unreceipted fit is again checked against that same reproduction
before accepting arrival148. All previous 3630 prediction and receipt hashes are
retained. Query residual/Adam reset and complete inherited LN state follow the
original runner's full saved chain; no source reset mid-stream or old685 prefix
splice is permitted.

## Postseal readback and completion

The original global HC2+Vid P1 deployment barrier still precedes every GT read.
The CPU bridge dispatches each prediction's audit by its saved revision metadata:
original, revision001, or revision002. It must reproduce its exact saved audit
dictionary and does not rewrite old receipts. Independent statistic/dense/state
audits, actual root figures and cases, public anonymous results including negatives,
remote verification, and research archive closure remain mandatory after full P1.
Recovery/requalification or finite controller launch is not whole-paper completion.
EATA, source-media/Fisher preparation, and other old paused queues stay paused.
