# Authorized native box chart revision003

The human instruction "修复接着做" on 2026-10-09 authorizes deployment of the
previously reviewed finite native logit proposal after actual qualification.
This explicitly revises the original exact-0/1 failure behavior. The original
protocol, failure, replay, code and all receipted predictions remain preserved.

For each selected coordinate, ordinary interior values retain exactly
`torch.logit(native_box)`. When float32 sigmoid rounds a finite native logit to
0 or 1, the coordinate mean is that finite pre-sigmoid logit from the same
decoder call. A return-neutral head hook proves the raw-logit/native-box
binding before construction. Nonfinite logits, nonfinite/out-of-range boxes,
wrong shape/dtype or different native outputs are rejected. No epsilon clamp,
query skip, step reduction, exploration change or hyperparameter change occurs.
The deterministic box readout remains the original native float32 sigmoid.

Each new fit records the before/after chart trace with its round, selected
boxes, native logits, finite means and exact endpoint mask. Independent CPU
readback binds the trace to the original state path and round means, checks
the endpoint identity exactly, and recomputes interior logit/native sigmoid
with explicitly stated float32 tolerances. Original revision002 Gaussian,
detached IoU, gradient and Adam checks continue unchanged. This audit does not
claim an independently reconstructed complete decoder Jacobian.

Before continuation, two predeclared ordinary actual VidSTG fits are checked
against their old complete saved fits for bitwise identity of all original
outputs/actions/gradients/Adam/state fields. The first missing failed query is
replayed from the complete 1,882-arrival prefix. Its first eight complete
rounds and all nine pre-failure updates must match the captured reproduction
bitwise; the repaired tenth round must complete, pass independent arithmetic
and a real native-head-output VJP check, and repeat bitwise. The dead worker's
lost memory was not serialized; equality is with retained actual reproduction
and between new repeated executions, not with that inaccessible memory.

Qualification accepts no prediction. Formal continuation must match that
qualified first missing fit bitwise before accepting it. Already sealed HC2
10,446 predictions and all original VidSTG 1,882 predictions are not rewritten.
Only the missing VidSTG suffix continues, with original per-order source resets,
per-query residual/Adam resets, final-round output and fixed LN writeback.
Post-seal CPU audits dispatch by each saved numerical-audit revision so old
receipt dictionaries are reproduced exactly. All 41,355 deployment arrivals
and both directional barriers precede P1 GT evaluation. EATA remains paused.

The separate runtime binds this bridge, original scientific/runtime hashes,
human authorization and retained failure. Code/contracts/authorization alone
do not establish real GPU qualification, completion, efficacy or robustness.
