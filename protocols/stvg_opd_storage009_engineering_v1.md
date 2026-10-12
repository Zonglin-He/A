# Frozen attention ephemeral-storage revision009

O retains the original 1792 cells, 5376 three-arm formal records, inputs,
weights, offsets, dtypes, fitting, gradients, Adam, actions, rewards, budgets,
Native WHEN, parameters and original scientific runtime. The three actual
frozen-prefix allocation failures and all accepted records are preserved.
Revision008's bounded qualification and first formal readbacks passed, but a
later mask-add allocation failed after 2832 accepted formal records. Those
valid snapshots did not prove future OOM safety.

Revision009 operates only in the eval/no-gradient frozen WindowAttention3D
prefix. It retains the original type promotion, bias addition, mask addition,
softmax, ordering and shapes while reusing the ephemeral attention buffer.
Bias and mask additions reuse the promoted original buffer. The original
softmax operation writes to that same buffer via its supported output argument.
The source operation is restored after capture. The original source file and
all scientific fitting code remain unchanged. No temporal/window/frame chunks,
precision reduction, prediction clipping, sample skips or alternate attention
formula are introduced. The host staging from revisions007/008 remains.

The original actual WindowAttention3D CPU and CUDA autocast operations versus
the storage variant are compared on both mask branches and three batch sizes:
nine CPU and nine GPU comparisons, each with 677376 output scalars checked
bitwise. These are bounded arithmetic contracts, not full fit qualification.
Full qualification is fixed beforehand: first two original HC and Vid cells
and all three actual failed Vid cells; seven cells, three arms, two complete
capture repetitions, 42 real full GPU fits. Existing saved qualification,
original Full complete trajectories, repeated new trajectories, math/state and
the retained 2832 predictions/receipts are checked. New qualification output
does not replace formal output. First two resumed cells have a separate
complete qualified/formal and CPU math/state/input/hash readback.

All 5376 formal predictions must seal before task GT. Independent statistics,
actual PDF pixel review, all anonymous positive and negative outcomes, remote
byte/SHA/Git-blob/tree checks, archive and genuine phase FINAL remain pending
until actually executed. Missing complete failed-run witnesses are not treated
as recoverable dead-memory evidence. Qualified cases cannot prove safety for
all future memory demands or complete decoder Jacobian correctness.

The same Luna max monitor checks every 30 minutes under the human instruction.
It is read-only, reports to the root task, and cannot prevent an allocation
failure occurring inside a query. It does not send process signals or modify
experiments. Original baseline memory and any desktop session remain preserved.
