# Source label identity correction, before any critic training

The original source input roster and all candidate computations remain frozen.
Inspection of the old processed VidSTG pointer found repeated identical captions
within the same video but different `used_segment` events. Video+caption alone
does not uniquely identify the source task. The old downstream source label
function has not executed; no critic has trained on those labels.

Correction: use original official VidSTG train records, match actual input
video, complete Query and exact `used_segment` start/end. Require unique
target-ID/event identity; retain equivalent duplicate annotation indices. Read
the matching original VidOR trajectory by physical frame ID, within the original
event span, using the already audited benchmark endpoint conventions. Source GT
is allowed for training labels only, never critic features. Source GT schema and
binding metadata were inspected by root before source candidate completion;
actual training labels are computed after the full source prediction barrier.
No target GT score, prediction improvement or model output resolves ambiguity.

The source GPU worker completes all96 candidates unchanged. Only its waiting
CPU parent is SIGSTOPed with verified pidfd/PID/starttick/cgroup; no GPU signal.
Controller002 waits on that original worker pidfd, requires complete96 barrier,
then retires the old CPU unit and runs corrected labels, original critic
training, frozen inference and scoring. The original 4h deadline is retained.
Original plan/runtime/code/old pointer/failure receipts are preserved, not
silently overwritten. This is a source data identity correction, not a target
performance-driven method revision or a new scientific search.
