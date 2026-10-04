# R6 fixed-configuration cross-domain qualification

This implements the user's full six-round attachment, preserving the prior
assistant-added same-domain resource gate as a scope correction. That gate
was not requested by the user and does not replace an actual cross-domain
measurement. No method, source, parameter or checkpoint was selected again
from confirmation results. This is a new isolated finite queue, not a resumed
historical cross-domain/full-query queue.

Target VidSTG uses the official HC-STVG-v2 checkpoint; target HC-STVG-v2 uses
the official VidSTG checkpoint. Both retain their original fixed 32 development
and 16 historically exposed confirmation sources, one query/source, two orders,
clean and five 5% corruptions, exact old RGB pixels and frame grids. The two
checkpoints are source trained; no new supervised training or calibration.

Frozen configuration is frame-level all-proposal critic, Adam .03, 10 steps,
own-energy minimum over steps0..10, joint1792 query/LN parameters, temperatures1,
query and optimizer reset per arrival, LN1536 writeback1/16. Native temporal
readout stays fixed. Full/Extent posterior, T1 and occupancy were unqualified
in R1; track and modified scope failed their prelocked development preservation
rule. None is silently added. No parameter search or confirmation selection.

Capture576 unique inputs using the matching opposite checkpoint and obtain
four native-uniform DINO observations per input, capped2304 detector requests.
Two clean inputs/dataset validate raw/normalized native parity, every fit step,
state/gradient and final original-model reinsertion without GT. Their capture
and observations are reused by hash receipts in the576 input acquisition.
No within-domain H/observations are reused when checkpoint or positions differ.

Run episodic and actual100%-expert online chains independently. Each stream
has1152 logical arrivals across both datasets; source LN resets at each
dataset/split/corruption/order boundary. Reset query residual and fresh Adam
per arrival. Persist only1/16 LN displacement, discard query residual.
The source model and detector remain byte-identical. Old P0/P1 results are not
substituted for new cross-domain trajectories. Same-domain25/50/100 budgets
were already fully measured and are reported separately; this cross-domain
qualification has one fixed100% expert policy, not a new budget sweep.

All2304 readouts plus their capture/evidence hashes must globally seal before
GT opens for this stage. Historical exposure is disclosed. Official dense
metrics, independent optimizer/objective/lifecycle arithmetic, source-macro
and10000 paired source-bootstrap, clean/corrupt/condition/order, episodic,
Before/After/Frozen, positive/negative tails and observed/unobserved harm are
reported. Confirmation tests transfer qualification, never changes method.
Report real detector/capture/gradient calls and wall time separately from
logical observation budget; wall time is not pure GPU-kernel latency.

All current and past engineering failures, scope correction, protocols and
negative findings remain. Publication excludes private media, annotations,
weights, captions and raw payloads. The route closes only after root visual
review, archive check/snapshot/check and verified public GitHub synchronization.
