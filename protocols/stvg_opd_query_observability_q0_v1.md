# Q0: frozen Query controllability, expert observability and mean readout

Human authorization: the supplied Query-Level OPD attachment, followed by
“做这个” and “先做一下”. This is the bounded Q0 diagnostic from that attachment,
not a new Gaussian policy, iterative adaptation, production registration or
permission to reopen old queues. Prioritize Q0 at a saved-query boundary;
preserve the current O and S processes, their in-memory state and all prefixes.

Question: with a common saved Before state and one output RMS budget, do
observable Query directions offer candidate improvements that existing sparse
expert rewards can identify and a true parameter-mean decoder readout retains?
LN-observable is the required matched control. Old mixed Query+LN candidate
search and compatibility distillation are historical evidence, not rerun here.

## Fixed population, inputs and states

- HC: all 32 original development parents, order1 of the already selected
  `stvg_opd_hc2_coordinate_v1/cfg_e8735d3fe1dd4bc2` trial.
- Vid: all 32 original development parents, order1 of the already selected
  `decota_spatial_opd_tuning_v1/vidstg/refine_10` trial.
- One exact saved clean query and complete 1792-dimensional Before state per
  parent; these are historically exposed development samples. The existing
  checkpoints, Native WHEN, frames, admission and Top1 evidence are fixed.
- Reconstruct the frozen native prefix and verify original input/pixel/source
  bindings and bitwise Before output. Reuse original admitted expert payloads;
  no new teacher calls, optimizer updates or cross-query candidate writeback.

The **single pooled budget** is the median over these 64 original first-step
full-native normalized cxcywh RMS movements, including zero movements.
It is locked from saved pre-GT GPU trajectories before any new Q0 GT access.
Reward temperature stays at the original dataset value (HC .05, Vid .25).

## Direction construction and amplitude calibration

Compute actual autograd Jacobian rows of the normalized cxcywh output at the
original admitted observation positions, for Query256 and LN1536 separately.
CPU float64 thin SVD; numerical rank uses
`max(J.shape) * float32_epsilon * largest_singular_value`.
The common direction count is `min(8, rank_Query, rank_LN)`.
Record the complete spectrum, numerical ranks, top-k spectral energy and all
undefined/zero-observation cases. Rank is a numerical convention, not a proof
that discarded directions are semantically irrelevant.

Three groups: Q-random (fixed-seed QR orthonormal Query directions),
Q-observable (leading Query right singular directions), LN-observable (leading
LN right singular directions). Canonicalize direction signs by the largest
absolute coordinate. All groups share central Before and the same k; maximum
17 candidates per group (central plus up to eight positive/negative pairs).

For each signed direction, restore the full initial state, then calibrate its
amplitude using **actual full-native output RMS**, not parameter norm. The
bounded parameter-direction amplitude is at most 1.0 in Euclidean units.
This engineering bound is predeclared and has no GT dependence; Query/LN units
are not considered equivalent. Probe at .001, use a linear initial estimate,
double at most ten times within the bound, then 12 bisections of a bracket.
Keep the best measured output no greater than the budget. Budget attainment
means residual <= max(1e-7, .01 * budget); unmet budgets remain in all results
and are separately reported. No infinite magnification, box clipping, radius
sweep or dropping troublesome directions/samples.

## Selection, projection and frozen-state restoration

Reward is the mean original admitted Top1 IoU over valid expert positions.
With no admitted expert, reward is zero. Reward-only top1 uses stable first
index ties; central candidate is index zero. Ties remain explicit.
Weights are the actual softmax of the frozen rewards at the original tau,
including central candidate. The mean **parameter delta** is the weighted sum
of signed deltas. Restore the full Before state and run the decoder at that
mean. A weighted mean of candidate boxes is only a Jensen-gap diagnostic,
never the method output. Equal-reward cases remain visible, including any
drift from asymmetrically calibrated positive/negative latent amplitudes.

Restore all 1792 coordinates after every readout. Preserve Query/LN inactive
coordinates, source model hash, Native timing and original inputs. Jacobian,
SVD/directions, amplitude traces, candidates, reward/ties/weights, actual mean
decode and before/after state hashes are private complete evidence.

Four fixed qualification queries (first two per dataset), each executed twice
in full, precede all 64 formal queries. Qualification is not reused as formal
prediction. Check bitwise scientific outputs/state restoration/repeat rewards
and geometry, including no new DINO/optimizer calls. Actual directional finite
differences are also retained as descriptive Jacobian diagnostics, not claims
about an independently reconstructed complete decoder Jacobian. CPU synthetic
contracts are preparation, not GPU qualification.

## Post-global evaluation and closure

All 64 complete formal payloads, all three groups and mean readouts must be
globally sealed before loading new Q0 GT. Original official dense evaluator;
independent dense reproduction and 10,000 paired parent bootstrap seed
20261012. Whole-tube oracle chooses **one complete candidate** by vIoU; no
per-frame GT stitching. Report oracle, reward-only top1 and true parameter-mean
vIoU changes relative to common Before, and matched observed/unobserved decoder
event-frame IoU changes. Keep interpolated/outside-Native partitions separate.
Retain all parents, no-expert/unknown-evaluation/tie cases, budget shortfalls,
5pp and 20pp negative tails, full continuous rows, paired group contrasts and
uncertainty. These development intervals do not establish online stream value.

No arbitrary performance gate automatically launches further Query OPD.
Evidence is classified as candidate reachability, expert identifiability,
mean retention and Query-specific advantage separately. Actual root source,
complete-state and scalar review, report PDF pixel view, anonymized positive
and negative publication to Zonglin-He/A, remote bytes/SHA/Gitblob/tree
verification, archive check/snapshot/check and true Q0 FINAL are required.
Existing S/C/N/O work remains authorized and resumes; its FINALS do not close Q0.
