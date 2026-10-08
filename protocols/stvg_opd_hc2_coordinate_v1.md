# HC2 OPD sequential coordinate search, 2026-10-08

The human explicitly reopened HC2 tuning and requested optimizing one parameter
before searching the next. This supersedes the previous fixed HC2 configuration
for future runs. Preserve P0 negatives and the paused P1 prefix unchanged.

Use the original 32 historically exposed HC2 development parent movies, one
locked official validation clip/query each, and both original stream orders.
The source is the frozen VidSTG checkpoint. Neither the 128-parent confirmation
panel nor P1/full validation scores may select parameters. This is development
selection on validation data, not independent confirmation or fresh evaluation.

Start at lr=.03, sigma=.1, tau=.25, steps=20, writeback=1/16, M=32.
One finite greedy pass has this predeclared order and grids:

| Coordinate | Candidates |
|---|---|
| Adam learning rate | .001, .003, .01, .03, .1 |
| Gaussian logit sigma | .025, .05, .1, .25, .5 |
| Detached teacher temperature tau | .05, .1, .25, .5, 1 |
| Last-round update steps | 1, 3, 5, 10, 20, 40 |
| Persistent LN writeback | 0, 1/64, 1/32, 1/16, 1/8 |

For one coordinate, every other coordinate equals the last locked winner. Run
all its candidates before reading any scores from that coordinate. Seal complete
64-arrival predictions or an explicit unscored numerical-invalid disposition
for every candidate, then seal the coordinate barrier. Only then run CPU GT
evaluation. Never score a numerical-invalid prefix. Preserve every failure.
An assertion that the policy's native box chart is not invertible is a numerical
invalid trial; unrelated errors require preserved engineering diagnosis.

Rank complete candidates by development parent-macro After-minus-Frozen vIoU.
Exact mean ties: fewer parent losses >20pp, retain the incumbent, then smaller
steps and smaller grid ordinal. Lock the selected complete configuration and
its barrier/score hashes before constructing the next coordinate. No joint
Optuna proposals, alternating coordinates, per-query choices, best-step output,
result-driven grid additions or second pass are authorized by this protocol.
"Best" is grid-best conditional on preceding choices, not global optimality.

At most 26 logical candidate evaluations and 22 unique complete configurations
(1408 adaptation arrivals) plus four real qualification fits. Exact prior
64-arrival development trials may be reused only with identical source,
configuration, roster, full two-order history, reset semantics and verified
immutable prediction/input receipts. Recompute their scalar scores under the
new audit; never reuse a shorter stream or confirmation trial.

Keep Native WHEN, original Uniform4, one frozen DINO, original admission and
Frame-Top1, joint 1792 parameters, M32 antithetic on-policy Gaussian likelihood,
detached IoU feedback, per-query residual/Adam reset and last-round output.
No online GT, new expert, scorer, gate, temporal module, backbone or memory.
Do not change VidSTG's configuration. Keep EATA and media/Fisher preparation
paused. Other paper stages remain paused throughout this search.

Independently audit every selected coordinate, state/Adam/likelihood arithmetic,
official and dense metrics, all 32 sources, 10000 paired source bootstraps,
current/inherited decomposition, positive and negative tails and real GPU/CPU
cost. Publish every actual candidate including failed/negative trials, anonymous
scalar results, configuration, code, sensitivity plots and material limitations.
Actually view plots and verify every published remote file. Update the archive
check/snapshot/check. Register one selected HC2 configuration only after these
audits; retain VidSTG byte-for-byte and preserve prior configuration bytes.

The old P1 prefix is resumable only under its original configuration. A changed
HC2 configuration needs a separately locked P1 revision starting from the source;
never concatenate different configurations into one claimed full stream.
