# STVG ports for the frozen DeCoTA paper, 2026-10-05

These rows are explicitly TENT-STVG/EATA-STVG/SAR-STVG task ports. Their
classification papers are not unchanged STVG reproductions. The selected DeCoTA
method is unaffected. No metrics from the official target panels tune a port.

Original algorithm sources: https://github.com/DequanWang/tent,
https://github.com/mr-eggplant/EATA and https://github.com/mr-eggplant/SAR.
The previous episodic small-cohort baseline implementations/unfinished queues
are not reused as completed formal results.

Shared interface: deterministic eval dropout, original two temporal offsets,
ground_decoder LayerNorm affine only (all such LN, not only Ours' last spatial
layer), original image/text encoders and native heads frozen. A live closure
through this scope is mandatory. Entropy is native H(start)+H(end)+mean binary
actionness entropy, averaged across the two original offsets. Masks and physical
frame grids are checked; actionness uses precisely one sigmoid. Final output is
a common post-update forward on the original input, not a development-selected
intermediate step. Full parameter and optimizer/moving states persist over the
same orders; reset only at condition/order/checkpoint boundaries. No DINO calls.

Fixed settings before target prediction: TENT Adam .001, one step; EATA SGD
.00025, momentum .9, one step, Fisher alpha2000, entropy margin .4 times the
query's native maximum entropy, cosine d_margin .05, EMA .9; SAR SGD .001,
momentum .9, three steps, SAM rho .05, same .4 ceiling-scaled reliability margin,
EMA .9, recovery below entropy .2. Relative entropy scaling, parameter scope,
post-update readout and source pseudo-native Fisher are disclosed task-port
choices, not claims of canonical classification hyperparameters for STVG.

EATA's redundancy descriptor uses a fixed 32-bin normalized observed-clip
endpoint probability histogram per endpoint, plus binary mean actionness;
concatenate the three unit masses with equal 1/3 normalization and average the
two original offsets. This descriptor is detached and used only for redundancy
selection. It is not a shared semantic class vector across different queries,
and the port's limitation is reported. The entropy objective remains on each
original valid grid. Neither descriptor bins nor margins are searched.

Full EATA requires squared live decoder-LN gradients from 2000 deterministic
hash-selected clean **official source-training** queries per source checkpoint,
at unchanged source parameters. Use pseudo start/end argmax and pseudo binary
actionness from the frozen source model, without target GT. Lock the source
input roster before inference, average per-query squared gradients, save exact
source parameter values, names and Fisher values, and independently check
finite/nonnegative values and scope. Target evaluation inputs are ineligible as
Fisher preparation. No Fisher=None EATA and no fabricated row if preparation
cannot be completed. Source/media preparation and actual source forwards are
reported as a separate cost. Source-only queries may be historically exposed;
do not claim a pristine source development split.

Before formal ports run, require live two-query no-GT smoke per checkpoint,
finite actual gradients (or explicitly logged selection skips), exact lr0
source parity and restoration, stream state persistence/recovery, independent
Adam/SGD/SAM arithmetic checks, and input/method/runtime pins. Synthetic CPU
contracts are implementation checks, not GPU qualification or benchmark scores.
No parallel GPU port smoke while Ours owns the single GPU queue. Every requested
row remains pending until its actual predictions and sealed coverage exist.
