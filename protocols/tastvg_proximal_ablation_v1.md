# Evidence strength and bounded spatial updates

User authorized this matched development experiment on 2026-10-02. This reopens
the preference-to-parameter mechanism because sealed quick diagnostics showed
sharp rank targets for weak evidence, nonzero flat-reward updates, and harmful
updates beyond the probe radius. It does not reverse the old negative results.

Original 32 search sources per dataset, one query/source, two original orders,
clean and five original GT/query-independent 5% transient corruptions: 384
arrivals/arm/dataset, 3072 arrivals for the four spatial arms. All sources are
historically exposed development data. Reuse identical v3 H and expert receipts;
no new backbone or expert inference, no new corruption, no full-query evaluation.
25% scheduled experts, persistent state reset per condition/order, dynamic native
suffix, 1792 spatial parameters, nine detached local intervention candidates.
Current output is sealed before updates and updates serve subsequent arrivals.

Per-dataset sealed v3 best parameters remain: Vid lr=.033761698432507946,
teacher temperature=.34902548789596055, K=1; HC2 lr=.006097133675874025,
teacher temperature=1, K=8; rho=.05, student temperature=1, four directions.

A: rank teacher and SGD. B: adaptive proximal teacher and SGD.
C: rank teacher and an arrival displacement budget. D: proximal teacher and budget.
The four arms are run with refreshed candidates and gradients as states evolve.

Calibration uses only first-step reward spreads in arm A, verified bitwise against the sealed selected v3
development run (all six conditions/two orders, one spread/scheduled arrival).
s_ref is the median strictly positive max-minus-min spread, separately by dataset.
Old compact v3 logs lack raw rewards, so A replays them using existing H.
All A outputs seal before this no-GT calibration; no GT is read for calibration. lambda=spread/(spread+s_ref).
The proximal target is (1-lambda)*stopgrad(p_current)+lambda*q_rank, constructed
anew at each step and frozen throughout that step. Exact equal rewards imply
lambda=0, exact zero loss/gradient and no update. This coefficient is evidence
strength, not a correctness probability. p is geometric compatibility, not a
native tube likelihood. Frozen targets are also used for post-step KL logging.

The cap is .1 times the original ABSOLUTE probe radius (rho times source center
L2 norm), shared by all steps in an arrival. Each proposed SGD state is projected
around the incoming arrival state. Small updates are unchanged. There is no
fixed-norm gradient normalization and no source-initial-state anchor.

No-GT smoke checks baseline parity and multi-step bounds before the batch.
All four arms on both datasets seal before CPU GT scoring. Main endpoint is
corruption future/nonexpert source-macro dense vIoU; report paired 10000-source
bootstrap, tIoU, clean, all/expert, gross gain/loss, >5pp negative tails and
per-step correct-preference/GT-harm diagnostics, lambda and actual displacements.
Preserve all positive and negative results and engineering failures.

If B has positive mean paired future-corruption vIoU gain over A, run one
additional fixed-mixture control on that dataset. Its lambda is the mean adaptive
lambda from the frozen calibration spreads (empty evidence counted as zero).
It is a mechanistic development comparison, with confidence intervals reported;
positive mean alone is not a confirmatory significance claim.

Spatial selection per dataset uses future-corruption source-macro vIoU on these
development sources; ties within 1e-12 prefer A,B,C,D in that order. Selection
does not promote a production method. Following sealed spatial selection, one
separate matched native temporal-head persistent-update comparison is authorized.
That interface must preserve both offsets and physical-time mapping and identify
full legal span versus restricted-candidate probability semantics explicitly.
Its additional implementation/runtime lock is required before inference. Same
Fast critic and spatial arm with/without temporal update, pre-update current
output, future nonexpert tIoU/vIoU and spatial spillover are the relevant endpoints.

Event support weighting and HC source anchoring remain conditional future routes;
they are not silently added here. No GT gates, reference resampling, extra model,
baseline, old queue restoration, full-query run or further parameter grid.

TOP-D (arXiv:2607.04751, Sec.3.1) motivates probability-space arithmetic mixing.
The parameter L2 projection here is not TOP-D's policy-ratio-clipped optimizer;
its autoregressive theory is not a STVG GT guarantee.

Independent root audit must verify mixture, projection, state chains, detached
targets, coverage, metric algebra and sealed predictions before GT. Publish code,
protocol, all anonymous outcomes, negative cases and costs to Zonglin-He/A with
remote content verification and research archive check/snapshot/check.
