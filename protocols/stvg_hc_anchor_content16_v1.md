# HC independent anchor content state: E / Direct / Fixed / Live

User-authorized finite development experiment, 2026-10-19. All NH16 exposed
parents, one query per parent; no Vid gate, new cohort, expert/model/head training,
new observation, DINO call, temporal change or parameter search. Registries and
historical paused queues stay unchanged. No automatic confirmation.

Frozen original Vid-source EMA TA-STVG. Reuse exact original Source, E, four
planned physical positions, original dual-offset cached spatial decoder inputs
and cached observation_factorial32 RGB-crop detections. Admission/text rules are
unchanged. At each planned position choose the maximum target-score crop box if
the crop was admitted; otherwise choose the maximum target-score original
admitted pool box; otherwise no teacher/state at that slot. No E residual
association or GT-dependent filtering. Unit teacher weights, missing slots
unknown, all16 parents retained. Existing masks/no-reference/harmful/time-wrong
examples are not removed. This teacher choice is fixed before scoring.

State: at most K=4 independent256D temporary vectors, initialized0 per arm/query,
added to the corresponding `query_tgt` token immediately before spatial block0
temporal self-attention. Physical p maps to offset p%2, local p//2. No broadcast,
learned mapping, LN updates, original weight changes or cross-offset attention.
Both offsets decode all original actual frames through all six frozen blocks.
Paired center-difference/log-size E-retaining mapping gives bit-exact E at0.

One fixed configuration inherited numerically from existing HC adaptive budget:
Adam lr.01, betas(.9,.999), eps1e-8, weight_decay0;40 updates/final iterate;
Gaussian sigma.025,32 antithetic samples, teacher temperature.05. No step/layer/
state-size/temperature scan. No fractional writeback, loss-selected step or GT
rollback. Direct/Fixed/Live all have the same states,40 available updates and
one full-trajectory mean decode per step. Record actual work separately.

Common feedback cost per anchor is 5*L1(cxcywh)+2*(1-GIoU), evaluated in the same
E-retaining coordinates. It replaces old flat IoU-only rewards, is not assumed
calibrated, and stays informative for disjoint boxes. Direct minimizes mean
anchor cost (normalization1/K, explicitly not old sum/4). OPD constructs detached
weights softmax(-cost/.05), then applies the original Gaussian likelihood/
control-variate forward-KL gradient with1/K normalization. This is a route test,
not a single-factor comparison against historical fixed-IoU experiments.

The stochastic policy is a native-logit Gaussian at observed decoder outputs.
It does NOT sample content-state vectors or actual stochastic decoder trajectory
hypotheses. Every current policy mean and deterministic final trajectory are
real frozen-decoder outputs conditioned on the current anchor states. Samples
are mapped to E coordinates, feedback and candidate support stop gradient, and
the likelihood gradient passes through the mean and actual decoder to states.
Save samples, means, geometry costs, weights, gradients, Adam, states and outputs.

Fixed saves the initial32 candidates/feedback and reuses them while fitting the
current likelihood. Live resamples around each updated mean using the predefined
per-step CPU noise stream and evaluates the SAME frozen teacher/cost. Both arms
must match first samples, feedback, update and state. Fixed changes neither
student probabilities nor optimizer into a frozen object. No RGB re-encoding is
needed for on-policy candidate refresh. Direct is ordinary sparse localization
supervision and cannot be relabelled OPD. All arms use identical teacher boxes.

Engineering: CPU independent scatter/zero-gradient/non-overlap cost contracts;
per-query cached Source/physical-offset alignment; actual zero-state Source/E
equality; deterministic single-anchor probe on NH001 for offset isolation;
frozen parameter hashes/grad checks; finite likelihood analytic gradients;
independent replay of updates; Fixed/Live first-step equality and saved Live
current-mean provenance. No quality gate on a hard readout before running OPD.

GPU allocation cap900s including qualification/loading/failure work. Owned
systemd unit hard stop880s plus5s stop timeout; MemoryHigh16G/Max20G/Swap1G,
CPU400%, original shared nonblocking gpu.lock, disk30GiB/host6GiB/GPUfree4GiB
floors. One pilot NH001 uses the final configuration, no GT; its timing estimates
remaining work. If infeasible, preserve partials and stop, never reduce sample
count or change scientific parameters silently. No unrelated process signals.

Seal all16*4 outputs and full fitting traces before new HC GT access/scoring.
Evaluate original official interpolation/support with all parents. Report full
vIoU, fixed-support sIoU, observed and unobserved actual-node sIoU, paired parent
bootstrap95%, signed changes, severe losses, update counts, anchor fit, state
movement, affected offsets and exact costs. Confirm E matches old metrics and
time remains fixed. Formal FH/FV64 and new HC sources are excluded.

Interpret Direct−E as interface evidence; Live−Fixed as candidate-refresh
evidence; Live−Direct as benefit beyond sparse direct supervision.2–3pp refers
to complete-cohort full-v, not anchor fitting. All are development estimates,
not independent confirmation. One target direction may establish a result;
do not impose double-domain positivity. If Direct alone improves, do not claim
OPD. Failure of this finite state/cost/budget configuration is not proof that
all anchor-specific states or OPD are impossible.
