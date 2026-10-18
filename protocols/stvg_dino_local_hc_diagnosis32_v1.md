# Cached HC local-residual diagnosis

Question: why does the independently aggregated local residual still harm HC?
Separate teacher quality, finite24 residual aggregation, and applied displacement.
Use the same HC16/Vid16 only; Vid is a contextual comparator, not a new cohort.

Inputs are immutable 505103c1 saved actions/accepted aligned teacher feedback,
saved per-sample GT IoU scalars, and d9f0c700 saved outputs/complete metrics.
No model, expert, feature extraction, network training, new prediction, new GT
annotation/media/weight read, new GT evaluation, alternate amplitude or parameter
search. This is accounting of already measured quantities, not a method trial.

For a planned node, E state e, input displacement eps_k=z(input_k)-e,
teacher displacement d_k=z(T_k)-e, and admission m_k, reconstruct exactly

    applied r = sum_k m_k(d_k-eps_k)/24 = teacher_term - input_term.

Also decompose input_term into p*mean(all24 eps) and
p*(mean(accepted eps)-mean(all24 eps)), where p=n_accepted/24.
This is finite-set accounting, not evidence that the latter is a population
bias or an independent causal effect. Retain no-feedback exact-zero contract.
Use the existing source scales for dimensionless norms; report center motion
relative to E width/height and log-size change separately. Do not invent a
new scale or use GT to select nodes.

Record support count, normalized directional coherence of corrections,
teacher-displacement versus applied-residual cosine, accepted-input variance,
and fixed-denominator24 leave-one-term sensitivity. The latter evaluates vector
stability only, not alternative predictions or GT-selected subsets.

Read saved teacher/input/E IoUs at exactly known nodes. Decompose
gated teacher-minus-input = gated teacher-minus-E + gated E-minus-input,
averaging all24 slots and all known planned nodes per query. Teacher-minus-E
near zero cannot be presented as a useful E-improvement signal just because
teacher-minus-perturbed-input is positive. Compare saved aggregate observed-node
quality but do not call this a pure averaging intervention: transport from the
sampled state to E and IoU nonlinearity are also present.

Applied amplitude is described, not tuned: norms, per-coordinate and center/size
contributions to vector norm, fixed-denominator leave-one sensitivity and saved
signed outcome association. Without evaluating other amplitudes or the true
directional derivative, overshoot is not causally identified. In particular,
low coherence is not proof that cancellation causes harm, and a wrong geometry
cannot be renamed an identity error without visual evidence.

Keep all32 signed outcomes, unknown nodes and unsupported queries. Bootstrap at
parent/query level (10000, seed20261018). Predefine HC negative cases from the
saved loss ranking and retain H008/H043 positive controls. No dynamic OPD or
automatic new experiment follows this diagnosis. Publish anonymous scalar
accounting only and preserve all old inputs and production registration.
