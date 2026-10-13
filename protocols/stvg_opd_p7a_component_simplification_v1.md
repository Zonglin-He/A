# P7-A / OPD internal component experiment

Human authorization on 2026-10-13 is to run the corrected internal experiment.
The fixed objective is OPD; this experiment has no alternate-objective gate.

Use the original 32 development parent sources per target and both original
development orders, exactly from the original tuning design. These sources have
historical GT exposure. They are not the disjoint P7-A non-development panel.
Four arms form Query Residual on/off x attenuated LN carry/source reset:
joint_carry, LN_carry, joint_reset, LN_reset. Joint uses 1792 optimizer
coordinates; LN-only uses exactly 1536 with the 256 Query coordinates frozen
at zero. All arms reset Query and Adam at every arrival. Each order and arm
starts from the actual source checkpoint LN values and maintains its own chain.
Reset arms start every arrival from that source state, never zero-valued LN.

HC parameters: lr .01, sigma .025, tau .05, 40 complete updates, carry 1/16.
Vid parameters: .03/.1/.25, 10 complete updates, carry 1/8. All use 32
antithetic Gaussian actions, original Native Uniform4 observation positions,
original admitted Top1 cached DINO evidence, on-policy refresh and last output.
Noise is keyed by the original query key and round, common across arms and
orders. No new DINO, teacher, frame truncation or precision change is introduced.

Primary output is the existing P7-A actionness projection I1, computed from
source-frozen two-offset temporal logits/actionness/physical IDs with the
original per-target recipe. Native I0 is the secondary output on the same boxes.
The original development inputs require new frozen temporal capture: P7-A's
non-development captures cannot be substituted. Frozen Source+I1 is recorded.

Qualification runs the first two arrivals of both orders in both targets,
four arms and two full repeats with fresh frozen captures: 64 actual complete
fits, 16 captures, 1600 update records. Independent objective/scoped Adam/state
readback, complete repeats and the matched old joint_carry controls must pass.
Formal is 512 complete fits, 128 captures and 12800 update records. Qualification
fits are not formal aliases. First two formal fits per arm/order must match
qualification. All 128 original joint_carry fits are replayed and compared to
their complete original scientific fit. Global seal of all arms/targets/orders
and independent label-free root precedes any new GT scoring.

Report all absolute m_vIoU/m_tIoU/fixed-GT m_sIoU, R@.3/.5, Before/After/Frozen,
direct factorial contrasts, every order, parent-macro paired 10000 bootstrap
(seed 20261013), severe >5/>20pp tails, no-information paths, parameter scope
and actual instrumented cost. Average orders within each parent before sampling.
Intervals are unadjusted exploratory intervals, not equivalence tests. A CI
crossing zero does not automatically delete a component. No per-query GT step
selection, new parameter search, production registration or full-main launch.
This finite first experiment determines whether a later bounded tuning phase
is informative; it does not itself run the later phase or certify a final method.

Preserve all raw private fits/inputs/checkpoints and all failures locally.
Publish exact implementation/configuration and every anonymous outcome, including
negative/neutral/unknowns, independently review the actual PDF pixels and remote
bytes/SHA256/Git blob/mode/tree, then close and update the project archive.
The cancelled baseline stays in its original stopped identity; never signal it,
release its ordinary lease or recreate the user-deleted monitor.
