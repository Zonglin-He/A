# Explicit spatial-policy OPD execution

User authorization: attachment 5ab425de, 2026-10-06. The original paper queue
is saved for resumption after this independent experiment. Its old Ours remains
the energy objective with its original state selection, and CURRENT is unchanged.

## Verified preservation and restoration

The old TENT HC2->Vid worker stopped after 11,791 receipted predictions. Its
original files, runtime/configuration, logs and deliberate interruption evidence
are preserved in `artifacts/decota_paper_experiments_v1/user_opd_pause_20261006`.
Order1 is complete; order2 has 1,488 arrivals. The full Adam checkpoint at
order2 arrival1400 was restored and the following87 saved arrivals were replayed
without GT. Before/After predictions and every parameter matched bitwise.
`EXACT_RESUME_STATE.pt` is the exact continuation state; next arrival is1488.
No old prediction or optimizer state was substituted with new OPD state.

## Actual qualification and finite queue

Eight CPU tests pass, including detached action/feedback, analytic and numerical
Gaussian gradients, antithetic refresh, and a true Adam no-op with existing
moments. Four real queries (two per dataset) and three arms give12 qualified
fits. Their independent NumPy feedback/mean-gradient/full1792 Adam calculations
pass. Native final-frame readout is preserved, with the inverse-sigmoid chart
round-trip error5.96e-8 in these queries. Qualification is not task efficacy.

Original HC qualification completed, then a second same-process GPU setup
nested the clean annotation-loader wrapper and triggered its strict two-file
interception assertion. The failed code/runtime/logs/status are saved at
`recovery/loader_isolation_001`. Revision001 uses one qualification worker per
dataset; the original assertion and scientific configuration are unchanged.
The existing six HC fits were retained and Vid qualification completed in its
own worker. Original runtime lock bytes remain, with explicit revision pins.

The finite controller is `scripts/continue_decota_spatial_opd_v1.py`; actual
PIDs/stage/log are in LAUNCH/STATUS, not fixed by this document. It runs each
phase's two dataset directions before the phase's CPU scoring, then proceeds
development ->128-source confirmation ->same-domain5% mechanism. There are
2,880 adapted arrivals over three independent arms. Each dataset has32 exposed
development parents and128 parents disjoint from this round's development.
All pools have historical exposure; none is described as a fresh dataset.

## Completion and saved paper resumption

All six stages must seal and finish their CPU audits before root report/figures,
independent aggregate checks, visual review and verified GitHub publication.
Root records an actual `ROOT_CLOSING_RECEIPT.json`; a prepared report script or
pending-root status cannot stand in for those actions.

`scripts/continue_decota_paper_after_opd_v1.py` requires that closing receipt,
the exact old resume proof, and original guards/runtime before it resumes TENT
and the remaining original Table1 arms. This continuation entry is prepared;
it is not running while OPD is active. The old HC source2000-media dependency
and Table2 parent/clip clarification remain pending. This study's parent roster
does not answer or release either boundary, and does not skip any old baseline.
