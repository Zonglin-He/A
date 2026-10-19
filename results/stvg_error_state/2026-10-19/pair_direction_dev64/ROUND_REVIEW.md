# Root audit and existing GPT6.1 Sol read-only review

Scope: completed cached direction diagnosis only. No new-model permission or method promotion is inferred from this review.

The existing `/root/fine_feature_sol_review` GPT6.1 Sol/max reviewer read the locked code, measured public/private decomposition outputs, conditional supplement and REPORT. Its final read-only decision was pass: no material scientific blocker requiring rerun or alteration of the core results. Root independently evaluated the findings and performed the scalar/aggregation audit reported separately.

Review findings incorporated:

- Raw and posterior direction accuracy are identical for equal-radius antithetic pairs sharing the prior. This is expected, not an extra softmax improvement.
- Finite-pair correctness is relative ranking, not a local GT gradient. Both endpoints may harm current while a small combined step still helps.
- Core 17-point headroom originally ignores the valid support mask. It is retained as geometric headroom; a separately disclosed post-core supplement computes valid-support headroom. They happen to match here, but their definitions are different.
- Marginal opportunity and accuracy rates do not identify one bottleneck. The supplement cross-tabulates headroom strata, direction classification and update norm mass.
- Candidate quality improvement relative to the prior must be separated from quality relative to current and actual single-box changes.
- Correct-vector opposition and cancellation are geometric diagnostics, not additive GT loss attribution. Unknown GT contributions remain unknown.
- Formal FH/FV64 is protected and excluded from new scoring or method choice.

The only NEXT proposal is an unexecuted matched readout control: q-MBR and p-MBR on the same frozen final-round 17-candidate support versus the saved Gaussian mean output. Including p-MBR is necessary to distinguish teacher information from a generic readout/current-retention effect. Current means that round's sampling center and is not necessarily E; q is constructed working weights, not a calibrated posterior. This control is not proven useful, and failure would not uniquely identify teacher error. Old shared-trajectory correction-distribution negative results remain in force.

Root engineering/scientific audit: all 128 actual steps, 24,320 pair classifications, 26,690 scalar official candidate IoUs, 64 original-denominator final spatial telescopes and 102 parent-bootstrap summaries pass. Actual-step max error 1.665e-16; scalar candidate IoU error zero. Unknown/tie/support-limited cases and all 64 parents retained. No GPU/model/expert/DINO/training/new OPD action.

A report-only rerender encountered the existing-file guard for unchanged COST_BREAKDOWN. The guard was preserved; the renderer now checks equality before skipping an existing JSON. No core output, prediction, label, metric or frozen protocol was changed or rerun to resolve it.

CURRENT/MAIN/PAPER unchanged. Broad exploration and heartbeat remain paused. The new readout proposal was not implemented or executed in this run.
