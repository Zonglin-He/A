# Frozen paper-review experiments (Fig.1 excluded)

This is an active supplemental suite, not a new production method. Current SimplePath/Native2D settings remain frozen. The latest user explicitly authorizes the review experiments, with local CG-STVG second backbone scheduled last.

- Cached temporal/prototype controls: all HC1902 + Vid10303; predictions seal before independent CPU labels. Source-box and frozen-Full-box tables explicitly isolate time and do not pretend to re-decode altered space.
- New spatial supports and first/final DP2x2: actual native grids,64screen+128fixedconfirm parents/domain. Altered auxiliary paths always repool and recompute time.
- Strong pseudo-box TTA: completed384; all positive/negative outcomes included. This four-residual GIoU/L1 adapter is a named baseline, not a VLOD implementation.
- Matched temporal LN entropy and photometric consistency, spatial/direction/reliability/sensitivity: complete. Horizontal-flip v1 was cancelled before inference; PhotoConsistency v2 uses fixed brightness. Same-expert observation budgets/schedules/dense: screening and fixed confirmation complete; controlled shifts running and CG-STVG last.
- Official full lists were previously examined. New screen and confirm are parent-disjoint for these contrasts; they are NOT previously unseen data. No target-label parameter reselection or main-method promotion.
- Failed dense configurations, incomplete coverage, exact-solver failures and actual cost are retained. Cached timing is not uncached deployment latency. Videos/captions/labels/coordinates/features/masks/checkpoints stay private.

See protocol.json and each family protocol for the complete task ledger, exact definitions and phase boundaries. A queued task is not an experimental result.

## Completed budget screening and current execution

See [all screening tables, costs, failures and actual figures](budget_v2/SCREEN_ANALYSIS.md). The128-query stage produced1660 valid predictions and4 explicit exact-search resource failures, then independent CPU scoring. Ten configurations enter fixed confirmation by completeness and cost only; DenseNative is retained in screening but exceeds the Vid cost rule. A score report-path failure was preserved and repaired with an independent CPU-only adapter; prediction hashes and both domain report/TSV bytes remain identical.

Current numerical budget fingerprint remains c5c24f739ebd160f8b252ddf5b9539edc6e2beda9cf3215c7532656d320dc422. New queue_budget_v4 and queue_shifts_v4 only repair execution dependencies. Prepared CGv3 fingerprint4d9f93321edb06201806dfd037b59375a03c5e8d7e303028684805f6725c7dde fixes its own screen receipt references and tracks current TA owners;5CPU groups passed, no CG model constructed. Previousv1/v2 source/plan snapshots remain unchanged. Actual CG execution is still LAST and pending.

## Completed fixed budget confirmation

[All10 confirmation configurations, paired CIs, severe tails, costs and figures](budget_v2/CONFIRM_ANALYSIS.md) are complete:256queries/2560predictions,0failures, officialdensechecksmax1.55e-15. ProductionK8 is unchanged; K16 positive outcomes are retained, not promoted. FixedRGBshifts have actually started, CG remains CPU-only preparation and LAST. Independent shifts_reporting.py is ready to derive signed/severe/shortlong/cost from already scored stages; it refuses unfinished stages and performs no newGT scoring.

## Controlled shifts screening complete

[AllfixedBlur/JPEG/Dim screen results](shifts/SCREEN_ANALYSIS.md) are complete:128queries,384same-transformedRGB Source/Full pairs,384actualSourcequeries+384sameReferDINOjoint8calls,0clean-cache reuse,768officialdensechecksmax1.67e-15. Allsigned/severeunion/shortlong/parentCIs/componentcost are public. Confirmation is running with unchangedviews/severity/settings. CG-STVG remainsLAST, modelnotstarted.

## Controlled shifts complete; CG native memory recovery

[Fixedshift confirmation](shifts/CONFIRM_ANALYSIS.md) finished256queries/768same-input pairs,1536officialchecksmax1.55e-15, allsigned/severe/shortlong/cost reporting. AllTA stages are now done. CG was actually startedLAST: original420FP32 attemptOOM before0completedSource/0expert/0prediction. [Allocation-only recovery](CGSTVG_MEMORY_RECOVERY.md) preserves nativeequations/settings and oldfailure, tests16CPU/GPU fixtures+512window bitwise, and runs independentv4. ActualfirstHCcheckpoint bridges pass; Vidbridge/CGaccuracy stillpending.


Latest execution (supersedes earlier running snapshots): controlled-shift confirmation is fully sealed/scored/reported; see [CONFIRM_ANALYSIS](shifts/CONFIRM_ANALYSIS.md). All TA experiments are complete. CG native420/FP32 v8 is screening, after preserved v3-v7 engineering failures and byte-exact compatible migration; see [recovery evidence](CGSTVG_MEMORY_RECOVERY.md). No CG quality conclusion exists before a complete stage seal/score. Main production and Fig1 are unchanged.

Current CG execution supersedes v8 snapshot: v9 fixes only documented input segment-boundary conversion before native sampling;65byte-exact compatible completed predictions retained, no GT scoring yet. See recovery document and cgstvg_v9_boundary_conversion.json.

Live v9 bridge verification: both Vid first-two real checkpoint/native/observer/pooling/coupled replay checks now passed; four required bridges complete across both domains. Screening has passed the corrected boundary case and continues unscored.
