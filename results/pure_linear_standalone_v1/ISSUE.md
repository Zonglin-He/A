# Pure interpolation implementation / evaluation contract

Confirmed in current source: `spatial_table_recheck_v1/common.py:rebuild` returns only `len(ids)` predictions on the Source sample grid. Official `TA-STVG/engine/evaluate.py:linear_interp` fills only the first-to-last provided frame range. Complete-video nearest-expert tails are therefore absent outside that sample grid. New standalone output covers every frame in the input segment from raw8corner boxes. Numerical effect on each published score must be measured; tail omission outside an event interval alone cannot explain that event's vIoU.

The old `PureLinearFixed_interval` explicitly equals `temporal['Ours_interval']`. This is an intentional same-Full-time spatial diagnostic, not a standalone temporal baseline; previous chat conflated these labels. The independent baseline uses original Source time, separate from the pure box estimator. An optional controlled Full-time score uses the same generated pure trajectory with an external evaluation interval. Neither time protocol is an input to the pure interpolation function.

This is not evidence that every aspect of the old linear formula or every old W3 statistic is wrong. Existing controlled comparisons and all prior arrays/reports remain immutable. New scoring establishes the corrected pure baseline rather than changing outcomes to match an expectation. No method/Source/expert/parameter/paper changes.

The first new wrapper also bound the legacy production Source_interval alias instead of current native Source. Its v1 SourceTime is a preserved failed-binding diagnostic. native_execution_v2 fixes only that external interval reference; verified/ reports are authoritative. No trajectory/FullTime regeneration or rescoring was needed.

Pure dense-unobserved support differs from old Full native-sample unobserved support. Unmatched unobserved differences in private intermediate reports are excluded from verified public comparisons. All reported vIoU/tIoU/recalls retain identical official dense annotation support.
