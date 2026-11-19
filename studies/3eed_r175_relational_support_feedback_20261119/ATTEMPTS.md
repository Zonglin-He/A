# R175 execution notes

The first source training attempt was stopped before target prediction. It supervised the support-confidence head only in the feedback arm, which would also have changed the shared training objective in the pair-only comparison. The frozen features and every source row were kept. The replacement runs use identical confidence calibration in all dual-labelled arms. The no-dual control masks the same term with its reference labels. The incomplete source outputs and the first log are retained privately. No target labels or protected inputs entered this correction.
