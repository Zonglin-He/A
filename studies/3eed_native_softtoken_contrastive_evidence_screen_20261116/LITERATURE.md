# Original implementation authority

The released [3EED evaluator](https://github.com/worldbench/3EED/blob/main/src/grounding_evaluator.py) separately implements `evaluate_bbox_by_span` (BBS soft-token classification) and contrastive BBF. Its native BBS denominator includes all256 token positions, including the no-object category. The original evaluator uses annotated positive_map; this experiment DOES NOT use it, and instead applies the exact current inference-only text mask to both heads.

[Official losses](https://github.com/worldbench/3EED/blob/main/models/losses.py), `loss_labels_st`, supervise matched queries with token span distributions and unmatched queries at last category255; this head is independently parameterized from contrastive query/text projections, but SHARES learned query representations and source supervision. It is not an independent expert, verified identity, calibrated teacher or new modality. R4 already tested BBS on an old checkpoint/cohort and found it weaker; R108 is a prerequisite check in the current three-source/native LiDAR setting, not a novelty claim or a universal untested-route claim.

All source checkpoints were trained with official CE + box + contrastive + seed objectness objectives. Seed objectness selects near-centroid points and is NOT a ready-made per-final-box quality estimator. Therefore it is not silently multiplied into the score.

If the frozen source-supervised alternative does not provide useful candidate preferences on the complete development cohort, fitting its outputs is not justified. No source retraining or another generic self-training loss is launched by this screen.
