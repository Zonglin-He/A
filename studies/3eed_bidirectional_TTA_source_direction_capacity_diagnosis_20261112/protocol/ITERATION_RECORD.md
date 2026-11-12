# R77 controlled iteration

Question: weak update versus auxiliary-to-grounding direction mismatch. No new loss, source checkpoint retraining, external model or FINAL access. The already source-trained bidirectional epoch6 module and exact R76 masks/objective/102498 trainable scope are fixed. The 128 Waymo VAL inputs were previously used for epoch selection; source labels are expressly allowed for controlled diagnosis, never target adaptation.

Competing explanations are tested by identical per-input Euclidean-radius edits in auxiliary and supervised directions; independent 10-step supervised fits of the same interface and copied matching head check capacity. All interventions include every input. Source soft target is the original candidate-IoU softmax with temperature0.1, not a perfect task oracle. GT controls can still hurt examples and must not be called TTA or generalization.

The predeclared source gate fails: tenfold auxiliary edit does not beat frozen or native one-step and only61/128 Adam directions reduce the linearized grounding objective. Consequently no enlarged-step target method is launched. All R76 DEV336 controls and negatives remain as explicitly historical context.

The current recipe is not adopted. Preserve the bidirectional static development candidate; stop magnitude-only escalation. A later source-preparation experiment should test post-auxiliary-update task alignment against ordinary joint training at matched budget, before any target claim. Not launched here.

Audit attempt001 had a dictionary iteration bug; analysis attempt001 omitted frozen endpoints from a mixed event table. Both source snapshots, logs, and partial tables are retained; only audit/reporting code was corrected. Model predictions, labels, scientific arms, thresholds, and numeric tolerances were not changed.
