# R71 fixed CLIP expansion

Claim: frozen CLIP's R70 projection-only candidate evidence improves grounding beyond source/ABC/R59 on the existing VAL1280 and TRAIN1024 cohorts. These cohorts were evaluated in R69; this is fixed-configuration expanded evaluation, not a blind independent test. No new data selection, thresholds, scale, crop, text parser, model, source training, adapter, loss or update-site search.

Unresolved comparison: the R70 best branch lacked a same-projection-weight visual mismatch control. Add it on cached DEV336 and on expanded cohorts. Permute similarities ONLY among projection-valid groups; leave the unknown mask, projection weights, source scores, two crops and text unchanged. This retains the observed visual-evidence score multiset on eligible groups and does not convert a missing visual observation into an artificial negative. Same deterministic R70 salt, no label-based permutation or acceptance. Center after assignment with unchanged weights.

Only R70 fields influencing projection-only G are computed: source A256 and scores, identical point-support/box complete-link groups, representative query, camera projection and clipped-area weight, literal subject phrase. Random RGB seeds/patches and its unused complex reliability are not computed. Validate equivalence on DEV raw inputs and all336 cached readouts before expanded inference.

Five arms: source, frozen ABC, R59, CLIP direct, projection-matched visual mismatch. Keep all unknown/fallback/negative observations. Seal predictions before existing IoU lookup; no original target annotation/GT geometry scoring. Independent all-input CPU correspondence/readout audit, prelocked stratified GPU reencoding, saved-scalar/scene/frame audits. Separate VAL1280, TRAIN1024 and DEV diagnostic, no favorable pooled selector. FINAL remains unread, source and all trainable adapters unchanged.

Outcome branches: positive net direct gains support retaining visual evidence for later adaptation research; mixed/platform-dependent results narrow its scope; no direct gain stops this fixed recipe. None establishes parameter-TTA benefit because no parameter update is run.
