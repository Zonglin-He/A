# R73 finite mechanism experiment

User-authorized change: fix a reproducible relation-span mismatch; separate image absolute position from entity-relative roles; keep source A256 and matching MLP, compare direct use and one objective-only backtracked update on a fixed 64-input mechanism set. No source retrain, external-feedback model, new loss, new update location or FINAL access.

The review ZIP is not locally attached. Independent implementation passes 14 synthetic grammar/interval tests; that does not establish task performance. Whole-list parser coverage is text-only and explicitly separated from the condition-selected task subset.

Real pre-score findings: 164 repaired direction-span comparisons and 72 recovered literal target-color requests; finite grammar identifies 1,051 image-absolute and 13 entity-relative descriptions. Fixed SHA selection gives 16 inputs in each VAL/TRAIN × Drone/Quad cell using text/projection conditions only. New position negatives exist in 61/64 inputs, but reject zero source winners even before group aggregation.

Run: raw metadata camera geometry plus frozen source caches, one Adam proposal, fixed nine-scale label-free line search. 56 accepted / 5 rejected / 3 no proposal; every accepted objective decreases, real weights change, rejected states restore exactly. All eight-arm predictions sealed before saved candidate-IoU lookup.

Measured: no main-method Top1 change or task gain. Forty source Acc@25 failures, 26 with a known projected winner already satisfying the coarse halfplane. These are localization diagnostics, not independently certified wrong-instance identities. All positive/negative comparisons retained. No new raw-GT scoring or source full forward.

Decision: retain implementation correction and objective-monotone acceptance safeguard; do not adopt current position TTA or expand performance evaluation. Scope remains a small, historically exposed, condition-selected mechanism set. No new margins, grammar, optimizer grids or layers are introduced after task scoring. No follow-on runs queued.
