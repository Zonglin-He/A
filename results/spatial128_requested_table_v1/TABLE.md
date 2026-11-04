# Original fixed128 spatial comparison

| Spatial trajectory | HC vIoU | HC Unobs. IoU | Vid vIoU | Vid Unobs. IoU |
|---|---:|---:|---:|---:|
|Expert interpolation|29.35|47.87|14.91|35.92|
|Our path|31.79|54.64|15.34|38.21|
|Δ (Our path − interpolation)|+2.45|+6.77|+0.43|+2.29|

Each dataset uses exactly its original fixed128queries/128distinctparents and same eight original ReferDINO observations, native sampled frame IDs and Current Full interval. Per-query tIoU is equal between paths. The user explicitly selected center-linear/log-width-height interpolation, nearest rawexpert frame outside coverage, exact8expert outputs. This is not the initialized-E-tail historical Interpolation and not the ordinary corner-linear variant. No E/Source/proposed geometry inputs enter this selected interpolation estimator.

Unobserved IoU retains the ORIGINAL audited column definition: query-equal IoU on GT-supported native sampled frames excluding exact eight observed positions, across the video rather than restricted to the predicted event. All128 queries/domain have support. This must not be mixed with dense inside-event IoU. The same native GT frame support is shared by both methods; inherited frame audit and receipt hashes verified, no new GT evaluation.

Both point-estimate gains are positive on this existing panel. HC paired v/unobserved CIs are positive; both Vid CIs cross0. Original fully exposed panel, no reselection, parameter tuning, new Source/expert/GPU/model/GT score or method/anchor change. These128results do not replace the separately preserved full-list results. All signed outcomes remain in TSVs.
