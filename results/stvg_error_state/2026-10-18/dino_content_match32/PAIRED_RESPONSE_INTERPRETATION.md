# Paired raw response and unchanged accepted feedback

Same-reference comparison changes the content row on1211/1224 HC and982/1104 Vid returned anchor references. This is an actual intervention, not a no-op. Every previously admitted reference already used the native target-high content row, so all admitted boxes and scores remain bit-exact; newly admitted0 and newly rejected0 in both domains.

| Domain | Raw known-anchor quality high minus nearest pp [95% query CI] | High raw correction versus input pp [95% query CI] |
|---|---:|---:|
| hc2 | +4.054891 [+1.304379,+6.960807] | +0.425133 [-1.901154,+2.513127] |
| vidstg | +8.999505 [+3.576394,+15.592488] | +1.639392 [-1.484690,+4.304608] |

Raw boxes improve relative to nearest, but target-high raw versus unchanged student inputs is uncertain in each domain. These raw statistics do not represent accepting rejected boxes, a lower-threshold method, or complete tube/OPD scores. The .35 gate is unchanged, and all actually usable corrections are identical. No fitted Student distribution was created.

High-content conditional score medians remain approximately .00603 HC and .00949 Vid, despite native frame top-score means .586/.617. A native query's score at its native reference does not transfer automatically to an arbitrary student reference. This matched content-only result narrows one missing control; it does not establish that reference position alone is the cause, nor that all decoder-conditioning interfaces fail. No alternate mask/content rule or threshold was tried.

Independent GT geometry recomputation validates5760 IoU scalars exactly after explicit FP64 conversion, using the original metric tolerance. The first CPU-only audit attempted FP32 coordinate arithmetic, creating a1.38e-6 discrepancy; original audit source and repair receipt are preserved privately. Scientific predictions, scoring and source pins were unchanged, and no model/GT-scoring experiment was repeated.

Current resource decision: stop this content-only variant before OPD. Preserve both the positive raw-response effect and the absence of new admitted evidence.
