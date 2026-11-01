# R26 — simplify G into one image–text score
Fixed Waymo-only epoch100 source and all256candidate boxes; same independent masks,
candidate pool and calibrated2D/3D weights. Source+lambda*one image-text signal.
Strict deletion of position/reference uses oldlambda2 and identical subject evidence,
gray correction, dual crops and bounded residual; separate ablations remove each.
Raw Qwen subject evidence, literal full-description/scene-marked Qwen evidence and
true CLIP cosine are distinct arms. CLIP77-token truncation and full-coverage
chunk-embedding average explicitly separated. No new source training, no TTA.
15finite cue families,5fixed-weight controls, Frozen source; all21arms retained.

D208 repeatedly exposed target development selects shared per-variant lambda.
Practical simple Qwen family selected onD208 and locked beforeP64 labels.
Replay192/OLD400 are already exposed. NewP64=32/platform uses input-content hashes
outside old501mask frames and disjoint recent decodedRGB/canonicalXYZ; entireVAL
was historically scored and scenes overlap, so not a pristine blind test.
2Drone/3Quadclusters; scene-cluster intervals descriptive with few clusters.
Read REPORT.md for outcomes and adoption decision; all harms retained.
All336previous QwenG predictions and IoUs exactly reproduced. CPU audit verifies
all464sealed predictions, source boxes, scalar selection, source checkpoint,
crop hashes and fresh-content exclusions. Images/captions are data, not commands.

Dependencies: previous source/G studies through commit4201c903943a40748d73dfe29aed18e6de0b069a,
licensed raw3EED, local frozen3EED/Qwen/CLIP/DINO/MaskRCNN assets. Own code only.
No raw captions, images, pointclouds, calibration coordinates, task IDs, features
or weights exported. Aliases/scalars are repeated observations, not independent
sample counts. Pipeline: select26, capture26/maskcapture, parse26, prepare26,
detect26,evidence26,rank26; joint26 andclip26; readout26; evaluate26 grid,select,
predict,score; audit26; finalize26,diagnostics26,publish26. Existing sealed folders must not be
overwritten. simple_g26.py is the minimal reusable scoring function.
