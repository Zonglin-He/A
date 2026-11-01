# R27 — one joint full-text score from target crop and scene
A real Waymo-only -> Drone/Quad frozen-candidate experiment. Four input controls:
same-prompt scene; higher-pixel-cap scene; two identical scenes; local RGB crop +
fullscene. A single frozen Qwen forward jointly reads both images and the complete
literal description, then gives one Yes/No logodds. No hand-added appearance,
position, reference, gray-image or source-relative residual components. Shared
2D/3D mapping and source+lambda*onecue. No parameter TTA or source-model training.

D208 exposed target-development selects one sharedlambda pervariant on a finite
equal grid. Primary crop_scene_joint fixed before outputs. Old400 and replay256
are exposed. NewQ64=32/platform, decodedRGB/XYZ disjoint recent history including
R26P64; allVAL historically scored and2Drone/3Quadscenes overlap. Not a pristine
blind or new-scene test. Fixed2 controls, FrozenSource, legacyG and R26old-prompt
scene retained. All negatives and aliases retained; comparisons not independent.

Read REPORT.md and CPU_AUDIT.json. Native processor grid/resize and local detail
are audited against actual model inputs, with prompt/caption/image hash binding.
38oldR26-regression winner-mask sizes are descriptive, not identity ground truth.
Upsampling cannot recover absent raw details. Higher input token budget and
repeated images are controlled, with actual tokens retained. No raw images,
captions, pointclouds, calibration coordinates, task IDs, features or weights
exported. Own gallery generator only, no generated raw-data panels exported.

Dependencies: previous studies through3386ef7e9fd7d4d4b05558dafdad9ad1a3389e55,
licensed3EED rawdata, fixed Waymo-only100, Qwen2.5-VL3B, MaskRCNN, Stanza and
GroundingDINO(legacyGcontrol only). No model upgrade following GitHub redirects.
Workflow: select27; capture27/maskcapture; parse27/prepare27/detect27/evidence27/
rank27; joint27 --probe thenjoint27; resolution27; readout27; evaluate27 grid,
select,predict,score; audit27; diagnostics27; finalize27; publish27.
Sealed output directories must not be overwritten. Initial CPU resolution run
missed the OLD400 manifest alias; repair changed no inference-input bytes.
