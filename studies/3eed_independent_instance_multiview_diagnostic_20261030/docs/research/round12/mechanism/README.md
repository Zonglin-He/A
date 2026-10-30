# R12 author-side no-GT mechanism census

Prepared only; do not execute until root supplies the actual8-arm G seal and hash-bound release. No G/source/model/segmentation reexecution is performed. Root release exact keys: `mechanism_reduction_permitted=true`, `GT_read_permitted=false`, `reducer_sha256`, `geometry_dir`, `geometry_seal_sha256`.

```
CUDA_VISIBLE_DEVICES='' crossplatform/env/bin/python -B research/round12/mechanism/reduce_mechanism.py --release ROOT_RELEASE.json --release-sha256 HASH --output NEW_DIRECTORY
```

The helper reads final G seal/registered plan/trace and an explicit selected-prediction array allowlist (`KEYS`). It verifies program/file/source/mask-seal hashes. It does not decode source prediction pools/features, raw images, masks, points, annotation containers, target/scoring indices or task scores. Input frame-index and pairmap content are not reopened; their existing hashes bind observation selection. Old source-module snapshots and three input-only probe plans are byte-hashed solely to retain lineage, not parsed/re-evaluated.

Outputs SUMMARY.json/md are aggregate all/platform census; CASES_PRIVATE.json includes private IDs/aliases and scalar counts/weights/residuals/hash bindings only (no box coordinates, media, captions, raw planes or point coordinates). SEALED.json binds all three. This is implementation-author mechanism explanation, not independent quality evaluation.

Funnel denominators are explicit: preselected291 real-view roles/146wrong roles (as actually present in final trace), actual attempts after anchor/noun eligibility, globalPnP pass with background-exclusion protocol, unique class-family LK track,>=6unique depth pairs, static support/contradiction/insufficient-depthNA, per-anchor admitted multi-view. Unattempted view roles are not registration failures. The census reports actual counts rather than assuming all retrieved roles ran.

For8arms, report accepted/actualchanged axis, center/size changes, statuses and actual solver states. Half-displacement arms add no solver; pure source fallback/no plane is distinct from numerical optimization success. Main and legacy-named unfiltered ablation must share track/static admission and mask, but their normalcontext/point support and accepted plane sets can differ. Report pointwise pixel/depth/3D filter counts, matched-anchor raw-ID union, allowed canonical model support, source-maskROI support fraction, merged unique raw points and main-versus-unfiltered prediction differences. Confidence, accepted coverage and residual decline do not indicate task correctness.

CPU fixture uses artificial rows to validate attempted-vs-retrieved counts, the multi-stage funnel, contradiction veto without losing single-frame output, identical admission/main-vs-unfiltered difference, distinct support unions, half/no-extra-solver and fixedq0. It opens no actual outputs.
