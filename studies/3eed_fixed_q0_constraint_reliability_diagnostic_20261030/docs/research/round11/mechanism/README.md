# R11 unlabelled mechanism reduction (author-side)

Prepared only; run only after the actual R11 geometry/reliability seal and root-issued binding release. No new model/G/reliability/segmentation execution is performed. The helper reduces bound seals, trace, cohort-control lock, actual-weight bridges and an explicit selected-array NPZ allowlist. It never decodes RGB masks, images, points, observations/features, source prediction pools, annotations, private target indices or scores. Geometry sealed file bytes are hashed for integrity without decoding excluded fields.

Release exact keys: `mechanism_reduction_permitted=true`, `GT_read_permitted=false`, `reducer_sha256`, `geometry_dir`, `geometry_seal_sha256`, `source_dir`, `source_seal_sha256`.

Run after release only:

```
CUDA_VISIBLE_DEVICES='' crossplatform/env/bin/python -B research/round11/mechanism/reduce_mechanism.py --release ROOT_RELEASE.json --release-sha256 HASH --output NEW_OUTPUT_DIRECTORY
```

Outputs: aggregate SUMMARY.json/md and CASES_PRIVATE.json with private IDs and mechanism scalars/weights (no coordinates, captions, physical scene/frame IDs, masks or media), bound by a new SEALED.json. Public aggregates retain all denominator counts. Author explanation is not independent quality validation.

Checks/reductions include base actual-change C; weighted/uniform output changes; association statuses inside/outside C; raw and post-gate face/axis/RGB definedness and zero weights; case score exact distinctness/ties/NA; active face and RGB uniform mean denominators independently reconstructed with `math.fsum`; selected NA/conflict counts for each exact-coverage control; actual r1/R10 and r0/source bridges; prior-only analytic results versus numerical solver convergence/failure; unique physical partition hash/count bindings and shared-source/RGB limitations. Spatial graph purity is only diagnostic; substantial prior-incompatible components and ambiguity factors are reported separately.

Association_supported is a source-conditioned geometric compatibility hypothesis, never verified literal-text identity. A/B hashes/counts plus locked implementation bind physical disjointness; this reducer deliberately does not reopen points to independently reprove it. Effective scores are primary. Raw positive evidence zeroed by gate, defined conflict0, association-undefined NA, and top-half NA filling remain explicit. Decreasing loss/stability/changed outputs are never called correct localization.

CPU fixture `check_reducer_cpu.py` uses artificial arrays and dictionaries, with supported/conflicted/undefined examples to test gating, missingness, ties, selected NA, prior-only status and the true-changed count. It opens no actual outputs.
