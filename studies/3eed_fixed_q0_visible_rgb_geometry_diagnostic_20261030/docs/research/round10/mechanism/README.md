# R10 author-side unlabelled mechanism reduction

Prepared only. Execution requires a root-authored release bound to the actual sealed geometry output and source seal. This is an implementation-author explanation of observed mechanism activity, not an independent audit or task-quality result.

The helper only opens the bound seals, bound geometry trace and the explicit array keys in `ARRAY_KEYS`. It verifies every geometry sealed-file hash but does not decode RGB masks, images, raw point clouds, source observations/features, annotation containers, scoring sidecars or target scores. The source seal is read only for provenance binding. The geometry trace may contain source-conditioned surface diagnostics; the reduction exports no plane medians, source/target coordinates, captions or physical scene IDs.

It reports all-case and platform-level counts for segmentation states/reasons; physical point/support sizes; acceptance versus actual float32 changes; qualified opposite-plane pairs versus stricter size eligibility; source/final RGB containment terms and solver success/nonconvergence/numerical fallback; full-versus-noRGB-coordinate equality/difference; RGB support retention versus unmasked geometric support and pointmask-ablation increments. Absent solvers and all-source fallbacks remain in sample denominators. Loss-term summaries separately report their executed-case count. Disabled RGB terms are marked as zero by construction, not inferred to be satisfied.

`SUMMARY.json` and `SUMMARY.md` contain aggregate quantities; `CASES_PRIVATE.json` contains only private IDs and mechanism scalars/flags (no raw boxes, plane positions, captions, physical scene IDs or media). `SEALED.json` binds these three outputs. No claim of correct identity, true amodal completeness, task improvement or held-out prediction follows from these diagnostics. Pointmask ablation inherits RGB acceptance and is not deployment PC-only.

Root release exact keys:

```
mechanism_reduction_permitted: true
GT_read_permitted: false
reducer_sha256: <reduce_mechanism.py hash>
geometry_dir: <sealed geometry output>
geometry_seal_sha256: <its SEALED.json hash>
source_dir: <bound source output>
source_seal_sha256: <its SEALED.json hash>
```

Execution after that authorization only:

```
CUDA_VISIBLE_DEVICES='' crossplatform/env/bin/python -B research/round10/mechanism/reduce_mechanism.py --release ROOT_RELEASE.json --release-sha256 HASH --output NEW_OUTPUT_DIRECTORY
```

Synthetic fixture: `check_reducer_cpu.py`. It checks missing solver/fallback denominators, acceptance versus change, and separate finite-nonconverged/numerical-invalid counts. It does not open real experimental data.
