# Prepared formal204 direct8 audit

Do not run on a partial directory. Root must first provide both complete actual seals and explicitly release this no-label audit. No polling or execution was performed during preparation.

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/absolute/local_home/mllm/.venv/bin/python \
/absolute/workspace/research/round15/evaluation/audit_direct_unlabelled.py \
--predictions /absolute/sealed/formal204_direct_directory \
--seal-sha256 ROOT_DIRECT_SEAL_SHA \
--semantic /absolute/workspace/crossplatform/round15/semantic_formal204_attempt001 \
--semantic-seal-sha256 ROOT_FORMAL_SEMANTIC_SEAL_SHA \
--manifest /absolute/workspace/research/round15/input/FORMAL204_GPU_INPUTS.json \
--cases /absolute/workspace/research/round15/input/FORMAL_CASE204_PRIVATE.json \
--output /absolute/workspace/research/round15/evaluation/DIRECT204_UNLABELLED_AUDIT.json
```

The auditor validates the complete formal204/612-call semantic seal, direct204×8 seal, eight shared source/mask seals, all member bytes and actual file sets, producer code hashes, actual plan/input joins, strict9key manifest, unchanged source pool/scores/q0, selected cache and mask byte identities, and compact metadata canonical bridge. It does not instantiate the VLM, grounder or G solver, decode images/point files, parse annotation containers or compute task IoU.

The semantic interpreter replays only already sealed text/visual records and compact scene metadata, under frozen v4 parser and 1024/2048 budgets. This checks recorded EOS/NA/allmask behavior and branch results without new inference. All statuses are retained. Model asset declarations and producer freeze flags are checked; no independent weight reload or second forward is claimed.

Every branch retains original source proposals for association and its own associated q1 for geometry. Main graph association/full/noRGB must share q1/anchor/mask/support/association gate. Geometry failure must return that anchor; accepted-axis and changed-axis arrays, full/off/half domain, no shrink and size≤1.2source, center bounds min(.25sourceSize,.5m), and nonapplicable flip full-result reuse are checked. A final float32-delivery bound allows only one outward ULP of the rounded bound, not an after-the-fact task tolerance.

Recorded exclusive support matrices are independently reduced to the fixed positive/foreign gates, reciprocal group uniqueness and source-score member selection. This is an internal numerical/provenance audit, not an independent recomputation of the raw point-mask geometry or target identity. It checks that all204 source expressions, fresh64/reg140 roles and four role/domain counts remain present; reservedB32 and old G endpoints are excluded. Coverage counts do not establish useful supervision or pass the task progression gate.

The receipt's `prediction_seal_sha256`, `semantic_seal_sha256`, `plan_sha256`, `semantic_plan_sha256`, `manifest_sha256`, `case_index_sha256`, `input_plan_sha256`, check count and per-domain/per-role counters bind actual artifacts. A failure raises and does not rewrite predictions or method code. Nine pure synthetic recorded-array tests exercise valid bounded geometry, wrong bounds/shrink/query/fallback/canonical readout, malformed counts and the complete eight-arm all-NA reader; they do not solve G or evaluate any real sample.
