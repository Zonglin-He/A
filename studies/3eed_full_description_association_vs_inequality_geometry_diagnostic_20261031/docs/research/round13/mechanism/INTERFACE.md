# R13 output-only mechanism reduction (prepared, not actually executed)

CLI:
```
CUDA_VISIBLE_DEVICES='' crossplatform/env/bin/python -B research/round13/mechanism/reduce_unlabelled.py \
 --association FINAL_A116_DIR --geometry FINAL_B116_DIR --engineering FINAL_ENGINEERING2_DIR \
 --manifest ORIGINAL_STRICT9_MANIFEST --cases CASE116_PRIVATE_JSON \
 --release ROOT_RELEASE --release-sha256 ROOT_RELEASE_SHA --output NEW_OUTPUT_DIR
```

Root release keys: mechanism_reduction_permitted=true, GT_read_permitted=false, stage=`R13_all116_noGT_mechanism_reduction`; reducer_sha256; association_seal_sha256, geometry_seal_sha256, engineering_association_seal_sha256; manifest_sha256, cases_sha256; nonempty program_bindings/input_bindings path→SHA maps. Input seal types must match actual completed formal A, B and first2 engineering scopes. Formal A has six calls per case and engineering has exactly12; all native UNKNOWN, invalid parse, consensus rejection, mapping failure and zero-change outcomes remain in their denominators.

No A partial/run-in-progress file is read. A/B/engineering seals and all members are integrity-byte-hashed; media/points are never decoded. Runtime decoded NPZ key allowlists:

- B: sample_id,arm_names,arm_query_index,arm_boxes_raw6,arm_status,q_source,active_coordinate_mask,actual_changed_axes.
- A/engineering: sample_id,arm_names,arm_query_index,arm_boxes_raw6,source_q0.

The reducer does not access B_support_model_points/source sample points/features/pools, actual image/mask arrays, any task GT/index/IoU/score, or model weights. It reads final sealed A case JSON only for grouping/mapping/call/readout scalar records, and B TRACE for support/inequality scalar diagnostics. It does not inspect the old-surface raw observation data or optimize any box.

Outputs SUMMARY.json, CASE_MECHANISM_PRIVATE.json, SEALED.json. new_content108 and regression8 remain separate, stratified drone/quad plus an additional all-platform count summary. Private case records use original private IDs; no automatic export.

B reductions: unique compatible independent mask, support-status and min8, initial point/RGB constraints satisfied/violated/NA, four violation classes, shared eligibility/search domain, actual center/size/axis movements, full-vs-noRGB final equality, per-arm statuses/solver convergence and half-from-nonconverged lineage. Active six-dimensional search is not six independently observed coordinates; source shape prior is not amodal truth.

A reductions: all256 complete dedup groups/size, total and mapped masks, geometric mapping failures, six strict parse/UNKNOWN statuses, physical-mask consistency across two numberings, raw order0 vs full/noun/deranged consensus changed queries/boxes, and selected-mask differences (both-defined denominator separate from NA-inclusive differences). Raw answer identity is reported only as a response property. Same candidate IDs across renumberings need not refer to the same mask.

Formal696 and engineering12 costs are separate: call input/generated/visual tokens, EOS/max-token-cap counts, model-generation seconds, formal/engineering process wall time, formal peak allocated memory. No text truncation is enabled in the locked producer; visual resizing is a distinct operation.

Eight synthetic structure checks passed before real output release. They are not A/B utility, identity quality or actual coverage results. This file does not grant execution; root's hash-bound release is required.
