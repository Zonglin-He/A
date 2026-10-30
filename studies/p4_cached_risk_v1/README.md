# Cached P4 risk statistics

2026-10-30 direct user scope: reuse completed P1 and P2 results, compute P4 on CPU only, defer full P2, no manuscript changes. The new scope does not resume any paused experiment or automation.

Current shared-corrected-ROI method and native Source metrics are reused from audited full-target outputs: HC1902/227parents, Vid10303/732parents. No model, Source, expert, optimizer, prediction, RGB decode, raw feature replay or new GT scoring. All signed scalar results are preserved, including null/negative groups.

`run.py prepare` freezes code/input SHA and definitions before statistics. Negative delta uses the existing1e-10percentage-point tie tolerance; severe is exactly <=−10percentage points; fifth percentile uses linear query-level quantile. Mean/proportion95%CIs use10000parent-cluster resamples, preserving query-equal means. Existing expert-quality quarter bins and missing-anchor groups are reused; Vid form labels use exact original-video/caption metadata joins.

Run low priority with CUDA hidden, CPU14-15, nice19, idleIO and one BLAS thread: `run.py derive --dataset hc2|vidstg`, then the separate `audit_v1.py --dataset hc2|vidstg`, followed by `root_export_v1.py`. The scripts reject GPU-device, checkpoint/raw-prediction, video and GT-label reads. Private inputs remain local; public outputs contain anonymous scalar rows and protocols only.

Results: `results/p4_cached_risk_v1/ANALYSIS.md`. P1 is reused at full scale; P2 Local/Global remains an existing192-parent/domain result, with full-domain pure-prior/smoothing controls separately reused. No changes to paper/Fig1, methods, parameters or old frozen files.
