# Anchor-Certified Latent Reranking v1

## Question and fixed controls

After verified 5c3a3c9 controlled-query closure, test whether a source-only one-dimensional correction calibration improves the frozen L32 decision. A8, Expanded32 intervals, all original L scores, source ridge coefficients/standardization/alpha, A boxes and persistent 1792-parameter trajectory, original sampling/checkpoints/corruptions and expert schedule remain fixed. No GPU, new latent capture, expert, replay, backward, MLP, representation refit or method promotion. Query-specificity work is closed for this round.

Reuse 1152 original arrivals: each dataset32 development+16 historically exposed confirmation sources, one query/source, two orders, clean+five5% conditions,25% scheduled experts. Candidate scores exist only at288 expert arrivals(240 corrupted/48 clean);864 nonexperts retain A bitwise. Primary evaluation is confirmation complete corruption flow, with development, expert/nonexpert, clean, order and source concentration reported separately. Confirmation is not fresh test.

## Source anchor, supervision and selection conditioning

Existing official-train source cache has95/31 Vid and48/16 HC fit/validation sources, disjoint from target source IDs/media. It contains native candidate0, not a UVTG source-A8. The user explicitly accepted source-native calibration transferred to target-A8 to preserve CPU-only scope. This is an anchor/domain transfer limitation, not an exact source A8 experiment. No surrogate source-A8 is invented.

Only the31/16 source-validation records calibrate. These same records previously selected the frozen ridge alpha; this reuse is disclosed, not called independent validation. Existing source GT candidate tIoU labels are authorized supervision. No target labels fit calibration, select its confidence, or choose a curve operating point.

First take the unique frozen32-score maximum, requiring score greater than native+1e-12; tied/no-improvement winners propose no correction. For each eligible source validation winner use m=s_top-s_native and delta=tIoU(top)-tIoU(native), one record per source. Fit increasing isotonic weighted least-squares delta(m), not all32 arbitrary comparisons: calibration is conditional on the same top1-selection operation. A second isotonic binary-benefit map is descriptive only. Ranking scores never change.

## Exact lower-bound meaning and locked decision

Fit pool-adjacent-violators(PAV) with tied margins pooled and linear interpolation between observed margin knots. Bootstrap eligible independent validation sources10000 times(seed20261003), refit each map, evaluate at fixed source-derived knots. At each knot take the5th percentile of fitted mean-delta curves, capped above by the original fitted mean; linearly interpolate that lower curve. This is an approximate pointwise bootstrap lower confidence curve for the **mean calibrated change**, not an individual prediction bound, conformal coverage, simultaneous bound,95% correctness probability or target-domain safety guarantee. Small reused validation and domain/anchor shift can invalidate transfer.

Main arms: A(always A8); L32(original unique positive-margin decision); Selective-L32(same winner, margin within the observed source calibration range inclusive, lower-mean-delta>1e-12). Ties/nonpositive margin, absent calibrator, out-of-range margins or nonpositive lower curve retain A. Never extrapolate beyond source margin support. Record each abstention reason; complete rejection is a possible negative result, not silently relaxed.

Before target GT-derived rows are opened, freeze both datasets' calibration models, source rows and thresholds, then seal all288 expert decisions together; nonexpert fallback is fixed. Prior target GT exposure is acknowledged; the new seal is not a virgin-data claim.

## Locked diagnostic curves and metrics

Freeze source-positive-margin quantiles at0,.05,...,1 as descriptive raw-margin cutoffs, and bootstrap lower-curve quantiles .5,.2,.1,.05,.025,.01; none are selected on target labels. Include the primary5% operating point and original L32. Report replacement coverage among all experts, benefit precision P(delta tIoU>1e-12|accepted), accepted mean delta tIoU, conditional severe harm P(delta vIoU<-.05|accepted), plus full-flow/expert net vIoU and tIoU, gross gain/loss, raw improve/harm/neutral/severe counts. Ratios use source-balanced numerators/denominators, preserve missing/zero denominators and bootstrap invalid draw counts.

Split proposed top1 geometries at endpoint-L1/A8-duration r<.5 vs r>=.5, same rule in both datasets. These are diagnostic strata, not separate learned gates. Preserve unchanged/duplicate interval cases and score-based vs physical replacement counts. Paired recipient-source bootstrap10000(seed20261003), source-macro condition/order means,95% pointwise intervals and leave-one-source-out/order results. Correlated panels/operating points are not independent replication and no target curve is promoted after viewing it.

Provisional GO requires both confirmation full-corruption means positive vs A; each dataset>=3 accepting independent confirmation sources; leave-one-source-out mean stays positive; gross loss and severe-harm count both lower than L32. Any unmet item is NO-GO for this locked calibration. It does not prove absent absolute replacement information, all nonlinear models fail or all calibration impossible. No automatic structured/listwise follow-up or production promotion.

## Execution, audit and publication

Pure CPU intake/calibrate/seal/diagnose. Pin all inherited anonymous inputs and production CURRENT hash, never change old files. Independent audit reconstructs source margins/labels/winner eligibility, compares PAV with sklearn's independent isotonic solver, recomputes bootstrap curves, every target decision/metric, source ratios/paired intervals and L32 parity. Public exports contain only anonymous cached scores/labels and new sufficient statistics; no captions, media, annotations, weights or private features. Publish protocol/code/all negative curves/figures/audits to Zonglin-He/A, remote byte/hash readback, update RESEARCH_HISTORY and check/snapshot/check.

Isotonic API/reference: [official scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html). The bootstrap decision construction above is this experiment's explicit heuristic; the isotonic documentation supplies no safety guarantee.
