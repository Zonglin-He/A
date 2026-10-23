# Frozen selected method

Both directions use the same fixed algorithm. Frozen TA-STVG Source at height224
runs its original two offsets, precision, text/subject parsing and native two-pass
process. One original ReferDINO supplies the same eight observations. Native complete
boxes, initialization calibration/gamma, correspondence, bidirectional affine complete
box propagation and all spatial legality/fallback rules are unchanged.

The temporal prior q0 is Global6: each layer uses the original two-offset interval
hull on physical frame IDs, and six normalized posteriors are arithmetically averaged
at temperature1. Obtain the auxiliary full-time BoxFixed path using the existing
geometry/score. Expand each target box by1.5 and retain overlapping visual-grid cells.
Use these cells to renormalize the original Source ASA motion word-head attention.
Full caption, original fused memory, grid and trained head stay fixed. No new visual
encoder or expert is introduced. These are word-class responses, not calibrated
frame-event probabilities or foreground/actionness feedback.

Use q0 soft interval membership w_t. Fit positive and negative Bernoulli means on
the opposite offset. With mu0 the fitting fold's unconditional semantic mean:

    mu_plus  = (sum(w*p) + 16*mu0) / (sum(w)+16)
    mu_minus = (sum((1-w)*p) + 16*mu0) / (sum(1-w)+16)

Classwise expected Bernoulli log-likelihood ratios are summed, then summed over
inclusive interval nodes. Form q(I) proportional to q0(I)*exp(S(I)) ONCE. Use the
original physical expected-IoU MBR and unchanged BoxFixed DP on that interval.
No second prototype estimation, gain/beta adjustment, query update or network training.

Spatial objective remains .5 mean observed support + .5 mean unobserved confidence
support + .25 mean confidence-weighted event-edge connection - .1 mean geometric
protection. Denominators, exact three-state DP/ties and complete native extents remain.
Sparse expert masks support correspondences; they are not thresholded into final boxes.

This profile is selected explicitly by the user after exposed target-development tuning.
Prior HC short-event validation failed and is retained. Selection is not a claim of
statistical validation, full-result improvement or new event supervision. The independent
full plan must evaluate all1902HC/10303Vid queries before labels are read per domain.
