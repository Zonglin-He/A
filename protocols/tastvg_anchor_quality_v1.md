# Anchor-score conditional delta: matched CPU experiment

Reopen the frozen scalar evidence only for the user's proposed extra anchor-score feature. First verify good/failure cases in previously exposed labelled caches, then compare two source-only linear models. No new temporal/spatial evidence, candidate, parameter update or GPU execution is authorized here.

## Frozen boundary

Official same-domain TA-STVG checkpoints Vid fbb1ed88 / HC ee72f0d9, original two-offset pixels, Old8 contained in Expanded32, unchanged frozen L scores, A's 1792 spatial parameters and trajectory (Vid K1 / HC K8), Uniform expert schedule and 25% arrival rate remain fixed. Source validation 31/16 already selected ridge alpha and earlier calibrations; no new independent validation. Target per dataset: 32 search + 16 confirmation sources, one query/source, two orders, clean + five 5% corruptions. Combined 1152 arrivals, 288 expert (240 corrupt/48 clean), 864 nonexpert unchanged A. Both panels have extensive historical GT exposure.

Prepare pins before baseline-case diagnosis. That diagnosis reads only existing anonymous scalar GT metrics, not media/raw annotations. Record this exposure explicitly. Source fitting runs in a separate process forbidden to parse target labelled files or baseline diagnostic outputs. Freeze both source models and all new target choices before joining target labels for their evaluation. This ordering prevents outcome-dependent new choices, not earlier knowledge or a pristine test claim.

## One added predictor

On each source validation candidate pool with a unique maximum L score k, retain **all 32 pseudo-anchors j, including j=k with y=0**. Keep negative/neutral GT deltas and duplicate intervals. An ambiguous top excludes that whole source with an explicit count. Every nonempty source has total weight one, equally distributed over its pseudo-anchors.

Use NumPy linear IQR, denominator IQR+1e-8 and score tie tolerance 1e-12:

    m_j = (s_k-s_j)/(IQR+1e-8)
    a_j = (s_j-median(s))/(IQR+1e-8)
    y_j = tIoU_k-tIoU_j
    M0: y = beta0 + beta1*m
    M1: y = beta0 + beta1*m + beta2*a

Both use source-balanced unregularized weighted least squares, NumPy SVD with fixed rcond=1e-12. Record rank, singular values and condition number; retain deterministic minimum-norm solutions if deficient, without claiming unique coefficients. No MLP, interaction, ridge/threshold/q search, new latent, P/R feature or extra expert. Leave-one-source-out predicts all pseudo-anchors from a model fit to the other sources. Report source-macro MSE/MAE, paired M0-M1 MSE with 10000 whole-source bootstrap seed 20261003, positive-source count, fold coefficients, and eligible non-self errors separately. Do not select a model or change the decision rule from LOSO outcomes.

Both locked target rules use the same unchanged unique L winner, strictly higher than A8 by 1e-12, accepting it iff predicted mean delta >1e-12; otherwise retain A8. This is a zero expected-delta rule, **not** a lower confidence certificate. No domain clipping: record marginal source-feature-range extrapolation without using it as another gate. M0 is a new matched linear control, not the previous isotonic Pair-Norm; include prior Pair-Norm descriptively.

## Interpretation boundaries and diagnostics

Exactly within one pool, m+a=(s_k-median(s))/(IQR+1e-8), a source/pool constant. M1 adds top-score context across pools; it does not add an independent measurement of anchor correctness. Record this identity, design rank, and proxy-to-GT correlations. Also delta tIoU=t_winner-t_anchor contains minus t_anchor; negative correlation alone cannot establish anchor-quality blindness. For diagnosis only compare its correlation to the exact mean-over-32-candidates delta, illustrating this arithmetic coupling. Neither GT anchor quality nor GT strata ever determine an online choice.

Baseline and new-arm corrupt expert strata: sort by A8 GT tIoU then stable cell key, split into four equal-count groups per dataset/panel, including no-op cells. Report cell means/counts (to check the attachment) and source-balanced means/10000-source intervals. Top quartile harmful L proposals rejected and bottom quartile useful proposals retained are diagnostic endpoints. Report all four quartiles, source coverage and uncertain/negative outcomes. Full-flow corrupt and clean metrics, expert/nonexpert, per-order/source/leave-one-out, gross gain/loss, severe delta vIoU<-0.05, acceptance and beneficial precision use unchanged cached official dense metrics. All new-arm comparisons use paired source bootstrap.

Prelocked development decision: M1 requires source LOSO MSE improvement over M0 in each dataset, confirmed full-corrupt vIoU mean >0 and M1-M0 mean >0 in each dataset, >=3 accepting sources/dataset, positive leave-one-source-out utility, and lower confirmed gross loss/severe count than L32. These are descriptive development checks, not significance, deployment approval or causal identification. Report CI even when a point check passes. Failure (especially HC<=0 or negligible acceptance) stops this locked scalar branch. A possible structured relative model remains unspecified and is not automatically started.

## Delivery

Isolated code/configuration, complete positive/negative anonymous scores and metrics, independent SciPy least-squares/LOSO/decision/aggregation/seal audit, figures, report, research archive check/snapshot/check and verified Zonglin-He/A publication. No GPU, new expert calls, replay/backward/update, model downloads, recurring monitor, historic queue restart or CURRENT promotion.
