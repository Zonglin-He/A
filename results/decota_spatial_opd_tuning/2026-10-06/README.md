# Spatial OPD main-method tuning and baseline continuation

The user selected explicit spatial-policy DeCoTA as the main method on
2026-10-06. Its registered interface is Native WHEN, original Uniform4,
one frozen Grounding DINO with unchanged admission/Top1, 1792 query/LN
parameters and 32 antithetic Gaussian actions in spatial logit coordinates.
Detached IoU feedback supplies true Gaussian-likelihood gradients. This
registration is user choice, not a new measured dual-dataset qualification.

Only lr, sigma, teacher tau, configured final step count and LN writeback
vary in a finite development search. Each target dataset receives one unified
configuration. The protocol fixes 16 single-factor configurations on 16 original
exposed development parents and two orders, followed by at most 12 proposals
along the two most sensitive coordinates on the original 32 development
parents and two orders. Duplicate refinement configurations reuse their
completed trial. The 128-source confirmation cohort does not rank parameters.

Actual implementation qualification passed eight query episodes in independent
workers across VidSTG and HC2. Four default-config episodes reproduced the
original output and updated state bitwise. Independent Gaussian/IoU/Adam
arithmetic and state-chain checks passed, with informative likelihood updates.
Finite tuning is actually running; the published startup config is provisional.
No selected configuration or tuned efficacy result is claimed yet.

Both dataset selections must finish and be sealed before the saved original
paper baselines resume. TENT VidSTG retains 11791 existing arrivals and its
complete Adam/LN state, with next cursor order2 arrival1488 (zero-based).
Original parameters, checkpoints, rosters and scientific code remain fixed.
The process-local verifier records only the expressly authorized registry hash
change. TENT, SAR and full EATA continue; missing exact HC source-training media
remains an explicit Fisher dependency. Other method experiments stay paused.

Old unfinished OPD corruption mechanism jobs were stopped and preserved by
user authorization. No full 2880-arrival completion is claimed. Cleanup removed
44751 obsolete independent binary/cache payloads, about215.34 logical GiB;
scientific reports/configurations, new OPD evidence, all baseline predictions
and full optimizer states, source checkpoints and required shared inputs were
preserved. Anonymous cleanup counts and manifest hash are exported; the raw
private path manifest is not published.

Actual search trials and failure evidence will be exported when sealed. GT
serves offline exposed-development selection only, after prediction seals;
it never enters the online objective, admission, output or reset. Selected
scores do not constitute an untouched test result. Original OPD confirmation
limitations remain recorded: Vid improved, HC total gain was inconclusive and
HC mean current-query correction was harmful under the original parameters.
