# Spatial OPD main-method tuning and baseline continuation

## Actual completed finite tuning and baseline resumption

Both dataset parameter selections are now sealed. Vid: lr .03, sigma .1,
tau .25, steps10, LN writeback1/8. HC2: lr .03, sigma .1, tau .25, steps20,
LN writeback1/16. M32 and the method structure remain fixed. Each dataset
has one unified config in methods/decota_spatial_opd_v1/configs.json.

42 distinct configurations completed1664 scored arrivals; one HC2 refinement
configuration (lr .1, sigma .25, steps20) was numerically invalid after57 saved
arrivals, retained and unscored. The two searches each used16 screens followed
by12 finite refinement proposals; duplicates reused existing runs (Vid6 unique
refinements, HC5 including the invalid config). Eight qualification episodes
are separate from the1664 scored arrivals. The original128 confirmation cohort
was not used for selection, and all development sources were historically
exposed.

Independent root readback passed252 aggregates/10000 source-bootstrap checks,
source weighting, total/current/inherited decomposition, cost sums, selection
ranking, seal-before-GT times and1664 payload byte hashes. Both plots were
actually viewed. A Markdown table row ordering issue and repeat-render immutable
write failure were preserved and repaired as report-only revisions. Scientific
predictions, parameters, rosters and all original GPU pins were unchanged.

Selected development delta-vIoU is Vid+4.026pp [1.793,6.630], HC+2.734pp
[.414,5.837]. Both panels have0 >20pp harm sources/cells. Current-query gain
is Vid+2.774pp [0.969,4.825], HC+1.987pp [-.272,4.821]; the HC current interval
still crosses zero. These selected-development intervals do not qualify
independent efficacy. Complete outcomes and the numerical failure are exported.

The serial continuation actually resumed saved Vid TENT after both selections.
11791 old arrivals and full Adam/LN state are retained; later fresh arrivals
already exist. Remaining original baselines continue. Other method experiments
stay paused, and missing HC source-training media remains an EATA dependency.


## Initial startup record

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
