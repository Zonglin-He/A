# Measured interpretation

The candidate teacher was actually constructed and evaluated on all32 current exposed NH16/NV16 queries. The frozen networks were not updated. Source predictions/logits were reproduced exactly, and fresh input_proj appearance maps captured once/query. Per-anchor ridge models used E-selected DINO positives, protected all retained candidate regions, and fitted only disjoint background ROIs.

The prelocked gate FAILED. Fixed dense distillation and two-round on-policy adaptation were NOT RUN, rather than run with zero effect. Their code and CPU KL/no-feedback properties exist, but there is no measured OPD result here. No optimizer/regularization/temperature/negative-box search followed the failed gate.

Unobserved actual decoder nodes, candidate weighted true-IoU differences, pp (query=parent bootstrap10000 seed20261018):

| Difference | HC mean [95% CI] | Vid mean [95% CI] |
|---|---:|---:|
| Ridge minus Gaussian prior | +1.144708 [+.485968,+1.922572] | -.429064 [-1.829208,+.705914] |
| Ridge minus all-E cosine control | -3.848287 [-6.032444,-1.806050] | -3.377808 [-5.944873,-1.279954] |
| Ridge minus matched-usable-anchor cosine | -1.267825 [-2.723487,-.030563] | -1.657849 [-3.280218,-.210921] |
| Ridge minus shuffled target-frame features | +.354981 [+.089674,+.727681] | +.228361 [+.031464,+.478977] |

Positive real-minus-shuffle evidence is retained: this interface uses actual target-frame visual content. It is not justified to call its feedback wholly uninformative. Nevertheless it loses to cosine even with exactly matched usable anchors, so lost anchor coverage alone cannot account for the gap. These are weighted-quality differences at the specified probability/temperature interface, not a scale-invariant proof about raw score ordering. ESS and the explicit proximity prior are reported. New center-only E-neighborhood candidates/prior/cohort differ from historical correspondence tests: current cosine numbers must NOT be used to revise old TTA conclusions.

Complete tube vIoU% E -> teacher candidate argmax: HC35.305477 ->33.049104 (-2.256373pp, CI[-4.559851,-.298643]); Vid15.755931 ->14.536266 (-1.219665pp, CI[-3.433580,0]). Unobserved actual sIoU changes -4.499341pp[-9.520030,-.480048] and -5.068755[-9.985422,-1.154352]. Time/tIoU, all widths/heights and planned-anchor actual outputs are exact E. Complete losses below -5pp: HC3/16 and Vid1/16. HC complete positive/negative/unchanged1/8/7, Vid0/3/13. Unknown GT remains null; all queries remain in denominator.

HC positive weighted improvement over perturbed Gaussian candidates does not imply improvement over the unperturbed E center. The ridge expected candidate IoU is12.377326pp below E on HC actual unobserved nodes, and9.871396pp below on Vid. Even prior/cosine expectations are below E in this local perturbed pool. These expectations are not deployment tube metrics. Candidate-argmax readout likewise did not beat E. Thus we cannot assume that a distribution update would create useful localization just because a candidate reweighting is positive relative to a noisy baseline.

Coverage limitation: E selected53 HC/35 Vid anchors, but only19/15 allowed the locked SAME-SIZE background construction;34/20 had no box disjoint from all retained candidates. Usable classifier queries9/16 and6/16. All34 models satisfy their normal equations and show positive in-sample foreground/background score contrast (means1.8654/1.8675), which does not establish unseen-frame localization. No-usable-model queries7/10 include original no-E1/5; their teacher/readout behavior is kept as a no-op. No hidden fallback or GT-based anchor selection.

Before any GT, the controller was briefly held while its Source-capture child finished all32. Pure input inspection revealed the coverage loss; the cosine PRIMARY control was corrected to retain ALL E-selected anchors, and its matched-usable subset kept as a secondary control. Original code/protocol/runtime are preserved in recovery_control_support001. No new-teacher scores/readout, candidate, hyperparameter, gate or original deadline changed; no Source recapture was needed. The all-E vs matched control distinguishes coverage loss from the observed remaining feedback gap.

Small/large/slow/fast strata and counts are in PHASE1_RESULTS_PUBLIC.json and REPORT.md; they only explain offline outcomes and never select an online method. This cohort has no HC Q1(slowest common-bin) source, so no HC Q1 result is invented. There are only2 Q1-size sources/domain. Preserve severe examples NH007(-13.662981pp vIoU), NH015(-11.458307), NH010(-6.664675) and NV005(-16.360342), and HC Q3-motion mean positive+.943181pp, rather than claiming every case failed. NV001 has unchanged complete vIoU but unobserved sIoU-28.959525pp, illustrating distinct metric supports; it is not counted as a complete-v negative case.

Actual added work:32 frozen Source captures,0 new DINO calls,34 closed-form per-anchor ridge fits,0 Student/OPD updates. GPU lease133.347818s including model loading/capture, CPU teacher construction/scoring3.758390s (GT scoring/audits separate), no first-online E generation cost included. No new video model/offline training/download/backbone/temporal adaptation.

Independent checks:34 ridge equations and class balances,34 background exclusion/actual ROI pools,3468 scalar frame-quality sums,32 primary rank aggregates,160 exact saved-E metrics and320 independent official metric scalars. Root repooled133484 real/shuffled score scalars and133484 per-anchor probability scalars,62628 mixture scalars; maximum score error5.55e-16. All-E cosine support checked for32 queries. Actual statistical chart inspected. No new identity labels or human-review claims.

Decision: stop this specific frozen-feature/ROI/ridge/background/probability interface under its locked gate; do not run its OPD optimizer or sweep parameters. Retain MBR+E and OPD as a research route. These findings do not disprove all discriminative teachers, trained tracking features or dense OPD, and do not measure an OPD failure.
