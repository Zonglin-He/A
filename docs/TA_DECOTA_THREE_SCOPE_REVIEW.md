# Native temporal scope, dual-offset P0 and pre-update LN compatibility

The three tests are separate. TTS changes acquisition only; boundary P0 changes temporal readout only; memory P0 predicts previously measured isolated utility with pre-update features. A conditional NO-GO is a decision for this locked mechanism, not a proof that all temporal adaptation or memory is impossible.

Actual fixed panel: each dataset 32 development + 16 source-disjoint confirmation, one query/source, two orders, clean + five original 5% corruptions. All have historical exposure. Confirmation is not fresh. Match the prior Top1 working point with a 100% scheduled-expert rate in both episodic and independent online streams; admission/eligibility may still produce no update. Official same-domain checkpoints, one frozen DINO, original admission + Top1 singleton energy, Adam .03, joint1792, ten steps/own-loss first minimum, Native time, reset query/Adam, selected LN delta/16. Deployed CURRENT_METHOD unchanged.

## TTS acquisition: current and actual online outputs

Uniform4 reference is the previous verified actual stream. TTS uses independent evolving LN; it never reuses Uniform post-update state. All2304 TTS logical outputs were sealed before scoring. Frozen uses identical Native. Unit pp, source-macro and 10000 paired source-bootstrap.

| Dataset/panel | Stream | Uniform−Frozen Δv | TTS−Frozen Δv | TTS−Uniform Δv | TTS−Uniform Δs | TTS >5/>20 harm vs Frozen |
|---|---|---:|---:|---:|---:|---:|
| vidstg/search | episodic | -0.1285 | -0.1115 [-3.1159, +2.5619] | +0.0171 [-1.5878, +1.7847] | -1.3506 [-4.7702, +1.7298] | 46/24 |
| vidstg/search | online100 | -0.0766 | -0.4561 [-3.4053, +2.2304] | -0.3795 [-1.9908, +1.0063] | -2.3142 [-5.8269, +0.5818] | 56/22 |
| vidstg/confirm | episodic | +4.9658 | +5.1743 [+1.9414, +9.6371] | +0.2085 [-0.5017, +1.0895] | -0.1572 [-1.2325, +0.9167] | 0/0 |
| vidstg/confirm | online100 | +4.8928 | +5.4274 [+2.0770, +10.0427] | +0.5345 [-0.2298, +1.6718] | +0.1992 [-0.9205, +1.4509] | 0/0 |
| hc2/search | episodic | +3.7737 | +3.8047 [+0.4770, +7.1872] | +0.0310 [-1.8586, +2.3029] | +0.3994 [-2.9743, +3.9294] | 42/10 |
| hc2/search | online100 | +3.4803 | +4.8347 [+1.7171, +8.5560] | +1.3544 [-1.0227, +4.5046] | +2.0301 [-1.6865, +6.1635] | 32/4 |
| hc2/confirm | episodic | +3.9230 | +4.5146 [+1.7069, +7.8626] | +0.5916 [-0.6974, +1.9657] | +1.6728 [-0.6519, +4.5515] | 0/0 |
| hc2/confirm | online100 | +4.1580 | +4.9406 [+2.2443, +8.1451] | +0.7826 [-0.3499, +2.0610] | +1.9067 [-0.2529, +4.6367] | 0/0 |

TTS scope qualified on both confirmation corruption panels: **False**. No automatic promotion. Per-order/condition/clean, Before and matched net-memory results are all retained in scope/SUMMARY.json. Current gain does not establish memory gain.

| Dataset/confirm | TTS Before−Frozen | TTS current After−Before | TTS online−matched episodic | Uniform online−matched episodic |
|---|---:|---:|---:|---:|
| vidstg | +0.5711 [+0.2665, +0.9412] | +4.8563 [+1.6605, +9.2497] | +0.2530 [-0.0112, +0.5630] | -0.0729 [-0.2337, +0.0923] |
| hc2 | +1.1805 [+0.6990, +1.7436] | +3.7601 [+1.2507, +6.7854] | +0.4260 [+0.0109, +0.9621] | +0.2350 [-0.0002, +0.5151] |

The safety comparison counts logical corruption arrivals, not independent sources; each source has five conditions and two orders. Confirmation has160 such arrivals per dataset, development320.

| Dataset/panel, online | Uniform >5/>20 pp harm vs Frozen | TTS >5/>20 pp harm vs Frozen |
|---|---:|---:|
| vidstg/search | 47/9 | 56/22 |
| vidstg/confirm | 0/0 | 0/0 |
| hc2/search | 30/8 | 32/4 |
| hc2/confirm | 10/0 | 0/0 |

TTS retains positive confirmation gains over Frozen, but its incremental confirmation gains over Uniform include zero in both datasets. The hoped-for common reduction of development severe tails is not established: Vid worsens while HC improves in the logical >20pp count. The matched online-minus-episodic effect is small, uncertain in Vid and positive in this HC panel; the Before effect alone cannot substitute for that net comparison. Keep Uniform4 as the working baseline, without claiming that every target-aware acquisition is ineffective.


## Evidence quality

Rates divide by requested slots, including unavailable/empty evidence. IoU mass is admitted Top1 GT-IoU summed over exactly GT-scored observations divided by slots; it combines coverage and quality and is not conditional identity accuracy. Conditional means, valid/scored/admitted counts, best/worst and every observation are in EVIDENCE_ROWS.json. No scored GT at a selected frame stays missing.

| Dataset/panel | Δvalid rate | Δadmission rate | Δobserved-event-frame rate | Δadmitted/event GT-IoU mass |
|---|---:|---:|---:|---:|
| vidstg/search | +0.0000 [+0.0000, +0.0000] | +2.9688 [-2.1875, +8.1250] | +1.0938 [-2.9688, +5.7812] | +1.0263 [-2.0054, +4.7923] |
| vidstg/confirm | +0.0000 [+0.0000, +0.0000] | -0.6250 [-7.5000, +5.6250] | +0.0000 [+0.0000, +0.0000] | -0.2309 [-3.3849, +3.7111] |
| hc2/search | +0.0000 [+0.0000, +0.0000] | +2.9687 [-3.2812, +9.2188] | +3.1250 [-0.3125, +6.7188] | +3.5145 [-1.4413, +8.3474] |
| hc2/confirm | +0.0000 [+0.0000, +0.0000] | +1.5625 [-2.1875, +6.2500] | +3.7500 [-0.9375, +8.4375] | -0.5144 [-4.4661, +3.0677] |

The event-frame rate counts actual observations in the GT event divided by requested slots; unavailable observations do not count as event hits. Conditional observation quality is descriptive: a different acquisition rule also changes which frames have scored GT. It is not an identity-accuracy label.

| Dataset/panel | Acquisition | valid/admitted/scored-valid | Mean scored Top1 GT-IoU | Mean admitted/scored GT-IoU |
|---|---|---:|---:|---:|
| vidstg/search | uniform | 520/418/293 | 41.12% | 39.76% |
| vidstg/search | tts | 520/437/300 | 41.64% | 41.26% |
| vidstg/confirm | uniform | 220/209/154 | 85.70% | 85.96% |
| vidstg/confirm | tts | 220/207/154 | 82.53% | 86.65% |
| hc2/search | uniform | 640/557/411 | 55.04% | 59.93% |
| hc2/search | tts | 640/576/426 | 58.72% | 61.92% |
| hc2/confirm | uniform | 320/267/209 | 53.30% | 60.43% |
| hc2/confirm | tts | 320/272/219 | 51.89% | 54.31% |

Confirmation evidence does not show a shared improvement: admitted/event GT-IoU mass has a negative point difference and a CI spanning zero in both datasets. HC obtains more observed event frames at the point estimate while its conditional admitted observation IoU decreases; event coverage and referent quality are separate. Conditional means above describe changed observation populations, not a causal same-frame identity test.

## Dual-offset boundary P0

Physical alignment interpolates discrete probability values on existing sampled frames, floors1e−12 and renormalizes. It is not a calibrated density or independent expert likelihood. Geometric endpoint means and strict i<j MAP; only evaluation interval changes, fixed Uniform Top1 spatial output. All1152 old online100 inputs covered and new readouts sealed before GT.

| Dataset/panel | Consensus−Native Δt | Δv | >5/>20 v harm |
|---|---:|---:|---:|
| vidstg/search | -0.7474 [-4.0843, +3.0956] | -0.2071 [-2.3624, +1.9500] | 20/6 |
| vidstg/confirm | -2.3979 [-4.4171, -0.4738] | -1.5973 [-3.0311, -0.2207] | 30/0 |
| hc2/search | +0.3221 [-0.8455, +1.6125] | +0.0574 [-0.5380, +0.6996] | 9/0 |
| hc2/confirm | -0.5732 [-2.8846, +1.3703] | -0.4928 [-1.6955, +0.5355] | 10/0 |

Temporal514 qualification: **False**. Status: skipped_P0_did_not_qualify. No gradient result is invented for a skipped stage. JS/error association is diagnostic only (boundary/JS_DIAGNOSTIC.json); all four source-bootstrap intervals for its association with Native error include zero.

## Pre-update memory P0

All939 exact isolated donor/recipient pairs are retained, including zero recipient-correction cases. Utility is actual selected donor LN write/16, source-initialized recipient Before−Frozen before a second nonzero write. It is not a full delta intervention or accumulated long-stream effect. Features: frozen raw-query RoBERTa masked mean768, Native final spatial latent mean256, box/time/TTS summaries. No recipient correction, future gradient or GT quality enters keys.

The ridge predictor uses existing GT-derived utility labels in development, alpha1 fixed. Both held-out donor and recipient nodes are excluded across all roles/conditions; confirmation fits development only. This is explicitly a supervised compatibility diagnostic, not an unlabeled deployed quality head. Similarity controls use no labels. Bootstrap uses common source-node weights in both roles and recipient-source-equal base weights; exact-zero utility is retained except binary AUC.

| Dataset/panel | Pre-update signal | corr(Uhat,U) | help/harm AUC | pairs/nodes |
|---|---|---:|---:|---:|
| vidstg/search | query_cos | -0.1920 [-0.6094, +0.2833] | +0.3415 [+0.0536, +0.7547] | 193/19 |
| vidstg/search | spatial_cos | +0.1624 [-0.2471, +0.5738] | +0.5990 [+0.2630, +0.9318] | 193/19 |
| vidstg/search | ridge | +0.1560 [-0.4937, +0.5223] | +0.4802 [+0.1097, +0.7977] | 193/19 |
| vidstg/search | constant | +0.2516 [-0.3389, +0.6837] | +0.8047 [+0.2123, +0.9877] | 193/19 |
| vidstg/confirm | query_cos | -0.2026 [-0.6745, +0.3779] | +0.4417 [+0.0000, +1.0000] | 200/14 |
| vidstg/confirm | spatial_cos | +0.2685 [-0.4376, +0.7612] | +0.5738 [+0.3542, +0.9208] | 200/14 |
| vidstg/confirm | ridge | +0.0638 [-0.3785, +0.5652] | +0.4711 [+0.0000, +0.8133] | 200/14 |
| vidstg/confirm | constant | not estimable | +0.5000 [+0.5000, +0.5000] | 200/14 |
| hc2/search | query_cos | -0.2566 [-0.6242, +0.3486] | +0.4041 [+0.1425, +0.7403] | 182/20 |
| hc2/search | spatial_cos | +0.2293 [-0.6700, +0.7410] | +0.6353 [+0.2251, +0.9446] | 182/20 |
| hc2/search | ridge | -0.0569 [-0.6288, +0.3562] | +0.4092 [+0.1171, +0.8019] | 182/20 |
| hc2/search | constant | -0.0298 [-0.5562, +0.5095] | +0.2899 [+0.0646, +0.7005] | 182/20 |
| hc2/confirm | query_cos | -0.1716 [-0.6922, +0.3809] | +0.5228 [+0.1940, +0.7251] | 206/15 |
| hc2/confirm | spatial_cos | +0.2117 [-0.2593, +0.5458] | +0.5988 [+0.2498, +0.9175] | 206/15 |
| hc2/confirm | ridge | +0.0374 [-0.4785, +0.6229] | +0.3929 [+0.1593, +0.7096] | 206/15 |
| hc2/confirm | constant | not estimable | +0.5000 [+0.5000, +0.5000] | 206/15 |

Memory qualification: **False**; passing shared signals []. Status: skipped_preupdate_P0_did_not_qualify. Three predeclared signals were tested; qualification only authorizes a separate matched memory trial, not production. Clean results and all pair predictions remain public.

Only14/15 independent confirmation source nodes contribute to the Vid/HC corruption pair panels, despite200/206 pair rows. Wide node-bootstrap intervals matter. Positive spatial-cosine point correlations do not establish a deployable predictor, and the NO-GO does not prove that transfer relations or conditional memory never exist. Development constant predictions differ across held-out folds because training means differ; confirmation constant predictions are truly constant and their correlation is undefined.

## Costs, cases and limits

576 unique native inputs reuse encoder H; no video backbone forward. Frozen TTS heads and text encoder plus cached native decoder are real compute. Newly chosen frames need the same DINO; overlapping observations are reused only with identical RGB/text/context receipts. Worker time is wall time, not isolated GPU kernel time. Exact calls/reuse/backward counts are in scope/COST.json.

Independent root audit checks physical selections, all939 ridge predictions with a separate Cholesky calculation, double-role source exclusion, all optimizer paths/state chains and official dense scores. Public audit recomputes anonymous aggregates and source/node-bootstrap CIs, with an independent weighted Mann-Whitney AUC calculation; it cannot re-run private GT or private feature extraction. No harmful or failed row is discarded. Historical good confirmation and harmful development cases are preserved. An initial additional root-readback assertion exposed mixed float32/float64 normalization in the auditor; its saved engineering repair uses float64 throughout and preserves strict tolerances, all predictions, gradients and GT metrics. See ENGINEERING_RECOVERY.json.

Results concern these native TTS classifiers, this physical-grid geometric barycenter, and these pre-update keys with fixed ridge; they do not establish failure of every routing, temporal self-supervision or conditional memory. No new full-query job, external expert, scorer sweep, latent readout repair or production promotion was executed.

## Actual integration

Frozen / S-only / Tscope+S are the actual independent per-query and online100 results above. Conditional T+S and retrieval arms are run only if their locked P0 qualifies; skipped statuses are explicit. Scope candidate performance is retained even if it fails qualification.

## Representative paired negative and recovery cases

| Dataset/panel | source/condition/order | TTS−Uniform v pp | TTS−Frozen v pp |
|---|---|---:|---:|
| vidstg_search/worst_vs_uniform | 28/motion_blur_5/order2 | -45.834 | -33.334 |
| vidstg_search/best_vs_uniform | 28/exposure_5/order1 | +44.538 | +13.393 |
| vidstg_confirm/worst_vs_uniform | 35/frame_freeze_5/order1 | -12.611 | +9.883 |
| vidstg_confirm/best_vs_uniform | 35/motion_blur_5/order1 | +24.866 | +25.778 |
| hc2_search/worst_vs_uniform | 25/occlusion_5/order2 | -21.010 | -19.765 |
| hc2_search/best_vs_uniform | 19/exposure_5/order2 | +64.107 | +52.601 |
| hc2_confirm/worst_vs_uniform | 43/exposure_5/order2 | -10.350 | +0.000 |
| hc2_confirm/best_vs_uniform | 46/occlusion_5/order1 | +20.805 | +19.186 |
