# Reviewer revision: outline, claim-evidence map and open items

The editable manuscript is a separate local working copy. The supplied source
is preserved byte-for-byte. This document does not claim a submission-ready PDF.

## Compact outline

1. Define the spatial adaptation task inside a fixed native temporal interval.
2. Describe fixed specialist boxes, geometric action rewards and the compact decoder.
3. Explain the query/LN lifecycle without assuming retention is always beneficial.
4. Report full-query system results with DINO Refine and original TTA configurations.
5. Use matched P2 Direct contrasts to evaluate independent OPD value.
6. Report conditional expert/actionability and actual decoder-frame propagation.
7. Separate offline spatial/temporal headroom from deployable temporal refinement.

## Paragraph roles in the revision

- Abstract: task and deployment challenge → compact spatial interface and lifecycle →
  measured full-query gains → conditional mechanism tradeoff.
- Method admission: exact score/margin/duplicate rule and no-admission identity.
- Method OPD: actual geometric feedback → weighted Gaussian fitting → fixed-weight
  regression equivalence → finite-sample zero-reward boundary → resampling.
- Main results: exact DINO baseline, with the original official query denominator.
- Mechanism results: matched Direct comparison followed by shuffled/rollout controls.
- Correction analysis: real decoder versus interpolation and task-gated temporal support.
- Oracle analysis: two offline interventions and the remaining temporal headroom.

## Claim-evidence map

| Claim | Actual evidence | Status / allowed scope |
|---|---|---|
| Complete system improves cross-domain grounding | Closed P1 TABLE1_SUMMARY; +2.566 pp HC and +2.968 pp Vid query-macro; excluded-development parent analysis | Supported on the historical evaluated population |
| OPD stably exceeds Direct regression on average | Closed P2 matched 128-parent controls, Full−Direct +0.576 / −0.213 pp cross-domain, both CIs cross zero | Not established; restore Direct and avoid universal superiority |
| Local sampling can be more conservative toward geometrically poor references | New low-Q/all-zero HC strata, +11.520 / +9.751 pp Full-current−Direct-current; other settings have wide CIs | Conditional descriptive support; not an identity detector or common-initial-state causal test |
| IoU zeros cannot impede correction | High-Q/all-zero HC strata strongly favor Direct, on 5 and 11 parents | Unsupported; report the failure mode |
| A reward change is now warranted under the fixed resource trigger | Only 1.274–6.593% high-Q first-round all-zero observations; all four trigger decisions false | No pilot launched; primary method unchanged |
| Sparse updates improve real unobserved decoder outputs | New Vid cross After−Before +5.419 pp, positive CI; HC −0.581 pp, CI crosses zero | Supported conditionally on Vid cross, not across every setting |
| Complete inherited model exceeds parameter-free DINO on all unobserved groups | All-eligible and common-parent comparisons are separate; HC cross common-parent CI crosses zero | Conditional; retain both populations and separate current effects |
| Native WHEN covers all GT-event support | New coverage approximately 56% cross-domain | Unsupported; outside-interval geometry cannot add coverage |
| Space is the universally dominant cross-domain error | Temporal oracle headroom ≥ spatial in both target directions | Unsupported; motivate a spatial intervention, not global error dominance |
| Slow state has stable independent value over Reset | Existing direct small-cohort comparisons; queued 905-parent/5430-fit study and missing LN-only across three real permutations | New confirmation pending; Before−Frozen is not Full−Reset |
| Attenuation is preferable to complete carryover | Existing three real permutation controls; all saved positive/negative findings retained | Assess direct per-order Full−Full-Carry, not bootstrap as order repetition |
| Decoder fit time is end-to-end deployment latency | Old fit-stage timings exclude shared capture/expert; complete audited cost and uncached Full/Frozen studies pending | No new low-latency claim |
| Current TTA main-table rows are the newly calibrated results | Original configurations remain in the draft; selected TENT/SAR full formal jobs are active | Calibrated replacement pending full deployment seal and root closure |

## Five-dimension self-review

1. **Contribution:** Is the contribution stated as a concrete spatial update interface?
   Yes. The draft no longer assumes semantic teacher judgments per action or unique
   superiority from the distributional label alone. Title strength still depends on
   the pending direct state-transfer evidence.
2. **Writing clarity:** Can a reader distinguish Native, Before, After and DINO?
   The revision defines inherited Before separately from cached Native WHEN and
   describes exact DINO whole-trajectory refinement.
3. **Experimental strength:** Does the direct comparator have the same denominator?
   Yes. DINO belongs in the full official-query table; Direct belongs in the matched
   128-parent P2 table. Their rows are not merged across populations.
4. **Evaluation completeness:** Are harms, absent observations and frame groups retained?
   All continuous anonymous quality/frame/cell records, all strata, negative tails,
   common-parent intersections, missing-annotation counts and offline oracles remain.
   Historical exposure and unadjusted descriptive intervals are explicit.
5. **Method soundness:** Are zero reward, no admission and temporal support conflated?
   No. No admission gives After=Before; a zero-reward observation is not a complete
   query no-op; fixed WHEN cannot recover GT-event frames outside its interval.

## Outstanding work

- Existing calibrated baseline global evaluation and actual root/view/public closing.
- Existing expanded non-development 905-parent Full/alpha0/alpha1 experiment.
- Existing complete uncached Full/Frozen deployment timing.
- Newly queued 768 LN-only fits on the same three actual parent permutations,
  24 complete GPU qualification fits, global seal, independent root and closing.
- Merge any actual new conclusions after those stages; do not prefill pending scores.
- Supply the original CVPR style, figure and bibliography project for final layout.
  The built-in compiler currently fails before source compilation because its
  Tectonic bundle index is unavailable from the configured network. This is not
  evidence of a TeX syntax failure. The editable source remains preserved and open.
- Final page limit and official submission layout are not verified by this draft.

No expert/backbone, reward, optimization parameter, gate, memory, production method,
or paused EATA/legacy queue is changed by this revision.
