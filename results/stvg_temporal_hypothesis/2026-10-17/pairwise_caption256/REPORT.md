# Joint pairwise Query-Caption consistency, fixed development256

Same original exposed HC128/Vid128, exact frozen I1/I2 and512 old R1 captions; no new video/caption generation. Same original512 integrated temporal head, merged prior/target, tau1/full-grid trust1, LBFGS50/Armijo and exact marginal MAP. Only text judgment changes. Both Teachers judge identical captions, both display orders, remapped and averaged; A/B directional, C indistinguishable and D both insufficient nondirectional. No GT selection, no confidence/disagreement filtering; old truncated descriptions retain original no-update admission. No Actionness, new Spatial fit or production promotion.

|Domain|Temporal arm|vIoU %|tIoU %|dense sIoU %|
|---|---|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|
|hc2|T2|26.331915|49.232269|53.809342|
|hc2|MARGINAL_NO_UPDATE|24.945094|46.626294|53.809342|
|hc2|R1_INDEPENDENT_B3|24.832876|46.553780|53.809342|
|hc2|R1_PAIRWISE|24.897626|46.618794|53.809342|
|hc2|QWEN_PAIRWISE|24.859644|46.514719|53.809342|
|hc2|R1_TARGET_HARD|24.724924|46.419359|53.809342|
|hc2|QWEN_TARGET_HARD|24.929404|46.625290|53.809342|
|vidstg|Frozen|9.843861|32.374371|30.729497|
|vidstg|T2|10.286840|33.278160|30.729497|
|vidstg|MARGINAL_NO_UPDATE|10.299485|34.126993|30.729497|
|vidstg|R1_INDEPENDENT_B3|10.290674|34.144786|30.729497|
|vidstg|R1_PAIRWISE|10.143823|33.783530|30.729497|
|vidstg|QWEN_PAIRWISE|10.228260|33.971147|30.729497|
|vidstg|R1_TARGET_HARD|10.187121|33.903882|30.729497|
|vidstg|QWEN_TARGET_HARD|10.239252|34.021462|30.729497|

Main contrast is each new fitted arm minus MARGINAL_NO_UPDATE, then versus old R1_INDEPENDENT_B3. Frozen also changes the decoder; its difference is not pure adaptation. Full signed per-source paired t95/tails/unknowns retained, no significance-only decision gate. Target-hard reads use the tilted candidate posterior and are not Student upper bounds. Both Teacher arms were fitted regardless of ranking; no offline GT sorting was used online. The comparison tests text judge and task, not new visual understanding. Model posterior is uncalibrated.

## Feedback and actual update diagnostics

hc2 / R1: 123/128 informative, 6150 accepted steps; interval-order proxy 62 agree / 51 disagree / 15 tied or unknown. Both original and matched-support old ordering are in SIGNAL_DIAGNOSTICS.json. Order argmax agreement 117/127; average mapped margin disagreement 0.1006471700840318; tie/insuff posterior 0.10560101749170131/0.7315079626649266.
hc2 / QWEN: 123/128 informative, 6051 accepted steps; interval-order proxy 61 agree / 52 disagree / 15 tied or unknown. Both original and matched-support old ordering are in SIGNAL_DIAGNOSTICS.json. Order argmax agreement 126/127; average mapped margin disagreement 0.011093722100388653; tie/insuff posterior 0.017135800751715377/0.9632353247619989.
vidstg / R1: 120/128 informative, 6000 accepted steps; interval-order proxy 46 agree / 50 disagree / 32 tied or unknown. Both original and matched-support old ordering are in SIGNAL_DIAGNOSTICS.json. Order argmax agreement 94/121; average mapped margin disagreement 0.21787589786897513; tie/insuff posterior 0.14786565811411778/0.48312229829386244.
vidstg / QWEN: 119/128 informative, 5912 accepted steps; interval-order proxy 46 agree / 49 disagree / 33 tied or unknown. Both original and matched-support old ordering are in SIGNAL_DIAGNOSTICS.json. Order argmax agreement 112/121; average mapped margin disagreement 0.0776612614305829; tie/insuff posterior 0.048170740393160945/0.8276410016023423.

This single revision does not establish semantic truth, independent confirmation, or cross-backbone transfer. Existing512-window semantic panel remains44 model-assisted/468pending. No further prompt, hypothesis, temperature, optimizer or reward-weight search is triggered. Raw query/caption/Teacher responses, media, GT coordinates and model tensors are excluded from the public export.

## Independent actual-output transfer audit

{'dataset': 'hc2', 'teacher': 'R1', 'total': 128, 'informative': 123, 'actual_optimized': 123, 'accepted_steps': 6150, 'actual_FP32_moves_in_Teacher_direction': 123, 'median_target_logodds_fraction': 0.8428578860964911, 'mean_initial_candidate_KL': 0.011980327366108746, 'mean_final_candidate_KL': 0.0006973881242730566, 'MAP_changed': 21, 'display_order_direction_flips': 67, 'matched_GT_interval_order_support': {'count': 110, 'new_agree': 59, 'new_disagree': 51, 'old_agree': 53, 'old_disagree': 57}}
{'dataset': 'hc2', 'teacher': 'QWEN', 'total': 128, 'informative': 123, 'actual_optimized': 123, 'accepted_steps': 6051, 'actual_FP32_moves_in_Teacher_direction': 123, 'median_target_logodds_fraction': 0.8457658097937754, 'mean_initial_candidate_KL': 0.001085046076646218, 'mean_final_candidate_KL': 3.4599300701840515e-05, 'MAP_changed': 6, 'display_order_direction_flips': 61, 'matched_GT_interval_order_support': {'count': 110, 'new_agree': 59, 'new_disagree': 51, 'old_agree': 53, 'old_disagree': 57}}
{'dataset': 'vidstg', 'teacher': 'R1', 'total': 128, 'informative': 120, 'actual_optimized': 120, 'accepted_steps': 6000, 'actual_FP32_moves_in_Teacher_direction': 120, 'median_target_logodds_fraction': 0.8496378879205522, 'mean_initial_candidate_KL': 0.03707671944829463, 'mean_final_candidate_KL': 0.0015456531894180504, 'MAP_changed': 39, 'display_order_direction_flips': 56, 'matched_GT_interval_order_support': {'count': 95, 'new_agree': 46, 'new_disagree': 49, 'old_agree': 41, 'old_disagree': 54}}
{'dataset': 'vidstg', 'teacher': 'QWEN', 'total': 128, 'informative': 119, 'actual_optimized': 119, 'accepted_steps': 5912, 'actual_FP32_moves_in_Teacher_direction': 119, 'median_target_logodds_fraction': 0.8522909307090172, 'mean_initial_candidate_KL': 0.012402904498042319, 'mean_final_candidate_KL': 0.00044510917079490714, 'MAP_changed': 17, 'display_order_direction_flips': 64, 'matched_GT_interval_order_support': {'count': 94, 'new_agree': 46, 'new_disagree': 48, 'old_agree': 41, 'old_disagree': 53}}

RAW_REWARD_READOUT.json separately reports direct raw preference selection versus Source marginal, derived solely from already scored fixed-hypothesis metrics. It is distinct from prior-tilted TARGET_HARD and actual Student adaptation; none is an upper bound. Readout recovery preserved49 predictions and all992 Teacher prefills/all fits; only207 remaining readouts resumed. Two previously known retired Source prefixes were recomputed from exact original pixels and checked against saved records/logits/hidden/boxes; no Teacher or optimizer was retried, original deadline/failure/cost retained.
