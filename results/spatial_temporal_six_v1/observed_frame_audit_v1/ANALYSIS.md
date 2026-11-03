# Read-only full-cohort observed-frame output and metric audit

Current implementation does not hard-clamp observed frames to original expert boxes. At an observed frame, E remains candidate0 and the original expert box is candidate1 (candidate2 duplicates it and is disabled). Both E and expert remain legal inside the event; appearance plus adjacent-box-IoU DP selects the final output. Outside the predicted event, only E is legal. The current finish returns DP output without an expert-box overwrite. Pure interpolation, by contrast, explicitly overwrites every anchor with its original expert box.

All HC1902 and Vid10303 saved predictions were read without model/Source/expert/optimizer/media calls or method changes. Every physical frame/position/interval and GT support matched. Pure outputs all8expert boxes exactly in every query. Every Current observed output is exactly either E or the expert box, with outside-event output always E. No wrong-index/GT-support mismatch or third unintended output was found. First3 differing inside-event cases/domain were replayed from saved geometry; full DP output equals the historical Current arrays bitwise, and both E/expert were legal when E was selected. This is solver behavior, not a later accidental overwrite.

The previously reported observed_actual_sIoU uses ALL GT-supported native sampled expert frames, independent of the predicted interval. It therefore includes outside-event frames that are not part of the output grounding tube. The observed-frame metric is numerically reproduced, but must not be described as event-only observed IoU. Both the missing hard-anchor constraint and metric scope explain the reported loss; the loss remains negative even when restricted to the common predicted event.

|Domain|GT-supported observed frames inside event|Current selects E instead of expert|Fraction|GT-supported observed frames outside event|
|---|---:|---:|---:|---:|
|hc2|5064|2321|45.833333%|955|
|vidstg|22640|7947|35.101590%|9671|

|Domain / support|Scorable queries|Annotated frames|Pure observed IoU %|Current observed IoU %|Current minus pure pp [95% CI]|
|---|---:|---:|---:|---:|---|
|hc2 / all_observed|1880|6019|59.790469|56.208695|-3.581774 [-4.253204, -2.914003]|
|hc2 / inside_common_event|1632|5064|59.721199|57.984671|-1.736529 [-2.026883, -1.441380]|
|hc2 / outside_common_event|698|955|60.612203|51.334752|-9.277452 [-11.683524, -6.812958]|
|vidstg / all_observed|8507|32311|45.940580|41.886492|-4.054089 [-4.477224, -3.660726]|
|vidstg / inside_common_event|7280|22640|47.007336|45.954453|-1.052883 [-1.205111, -0.907308]|
|vidstg / outside_common_event|4597|9671|45.970436|35.525922|-10.444515 [-11.384533, -9.515499]|

## Arithmetic decomposition under the original metric denominator

Inside/outside conditional means above have different query/support denominators and cannot simply be added. The following contributions retain each query original all-observed-frame denominator and reproduce its original delta exactly, then average equally across scorable queries. These are arithmetic contributions, not causal attribution.

|Domain|Inside contribution pp|Outside contribution pp|Original all-observed delta pp|
|---|---:|---:|---:|
|hc2|-1.372498|-2.209276|-3.581774|
|vidstg|-0.737355|-3.316734|-4.054089|

An independent second process read original production prediction NPZ fields (native_boxes/SharedInit) instead of the geometry PT files and recomputed IoUs in normalized coordinates instead of using the first audit pixel-IoU helper. All count categories/supports and per-query IoUs/contributions agree; maximum error <2.3e-13 percentage points. Every old method/source/score file hash bound by the audit plan is unchanged. Full source code/plan/scalar receipts and all anonymous query outcomes are retained; no private IDs, boxes, media, captions, features or masks are exported.

The same-expert-input comparison remains a comparison of the ACTUAL complete spatial strategies. It does not establish an experiment where both strategies hold the expert-anchor OUTPUT fixed and differ only at unobserved frames. The published vIoU numbers were not altered or rescored here. If exact retention of expert observations is the intended method definition, the implementation fails that condition; changing it would require explicit versioning/authorization and recomputation, which this audit does not perform. No clamp, postprocessing fix, counterfactual method variant, manuscript edit or main-table replacement was made.

All labels are posthoc analysis only; missing support is unscorable, not zero. Queryequal10000paired-parent CIs are pointwise, not familywise. This new audit is initially LOCAL; prior CPU experimental results remain published at94175804. Include this diagnostic with the next current-scope publication rather than claiming remote synchronization already occurred.
