# Frozen R1 independent sparse geometry32

Fixed exposed core first16 per domain,32parents/128original T2 anchors, unchanged Source/T2 and paper Spatial-Lite parameters/reset/response/acceptance. Current WF actual complete results are exact reuse. DIRECT and HEADS share the same unmarked context4, native-aspect separate anchor image and full Query, without DINO/Student geometry. DINO is an independent same-position original-interface baseline, acquired after the R1-only references were sealed. All three complete Student/ROOT/Global barriers precede GT. No head/Prompt/threshold/optimizer search.

Single-sample LocalizationHeads adaptation follows official commit9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a, default top5/chord/entropy.001/bottom-row.05/Gaussian1/above-mean all-positive bbox. This is not the paper dataset-statistical fixed-head main result. Actual attention is eager full-key causal softmax at the last complete-input text position, sliced only afterward to a separate anchor-image rectangular grid; no image-key re-softmax or merged-video-token frame alias. Direct outputs use actual processor-resized pixel width/height, invalid/multiple/out-of-range outputs remain failed/abstained, no normalized1000 conversion or clamp. Model namespace compatibility failure before first prefill and CPU pidfd waiter error are preserved with real extra cost; no science result changed. One native DINO boundary1.0000000298023224 failed a redundant exact check; recovery uses the already-existing1e-7 float32 interface tolerance, keeps every coordinate unchanged, resumes eight verified completed queries and only fits the remaining sources.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|31.250000|25.000000|
|hc2|T2|27.487135|50.497117|53.666875|37.500000|18.750000|
|hc2|Current_WF|28.641217|52.179161|53.666875|37.500000|25.000000|
|hc2|DIRECT|27.972261|51.210459|53.666875|43.750000|18.750000|
|hc2|HEADS|28.769897|52.627071|53.666875|43.750000|25.000000|
|hc2|DINO|28.440075|51.967586|53.666875|37.500000|25.000000|
|vidstg|Frozen|12.185607|32.870500|35.923576|12.500000|0.000000|
|vidstg|T2|12.239461|32.870500|35.680949|12.500000|0.000000|
|vidstg|Current_WF|12.072499|32.573620|35.680949|12.500000|0.000000|
|vidstg|DIRECT|14.323998|35.542683|35.680949|18.750000|0.000000|
|vidstg|HEADS|14.012450|31.388836|35.680949|25.000000|0.000000|
|vidstg|DINO|15.368798|33.573892|35.680949|25.000000|6.250000|

| Domain | Contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |
|---|---|---:|---|---|
|hc2|DIRECT_minus_T2|0.485126|[-0.08725682547987157, 1.0575080949945022]|5/0/11/0|
|hc2|DIRECT_minus_Current_WF|-0.668957|[-1.5821707386401531, 0.244257278155384]|4/10/2/0|
|hc2|HEADS_minus_T2|1.282762|[0.7031428776862021, 1.8623814581271514]|14/1/1/0|
|hc2|HEADS_minus_Current_WF|0.128680|[-0.4744033534625566, 0.7317629592765104]|8/7/1/0|
|hc2|DINO_minus_T2|0.952940|[-0.06659226767108173, 1.9724728202988024]|11/3/2/0|
|hc2|DINO_minus_Current_WF|-0.201142|[-0.9638743111996029, 0.5615901338279239]|7/7/2/0|
|vidstg|DIRECT_minus_T2|2.084538|[-1.970201950876858, 6.139277032474867]|5/3/8/0|
|vidstg|DIRECT_minus_Current_WF|2.251500|[-1.0809368513023316, 5.583936444212988]|5/6/5/0|
|vidstg|HEADS_minus_T2|1.772989|[-2.031187782595185, 5.577166016389089]|7/4/5/0|
|vidstg|HEADS_minus_Current_WF|1.939951|[-1.0764063511903421, 4.956309096296893]|7/4/5/0|
|vidstg|DINO_minus_T2|3.129337|[-1.2624423833019653, 7.5211161056774225]|9/1/6/0|
|vidstg|DINO_minus_Current_WF|3.296299|[-0.3587689164264365, 6.9513671501145415]|8/4/4/0|

Reference geometry/coverage/localization failures and unannotated unknowns are separately disclosed in REFERENCE_QUALITY.json; all128 planned positions remain. Available-reference mean IoU is conditional, while localization success rates use all GT-known planned positions including invalid/abstained outputs as failures. Box IoU and heatmaps do not establish identity/role correctness. Full source-paired deltas, observed/unobserved sIoU, signs/unknowns and update counts are in RESULTS.json. This is an exposed small development screen, not independent confirmation. Existing semantic512 panel remains44model-assisted/468pending. The practical stage2 decision is a prelocked resource decision, not production promotion or a universal impossibility claim. No CURRENT/paper change.

Prelocked decision: {"status": "complete", "count": 32, "candidates": [{"arm": "DIRECT", "mean_increment_pp": 0.7912715331064717, "practical_gate_pass": false}, {"arm": "HEADS", "mean_increment_pp": 1.0343155877301262, "practical_gate_pass": false}], "second_stage_authorized": true, "not_method_promotion": true, "GT_read": true}
