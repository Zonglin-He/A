# Reliable R1 Direct feedback: one uniform interface correction

Same32 exposed development parents (HC16/Vid16),128 fixed T2 anchors, original unmarked context4 and standalone original-aspect image. All128 old raw responses were audited without GT or new model calls; malformed unique numeric literals were never silently admitted. One source-independent synthetic-grammar-tested wire requests an answer-wrapped four-number pixel array or null. It is not claimed calibrated on real Source images. Processor pixels/clock, original checkpoint, greedy256 cap/repetition penalty and original attention backend remain fixed. There is no Prompt search, duplicate-answer recovery, coordinate-unit guessing/clamp or OPD parameter search.

R1_FIXED_OPD uses unchanged original1792-parameter episodic Gaussian32/response/fullstep rollback/reject-stop. R1_FIXED_D2 uses exactly the same new references and unchanged physical-time Source-relative center/log-size propagation. R1_ORIGINAL_D2 independently uses the original Direct references, so a DINO/R1-ID propagation result is not substituted. All prediction/root families were sealed before GT; no-reference and extreme sources remain in all32. Original Frozen/T2/WF/Direct/DINO predictions and saved scalars match. This is development evidence, not a new independent confirmation or production promotion.

Official model card lists HCSTVG-v1/v2 and VidSTG training domains. The paper reports removing HC validation/test duplicates from training; the local release lacks a complete checkpoint-bound parent manifest, so exact sample overlap is unresolved. No inference-time target GT does not imply a target-domain-unseen teacher. The teacher was trained around visually marked object IDs, not direct coordinate regression. Original sources: https://huggingface.co/XiaowenZhang/stvg-r1-model-7b ; https://arxiv.org/html/2602.11730v1 ; https://github.com/XiaowZhang00/STVG-R1 .

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|31.250000|25.000000|
|hc2|T2|27.487135|50.497117|53.666875|37.500000|18.750000|
|hc2|Current_WF|28.641217|52.179161|53.666875|37.500000|25.000000|
|hc2|R1_DIRECT|27.972261|51.210459|53.666875|43.750000|18.750000|
|hc2|D0|28.440075|51.967586|53.666875|37.500000|25.000000|
|hc2|R1_ORIGINAL_D2|26.324097|48.674131|53.666875|37.500000|18.750000|
|hc2|R1_FIXED_OPD|28.580257|52.108541|53.666875|37.500000|25.000000|
|hc2|R1_FIXED_D2|25.756999|44.559898|53.666875|31.250000|18.750000|
|vidstg|Frozen|12.185607|32.870500|35.923576|12.500000|0.000000|
|vidstg|T2|12.239461|32.870500|35.680949|12.500000|0.000000|
|vidstg|Current_WF|12.072499|32.573620|35.680949|12.500000|0.000000|
|vidstg|R1_DIRECT|14.323998|35.542683|35.680949|18.750000|0.000000|
|vidstg|D0|15.368798|33.573892|35.680949|25.000000|6.250000|
|vidstg|R1_ORIGINAL_D2|11.320556|30.896872|35.680949|12.500000|6.250000|
|vidstg|R1_FIXED_OPD|14.572580|31.878854|35.680949|18.750000|0.000000|
|vidstg|R1_FIXED_D2|13.222003|36.900708|35.680949|12.500000|6.250000|

| Domain | Contrast | delta vIoU pp | paired t95 pp | + / - / neutral / unknown |
|---|---|---:|---|---|
|hc2|R1_FIXED_OPD_minus_T2|1.093122|[0.40841700157761174, 1.7778273414173666]|12/1/3/0|
|hc2|R1_FIXED_OPD_minus_Current_WF|-0.060960|[-0.6358737176854079, 0.5139533306809865]|8/7/1/0|
|hc2|R1_FIXED_OPD_minus_R1_DIRECT|0.607997|[-0.22834089984043415, 1.4443339733207818]|10/3/3/0|
|hc2|R1_FIXED_D2_minus_R1_FIXED_OPD|-2.823259|[-15.331653555672439, 9.685136140796157]|3/11/2/0|
|hc2|R1_ORIGINAL_D2_minus_R1_DIRECT|-1.648164|[-4.019996092209006, 0.7236682770684109]|1/4/11/0|
|vidstg|R1_FIXED_OPD_minus_T2|2.333120|[-0.8591691388368914, 5.525408188846429]|7/3/6/0|
|vidstg|R1_FIXED_OPD_minus_Current_WF|2.500082|[-0.9289200895048979, 5.929083650827083]|8/3/5/0|
|vidstg|R1_FIXED_OPD_minus_R1_DIRECT|0.248582|[-2.02645941003291, 2.523623378444438]|4/6/6/0|
|vidstg|R1_FIXED_D2_minus_R1_FIXED_OPD|-1.350577|[-6.388865098824897, 3.6877109533279038]|4/8/4/0|
|vidstg|R1_ORIGINAL_D2_minus_R1_DIRECT|-3.003442|[-9.765151430544577, 3.7582668617075767]|4/4/8/0|

REFERENCE_QUALITY.json reports coverage and successes over every GT-known planned anchor, counting invalid/missing/abstained outputs as failures. Conditional geometry and all nonanchor/source-paired metrics are retained separately. No syntax or IoU proxy is presented as independent identity/role review. No further automatic Prompt or optimizer search; Qwen model continuation remains paused by user.

## Measured interpretation and matched-support audit

Legal references rise HC12->58/64 and Vid23->53/64. Across every GT-known planned anchor, successes@.5 rise HC1->17/39 and Vid7->8/30, with25/34 positions per domain still unknown. Actual fit sources rise HC5->15/16 and Vid9->15/16. New accepted/proposed steps are71/72 HC and37/39 Vid versus old25/25 and27/27; no optimizer settings changed. Common available GT-known positions are8 HC and17 Vid, with mean reference-IoU changes+3.077210/+7.709473pp; these are descriptive support subsets, not the primary source cohort or causal estimates.

The fixed OPD vIoU gain relative to original Direct is+.607997/+.248582pp; relative to T2 it is+1.093122/+2.333120pp. HC reference usability and geometry improve materially within this exposed cohort. Vid legality improves much more than all-position geometric success, and fixed OPD dense sIoU31.878854% remains below both original Direct35.542683% and T2 32.870500%; vIoU alone does not establish an across-trajectory geometry improvement. Relative to original Direct, removing one largest positive source only as posthoc sensitivity gives+.352466pp HC but-.635861pp Vid, while every source stays in the primary mean. The original large-negative R032 recovers13.515229pp, while R025 newly loses8.103532pp; both are retained.

For the exact same new sparse references, original OPD exceeds D2 mean vIoU by2.823259pp HC/1.350577pp Vid, but the paired intervals remain wide and cross zero. With original Direct references, OPD likewise exceeds its own D2 control by1.648164/3.003442pp. These contrasts support keeping OPD as the tested candidate and do not establish universal or statistically confirmed irreplaceability. This single interface correction repairs a major availability bottleneck; Vid reference reliability and generalization remain unresolved. No automatic optimizer search, new Prompt, new confirmation or production promotion follows.

The initial Student launch failed at runtime-source-pin validation before any Source model or fit, then002 added the missing existing fit/audit SHA bindings and corrected inherited template labels. All128 Teacher outputs/references and scientific configs were preserved, with no outcome retry. COST.json includes the actual002 fits/roots and the original failed launch. ENGINEERING_RECOVERY.json records the limited correction.
