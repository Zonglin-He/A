# Bounded Native2D shrinkage selection and fixed parent-separated confirmation

Only kappa in {4,16,64} is varied. ONE shared value for both directions. Original native Source, q0, fulltime BoxFixed context1.5, kernel, one update, MBR, spatial DP and expert eight observations remain fixed. No weights/layers/ROI/iterations search. Native2D is projected appearance BEFORE textfusion; CurrentRegion is the selected postfusion-motion implementation.

Previous32/32 are exposed development. Another32 parents per direction were fixed by Source-only SHA before development scoring, with ZERO parent overlap. These videos have historical original-full/other-method exposure; this is candidate-specific prospective confirmation, NOT an untouched benchmark. All stage outputs sealed before independent CPU official-label scoring.

## Development selection

| Direction | kappa | tIoU (%) | Complete vIoU (%) | Severe time / v vs G6 (>5pp) |
|---|---:|---:|---:|---:|
|hc2|4|50.5420|26.5321|1 / 1|
|hc2|16|48.8882|25.8101|1 / 1|
|hc2|64|46.3908|24.8331|1 / 1|
|vidstg|4|45.1827|21.4734|6 / 4|
|vidstg|16|45.3198|21.5424|5 / 3|
|vidstg|64|45.7373|21.8436|3 / 1|

Predeclared choice: require EACH direction t/v >=K16-.25pp and no increase in severe G6-relative counts; maximize equal-direction mean full-v gain among eligible candidates. K4 macro-v +.3265pp fails Vid severe protection; K64 macro-v -.3379pp fails HC protection. K16 is retained. No per-domain choice, search expansion or confirmation-driven reselection. This fallback is not an improved-parameter result.

## Once-frozen confirmation

| Direction | Method | tIoU (%) | Complete vIoU (%) |
|---|---|---:|---:|
|hc2|Source|47.9134|25.2544|
|hc2|NativeG6|48.1293|29.4338|
|hc2|CurrentRegion|47.6466|28.8952|
|hc2|K16|47.2053|28.4486|
|vidstg|Source|38.8951|18.3325|
|vidstg|NativeG6|39.1467|22.5482|
|vidstg|CurrentRegion|40.5042|23.4596|
|vidstg|K16|41.1937|23.8852|

| Direction | Native2D16 - Current t (pp, 95% parent CI) | Full v (pp, CI) | t improve / harm / tie | >5pp t / v harm vs Current |
|---|---|---|---|---|
|hc2|-0.4413 [-2.4854,+1.9346]|-0.4465 [-1.9904,+1.3926]|8 / 12 / 12|5 / 4|
|vidstg|+0.6896 [-0.5408,+2.1661]|+0.4256 [-0.2382,+1.3430]|9 / 8 / 15|1 / 0|

HC Native2D16 relative Source: t-.7081/v+3.1942pp; short t-.6549 and long-.7290. Vid relative Source: t+2.2986/v+5.5527pp; short t-1.1831 and long+4.3877. These are descriptive fixed-panel outcomes, not long/short success. HC severe G6-relative time/v is11/7 versus Current9/6; Vid2/1 versus Current3/1. Both dimensions and both baselines are retained; no claim that mean tradeoffs eliminate severe tails.

The original development HC gain does not reproduce on confirmation. Confidence intervals for the Native2D-vs-Current aggregate differences cross zero, which does not establish equivalence. Search stops at this point. No evidence of a jointly better shrinkage setting; do not tune again on these confirmation outcomes. The broader feature interface remains a measured feasibility result, not a newly adopted runtime or validated cross-backbone/MLLM method.

## Integrity and real research cost

Dev64*3 predictions, confirmation64 with Selected=K16 exact alias. 704 official metric checks, maximum error 1.332e-15. Original Source boxes/logits/q0 and CurrentRegion q/interval/final boxes are bitwise restored for all64 new confirmations. Confirmation 296.54s wall, one original Source forward/query (64), no additional expert/observation/training. Same-Source feature extraction is included; this is research acquisition cost, not deployment latency. CPU development score 6.91s; confirmation score 5.25s. Earlier v1 lexical preflight failure retained separately (0model/0prediction/0GT).

Runtime: keep the previously user-selected fused-motion RegionKernel full fp09acb7fd unchanged. Temporary pause at HC293 saved every receipt/prediction/capture hash; restore the SAME plan after confirmation. No Native2D/control automatic promotion or full-result parameter selection.

Full signed tables and all severe entries are in dev_signed.tsv, dev_severe.tsv, confirm_signed.tsv and confirm_severe.tsv. Protocol, frozen configuration, selection candidates, CIs, short/long groups and complete anonymous per-query values are retained in the JSON reports. Private video/caption/GT coordinates/features/weights are excluded.
