# Oracle–Mixer Gap Audit: complete source diagnosis

Previously scored source diagnosis, GT oracle and privileged evidence; not held-out or target evaluation.

**DECISION = A**; recommended branch = A.
Oracle native advantage, poor learned mapping; conditioning explanation remains a hypothesis

## Native policy results

|Arm|tIoU %|sIoU %|vIoU %|delta v vs B1, pp|95% parent CI, pp|
|---|---:|---:|---:|---:|---|
|B1|46.637512|48.627481|32.407600|—|—|
|seed20260928|46.887168|47.764785|32.167599|-0.240001|[-1.121475, +0.713042]|
|seed20260929|46.741185|48.246816|32.295832|-0.111768|[-0.915397, +0.750465]|
|oracle|53.402853|60.069317|44.679649|+12.272049|[+8.979625, +15.496381]|

Oracle complete practical qualification: **False**. Individual gates: {"v_positive": true, "v_lower_CI_positive": true, "t_nonnegative": true, "s_nonnegative": true, "no_severe_parent_v_harm": false, "native_good_retained": false}.

## Direction and finite outcome

|Readout|Seed1|Seed2|
|---|---:|---:|
|Cosine to oracle: mean / median|0.001459 / 0.000697|0.001767 / 0.000833|
|Positive temporal local descent fraction|0.588101|0.585812|
|Positive spatial local descent fraction|0.736239|0.713303|
|Fraction at >=99% norm cap|1.000000|1.000000|
|B1 v>.5 group delta-v: query mean pp|-2.599373|-2.788875|
|Other group delta-v: query mean pp|+0.754308|+1.094320|

The stratified query means above are not parent-macro effects. A positive local descent dot is not a finite native gain.

## All nine oracle / learned outcome combinations

|Oracle / learned|Seed1 count|Seed2 count|
|---|---:|---:|
|gain/gain|115|123|
|gain/harm|128|117|
|gain/flat|72|75|
|harm/gain|33|39|
|harm/harm|42|40|
|harm/flat|18|14|
|flat/gain|0|1|
|flat/harm|0|0|
|flat/flat|39|38|

## Retention, tails and evidence limits

Native-good counts: {"vIoU": {"seed20260928": {"eligible": 136, "retained": 127}, "seed20260929": {"eligible": 136, "retained": 126}, "oracle": {"eligible": 136, "retained": 122}}, "tIoU": {"seed20260928": {"eligible": 199, "retained": 185}, "seed20260929": {"eligible": 199, "retained": 184}, "oracle": {"eligible": 199, "retained": 164}}}.

Query changes exceeding5pp (both losses and gains): {"seed20260928": {"vIoU": {"loss_below_minus5pp": 38, "gain_above5pp": 35}, "sIoU": {"loss_below_minus5pp": 36, "gain_above5pp": 44}, "tIoU": {"loss_below_minus5pp": 41, "gain_above5pp": 37}}, "seed20260929": {"vIoU": {"loss_below_minus5pp": 33, "gain_above5pp": 32}, "sIoU": {"loss_below_minus5pp": 36, "gain_above5pp": 30}, "tIoU": {"loss_below_minus5pp": 45, "gain_above5pp": 34}}, "oracle": {"vIoU": {"loss_below_minus5pp": 66, "gain_above5pp": 245}, "sIoU": {"loss_below_minus5pp": 97, "gain_above5pp": 206}, "tIoU": {"loss_below_minus5pp": 84, "gain_above5pp": 190}}}.

All31 anonymous parent results and all group summaries are included in ORACLE_MIXER_GAP.json. No case, invalid geometry or neutral outcome was removed.

Independent validation: {"raw_episodes": 447, "backwards": 873, "norm_dot_max_abs": 1.8189894035458565e-12, "field_relative_L2_max": 2.9519750512953706e-07, "CE_FP64_max_abs": 1.3824220248537245e-06, "scalar_tensor_max_abs": 4.440892098500626e-16, "reused_metric_max_abs": 0.0, "summary_checks": 391, "summary_max_abs": 3.375077994860476e-14, "decision_checks": 3517, "decision_max_abs": 5.164875800265853e-10}.

Descriptive unadjusted parent bootstrap CI; two learned seeds share cases; analytic one-step GT oracle is not an upper bound. Local alignment is not finite native causality. Fresh31/388 remains metadata-only.

## Execution and next boundary

A/B/C only CPU modules/protocols; no candidate training or rescue measurement, no external/OPD/target run.

Measured current-stage GPU receipts total 6998.314231s; all-history total 70591.882244s, cap=null. All CPU audit times are recorded separately.

A/B support would motivate a separately registered source test. An INCONCLUSIVE/C recommendation is not a C1/C2/C3 result. No new scientific experiment starts from this report.
