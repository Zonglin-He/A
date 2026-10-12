# P8: prediction support x adaptation support, sealed-cache2x2

All512 original clean cells/256 historically exposed parents/two orders are fixed. A=(I0,Omega0), B=(I1,Omega0), E=(I0,Omega1), C=(I1,Omega1). E uses exactly the saved C boxes and changes only output time. D is the unchanged projected Direct reference. Zero new GPU fits, model calls, DINO calls or optimization. All2048 logical readouts were sealed before this offline GT score.

Parent statistics first average original orders, then equal-weight parents;10000 paired bootstrap, seed20261012. Old32/added96/all128 are separate. Intervals are conditional on saved histories, unadjusted for multiple reported contrasts. CI crossing zero is not an equivalence/weak-interaction conclusion; no equivalence margin was fitted.

| Target | Subset | Contrast | Parents | Mean pp |95% CI pp|>5pp harm|>20pp harm|
|---|---|---|---:|---:|---|---:|---:|
|hc2|all128|temporal_Omega0_v|128|+4.5603|[+2.7898,+6.3433]|10|3|
|hc2|all128|temporal_Omega1_v|128|+4.7274|[+2.9126,+6.5976]|9|2|
|hc2|all128|observation_I0_v|128|-0.6244|[-1.6181,+0.2141]|10|1|
|hc2|all128|observation_I1_v|128|-0.4574|[-1.9369,+0.7667]|12|1|
|hc2|all128|interaction_v|128|+0.1670|[-0.4472,+0.7950]|5|1|
|hc2|previous_P6_32|temporal_Omega0_v|32|+3.8165|[+0.3464,+7.1622]|4|1|
|hc2|previous_P6_32|temporal_Omega1_v|32|+3.9332|[+0.4283,+7.3841]|3|1|
|hc2|previous_P6_32|observation_I0_v|32|+0.0637|[-1.7507,+1.6453]|2|0|
|hc2|previous_P6_32|observation_I1_v|32|+0.1804|[-1.7081,+2.1027]|2|0|
|hc2|previous_P6_32|interaction_v|32|+0.1167|[-0.6090,+0.8165]|1|0|
|hc2|added96|temporal_Omega0_v|96|+4.8083|[+2.7094,+6.9118]|6|2|
|hc2|added96|temporal_Omega1_v|96|+4.9921|[+2.8535,+7.2132]|6|1|
|hc2|added96|observation_I0_v|96|-0.8538|[-2.0731,+0.0855]|8|1|
|hc2|added96|observation_I1_v|96|-0.6700|[-2.5301,+0.7976]|10|1|
|hc2|added96|interaction_v|96|+0.1838|[-0.6079,+0.9798]|4|1|
|vidstg|all128|temporal_Omega0_v|128|+1.4157|[+0.6409,+2.2823]|6|0|
|vidstg|all128|temporal_Omega1_v|128|+1.4975|[+0.7256,+2.3655]|5|0|
|vidstg|all128|observation_I0_v|128|+0.0170|[-0.8240,+0.8035]|4|2|
|vidstg|all128|observation_I1_v|128|+0.0988|[-0.6623,+0.8479]|4|1|
|vidstg|all128|interaction_v|128|+0.0818|[-0.0541,+0.2407]|0|0|
|vidstg|previous_P6_32|temporal_Omega0_v|32|+2.0559|[+0.2503,+4.2609]|1|0|
|vidstg|previous_P6_32|temporal_Omega1_v|32|+2.1401|[+0.3122,+4.3570]|1|0|
|vidstg|previous_P6_32|observation_I0_v|32|+0.4759|[-0.3171,+1.5394]|0|0|
|vidstg|previous_P6_32|observation_I1_v|32|+0.5601|[-0.3079,+1.7513]|0|0|
|vidstg|previous_P6_32|interaction_v|32|+0.0843|[-0.0412,+0.2881]|0|0|
|vidstg|added96|temporal_Omega0_v|96|+1.2023|[+0.3822,+2.1290]|5|0|
|vidstg|added96|temporal_Omega1_v|96|+1.2833|[+0.4676,+2.2034]|4|0|
|vidstg|added96|observation_I0_v|96|-0.1360|[-1.2361,+0.8329]|4|2|
|vidstg|added96|observation_I1_v|96|-0.0550|[-1.0322,+0.8511]|4|1|
|vidstg|added96|interaction_v|96|+0.0810|[-0.0930,+0.2813]|0|0|

|Target|Subset|A mvIoU %|B mvIoU %|E mvIoU %|C mvIoU %|D reference mvIoU %|
|---|---|---:|---:|---:|---:|---:|
|hc2|all128|23.5020|28.0624|22.8776|27.6050|26.9781|
|hc2|previous_P6_32|22.2644|26.0808|22.3281|26.2613|24.4838|
|hc2|added96|23.9146|28.7229|23.0608|28.0529|27.8095|
|vidstg|all128|18.3366|19.7523|18.3536|19.8511|19.9794|
|vidstg|previous_P6_32|16.5787|18.6346|17.0546|19.1947|18.6064|
|vidstg|added96|18.9225|20.1248|18.7866|20.0699|20.4371|

The interaction is exactly (C-E)-(B-A)=(C-B)-(E-A). Its framewise identity uses the official vIoU envelope denominator, distinct from tIoU temporal union: sum_f[(event and I1)/den1-(event and I0)/den0]*(IoU_Omega1-IoU_Omega0). Gained/lost/common-event contributions sum exactly to each cell interaction. This is a metric decomposition, not identification of expert quality or one optimizer variable as a cause.

Within A/B and E/C, the entire spatial trajectory and fixed-GT-support sIoU are identical. Interaction tIoU and sIoU are exactly zero; recall interactions can differ because thresholds are nonlinear. Genuine unobserved decoder event frames exclude both allocations' visited-frame union; interpolation-only and unsupported-event frames are separate. Undefined partitions retain their valid/undefined counts.

All old P7 source/math/state roots are bound and reused; this root independently rereads sealed coordinates and sources, dense metrics, support masks, every decomposition, all anonymous rows and all parent/bootstrap statistics. It does not rerun a model or independently replay the entire decoder Jacobian.

Severe negative parents for every main contrast are preserved in SEVERE_NEGATIVE_PARENTS.csv. Full per-order/old32/new96/all128 vectors, R@0.3/.5, support change, reference quality and all zero/undefined cases remain in ROWS/SUMMARY. No paper writing, gate, reward/temperature/lr sweep, roster change, P7-B retry or production promotion was performed.
