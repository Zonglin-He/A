# Native temporal prior + Native2D: fixed cached module ablation

**Conclusion:** Native2D feedback produces positive Source-relative temporal gains without Global6 on this exposed cached panel. Replacing Global6 is not a uniformly equivalent simplification: pooled means are slightly lower than production Full, and HC final32 has a larger paired loss. Production is unchanged.

Only the prior changes: native final-layer two-offset physical-hull posterior replaces the six-layer Global6 mean BOTH in soft prototype seeds and the final posterior prior. Original two-offset hull, physical MBR, native projected2D features, fulltime trajectory context1.5, kappa64/beta1/ONEround, evidence8, complete-box propagation and exact path DP are unchanged. No parameter search or per-query acceptance.

## Existing128 per direction: descriptive, not independent confirmation

These are the SAME exposed calibration sources (dev96 + previously fixedconfirm32); no new holdout claim. NativeSpatial uses original native time with unchanged BoxFixed, isolating the temporal feedback from spatial benefits.

| Direction | Method | mean tIoU (%) | complete mean vIoU (%) |
|---|---|---:|---:|
|hc2|Source|47.9152|25.0100|
|hc2|NativeSpatial|47.9152|27.7910|
|hc2|Global6|48.1961|27.7951|
|hc2|Full|49.6854|28.6759|
|hc2|NativeFeedback|49.1896|28.4036|
|vidstg|Source|38.9896|15.6450|
|vidstg|NativeSpatial|38.9896|19.2687|
|vidstg|Global6|40.8179|19.9398|
|vidstg|Full|41.2280|19.9203|
|vidstg|NativeFeedback|40.7986|19.8332|

## Paired precision and severe declines

Deltas are percentage points, with10000 parent-cluster bootstrap95%CI. Severe means time OR complete-v drop>5pp, counted once per query; separate time/v counts can overlap.

| Direction | NativeFeedback compared with | delta t [95%CI] | delta v [95%CI] | time improve/harm/tie | severe time/v/union |
|---|---|---|---|---|---|
|hc2|Source|+1.2744 [+0.6345, +1.9156]|+3.3935 [+1.1763, +5.5005]|64/27/37|6/14/19 of128|
|hc2|NativeSpatial|+1.2744 [+0.6345, +1.9156]|+0.6126 [+0.1984, +1.0348]|64/27/37|6/3/6 of128|
|hc2|Global6|+0.9935 [-0.1776, +2.2749]|+0.6085 [-0.1730, +1.4852]|58/44/26|22/10/23 of128|
|hc2|Full|-0.4958 [-1.4648, +0.6400]|-0.2723 [-0.9279, +0.4929]|35/56/37|19/9/20 of128|
|vidstg|Source|+1.8090 [+0.5909, +3.3290]|+4.1882 [+1.6447, +6.7279]|51/28/49|4/18/21 of128|
|vidstg|NativeSpatial|+1.8090 [+0.5909, +3.3290]|+0.5644 [+0.0047, +1.2262]|51/28/49|4/1/4 of128|
|vidstg|Global6|-0.0193 [-1.0684, +1.0999]|-0.1066 [-0.5760, +0.3933]|42/48/38|10/4/10 of128|
|vidstg|Full|-0.4294 [-1.1358, +0.1996]|-0.0871 [-0.4128, +0.2285]|34/48/46|8/1/8 of128|

Comparing severe counts must keep the baseline fixed. Versus Source: NativeFeedback union HC19/128 and Vid21/128; production Full27/128 and28/128. Versus Global6: NativeFeedback HC23/128 and Vid10/128; production Full9/128 and9/128. Lower Source-relative harm does not imply stronger Global6-relative protection.

## Previously fixed32 separately retained

| Direction | NativeFeedback t / v (%) | delta Source t / v (pp) | delta Full t / v (pp) | severe vs G6 NativeFeedback / Full |
|---|---|---|---|---|
|hc2|45.4918 / 25.6347|+1.6248 / +2.2889|-2.9464 / -1.7407|7/32 / 1/32|
|vidstg|38.8957 / 17.1398|+1.7331 / +5.9252|+0.3146 / +0.1767|1/32 / 3/32|

HC final32 NativeFeedback-Full: t-2.9464pp CI[-4.6543,-1.2985], v-1.7407pp CI[-2.9062,-.6863]. Vid final32 differences have CIs spanning zero. No selection is performed on these ablation results.

## Boundary and duration evidence

The JSON reports include physical boundary errors, interval lengths, zero-overlap rates, standard thresholds, observed/unobserved spatial support, dev96/final32 groups and full signed deltas. Source-relative short/long t deltas on descriptive128 are HC+.4605/+1.5023pp and Vid+.2332/+3.2437pp; these subgroup results do not establish new independent validation.

## Reproducibility and cost

Fingerprint `7d9ca4cbb10b21ba37058c1b9079c7ef21375729d295892fc788b1054b0cb944`. All256 predictions and posterior traces sealed BEFORE a separate CPU process loaded fixed official labels. Full posterior/interval/boxes and Source/Global6/positions matched sealed reference outputs bitwise for all256. Native no-feedback MBR exactly recovered original Source intervals. Constant-feature zero-contrast test is exact native no-op.

Official-vs-dense metric checks: 1280, max absolute error 1.33e-15. CPU prediction replay 79.32s; independent scoring 13.74s. These are cached ablation processing times, NOT uncached deployment latency. New video/model/visual encoder/expert/observation/training/GPU calls:0.

Production full plan,42 bound files, parameters, owned processes and GPU lock are unchanged. The Source+.9pp EACHdirection acceptance still awaits complete official production evaluation. This study does not replace production or retune any parameter.

- [Fixed protocol](protocol.json)
- [Complete report and signed per-query metrics](report.json)
- [All paired signed results](signed.tsv)
- [All severe paired declines](severe.tsv)
- Code: `studies/native_prior_v1/run.py`
