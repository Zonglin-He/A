# Measured interpretation: endpoints, admission and delivered update

The saved32-query evidence supports a specific explanation of the average gap: sampled endpoints remain close to the center endpoint, while the fixed denominator24 and admitted count shrink the delivered update. Scaling reduces beneficial center updates more than it protects harmful updates in these cohorts. The endpoint/support difference recovers only a small part of that loss. This is not a finding that larger updates are always better, nor a causal proof that every failure is caused by amplitude.

## 1. Exact saved-score path

| Complete-vIoU difference, pp | HC16 | Vid16 |
|---|---:|---:|
| Center_p minus Center_full: admission scaling path | -0.098512 | -0.013693 |
| Samples24 minus Center_p: endpoint plus support path | +0.015056 | +0.001604 |
| Sum: Samples24 minus Center_full | -0.083456 | -0.012089 |

The row sum is exact per query and after averaging, not an inferred attribution. Endpoint/support term95% intervals are HC[-0.009620,+0.052012], Vid[-0.000064,+0.004240]. On the same common-admitted nodes, endpoint-only saved sIoU changes are HC+0.034310pp[-0.080566,+0.143031], Vid+0.035133[-0.011877,+0.084867]. There is no established improvement in endpoints after controlling the scalar p. Same p still does not match displacement norm, so this is not a pure direction-only intervention.

Outside common support, Samples24-minus-Center_p also includes newly admitted nodes. There are HC5/Vid7 sampled-only nodes and HC2/Vid0 center-only nodes. On queries without either exclusive-support type, the saved full-tube endpoint path is HC+0.000115pp[n10;CI -0.018254,+0.021529], Vid+0.001822[n13;CI -0.000138,+0.005150]. These are descriptive subgroups, not a changed main denominator or independent results. Never call the whole full-tube endpoint/support term a pure endpoint-quality gain.

## 2. Teacher outputs have little diversity relative to changed inputs

On41HC/22Vid common-admitted nodes, the IoU between the sampled encoded-mean teacher endpoint and the center teacher endpoint averages0.978334/0.998090. Their mean source-scaled distance is0.095014/0.032712. Unshrunk sample-versus-center displacement-norm ratios average0.997351/1.003755; they are not generally proposing much larger corrections before p is applied.

For nodes with at least two admitted samples, the median within-node teacher/input RMS dispersion ratio is0.057887HC and0.024427Vid in source-scaled coordinates. Thus different accepted input references usually yield much more concentrated teacher endpoints. This is consistent with the interface mostly returning a similar localization endpoint rather than extracting substantially different state-dependent information. Concentration is not correctness: high agreement may share the same error. No identity claim is inferred.

No per-state GT IoU or GT score of the unshrunk mean endpoint was saved for this new32 cohort. Those quantities remain unmeasured in this cache-only task. Current matched-support saved outputs permit comparison of delivered updates, not a new claim that every sampled teacher is less accurate than the center teacher.

## 3. p mixes legality with acceptance, and it is not an accuracy weight

p=n/24=(L/24)*(n/L), where L is the number of whole states legal at all four anchors. Mean active-query legal fractions are0.783333HC and0.837121Vid. On common-admitted nodes, mean p is0.304878/0.765152, median0.291667/0.708333. These are ratios under the fixed sampling/legality/admission protocol, not calibrated correctness probabilities.

NV008 is a concrete legality case: only15/24 states are legal, and every legal state is admitted at all four anchors, so all four p values are0.625. Scaling is present even without a rejection among legal teacher outputs. Its Center_full-minus-E is+0.238363pp; Center_p-minus-E+0.147627; Samples24-minus-E+0.166282. The original whole-state rejection and24 denominator are frozen; this diagnosis does not replace them, clamp coordinates or resample.

NH004 has20legal states but p values1/24,4/24,1/24,0 across its anchors. Its center-only node loses the update altogether under sampled p. Center_full-minus-E is+1.000429pp, Center_p+0.078117, Samples24+0.061579. The exact path loss is-0.922312pp from scaling and-0.016538pp from endpoint/support difference. This case also contains a changed sampled endpoint, so its entire result is not assigned to amplitude alone.

## 4. Actual geometry confirms suppression, with important counterexamples

| Common-node mean source-scaled update norm | HC | Vid |
|---|---:|---:|
| Center_full | 0.439073 | 1.671693 |
| Samples24 | 0.134382 | 1.269403 |
| Center_p | 0.135572 | 1.264516 |

The ratios of mean Samples24 to Center_full norm are0.306059/0.759352. Physical center and log-size displacement also shrink; these are actual cached outputs, not just a nominal p argument. On common nodes the mean amplitude-vector term norm is0.303501/0.407177, compared with0.027802/0.022864 for the endpoint-difference term. Norms do not add like signed task-metric effects.

Using the already saved center-minus-E query outcome only as an offline descriptive label: HC8center-benefit queries contribute-0.136193pp to the full16 mean from p scaling; the3center-harm queries contribute+0.037680pp from reduced harm. The net is-0.098512pp. Vid4center-benefit queries contribute-0.013693pp; the sole center-harm query gets no saved complete-vIoU protection from scaling. Therefore suppression removes useful correction as well as harmful correction, and in this cohort the useful loss is larger.

NH008 illustrates lost benefit: Center_full gains+1.538001pp versus E, Samples24+0.653289. Its p values0.25,0.5,0.541667,0.625 shrink the update. The gap-0.884712pp decomposes into-0.858509scaling and-0.026203endpoint/support. NH007 illustrates protection: center harms by-0.181400pp, while24states gain+0.035740pp; the path is+0.213465scaling and+0.003674endpoint/support. NH014 also gets+0.233165pp protection from scaling, partly offset by-0.053121pp endpoint difference. These counterexamples rule out a universal prescription to increase every update.

NV014 is the other counterexample to p as quality: the only Vid query with a harmful center update has mean p0.989583 across its center-admitted nodes, while the center-benefit group averages0.810606. Its harm remains about-0.0745pp in complete vIoU. This one-query observation is not a population calibration test or a per-node identity judgment, but it prevents treating high admission as proof that an update is beneficial.

NH011 preserves an example favoring sampled feedback: Samples24-minus-center is+0.330469pp, of which+0.083815 is scaling and+0.246654 is endpoint/support. One anchor is sampled-only. The cache does not isolate whether that full-tube gain comes from the sampled-only node or changed common endpoints, so no new per-node causal score is asserted.

## Conclusion and limits

The immediate bottleneck is not absence of sampled feedback: additional sampled-only support exists. It is that the sampled interface mostly returns a similar endpoint, while p attenuates both useful and harmful updates and does not reliably preserve the former over the latter. The exact saved-score path explains the average gap; it does not uniquely identify all nonlinear interactions or establish a superior alternative rule.

Do not tune p, lower the0.35threshold, change the denominator, expand the cohort or add dynamic rounds based on this diagnosis. Preserve E+MBR and the frozen endpoint candidate. Sampling-specific added value remains unestablished. A different decision would require a separately authorized matched test of endpoint information versus update weighting, not more rounds of the unchanged rule.

This task used existing32cached predictions and saved metrics only:0new model/DINO/Source/GT/media/weights accesses,0new scoring/predictions/samples,0GPU. Original files and CURRENT remain hash exact. Missing cached endpoint-GT quantities remain unknown. All signed cases, subgroup sizes and intervals are public; historical full-result exposure remains disclosed.
