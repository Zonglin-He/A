# hc2: fixed128 EdgeTAM spatial comparison

All128queries/128parents sealed before independent official CPU labels, followed by independent full packed-mask boxes/reference/input/SHA/means audit. The four rows share each query’s actual prepared-current Ours time interval. This is a conditional spatial comparison, not independent full-tube temporal inference.

|Spatial trajectory|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|SourceMatchedTime|54.968548|30.123874|49.218750|23.437500|
|ExpertInterpolation|54.968548|29.385144|46.093750|16.406250|
|EdgeTAM|54.968548|34.218990|53.125000|27.343750|
|Ours|54.968548|31.791298|50.781250|22.656250|

Parent-paired10000-bootstrap pointwise95% CIs, in percentage points:

- EdgeTAM-SourceMatchedTime: v+4.095116, CI[1.3288322291228767, 6.941028269733838]; 80 improve/35 harm/13 tie; severe t/v/union 0/17/17, rescue union48.
- EdgeTAM-ExpertInterpolation: v+4.833846, CI[3.5472941817931662, 6.226239729071176]; 88 improve/27 harm/13 tie; severe t/v/union 0/2/2, rescue union45.
- Ours-EdgeTAM: v-2.427692, CI[-3.6232612422368544, -1.2373652271876434]; 30 improve/84 harm/14 tie; severe t/v/union 0/32/32, rescue union7.

Short/long strata (GT event duration<=5s/>5s), same parent CI:

- short_le5s, n42, EdgeTAM-SourceMatchedTime: v+2.925890, CI[-0.8111516000543982, 6.969052134898183].
- short_le5s, n42, EdgeTAM-ExpertInterpolation: v+3.723659, CI[1.612598948302615, 6.215893184949786].
- short_le5s, n42, Ours-EdgeTAM: v-1.797322, CI[-3.9693401842779625, 0.25792891762904546].
- long_gt5s, n86, EdgeTAM-SourceMatchedTime: v+4.666133, CI[0.9876977175177768, 8.442494583469285].
- long_gt5s, n86, EdgeTAM-ExpertInterpolation: v+5.376031, CI[3.7277240655236525, 7.0953028277491565].
- long_gt5s, n86, Ours-EdgeTAM: v-2.735547, CI[-4.176468234287658, -1.2720823973223772].

EdgeTAM exceeds the current method and interpolation in both fixed cohorts on average; all harmed cases remain, and no direction/checkpoint/configuration is selected from these scores. Same eight expert BOX prompts are used, while current correction also consumes the original expert masks/features; same observations are not equal compute or identical information interfaces. EdgeTAM uses an extra13.9M-parameter propagation model and the full native frame grid, all eight conditioning prompts before forward propagation from frame0, no reverse fusion and no current-method DP. Observed output boxes are exactly the original prompts; empty propagated masks are zero area without fallback.

Conditional query mean 7.393277s, request mean 5.432880s, peak query GPU 539308032bytes. This includes cached Source/current-expert inputs and raw checkpoint/provenance work; it is not complete uncached deployment latency. All query costs, actual worker/model loads, two extra successful controls and failed partial attempt remain in cost lineage. Nested stages are not additive; missing failed-request clock is not zero. The250 inherited current-expert inputs retain acquisition cost, and6 actual new current-expert calls were shared preparation.

Only128parents per domain, not all12205queries. Resource feasibility and prediction correctness support a later separately frozen full extension; they do not authorize selecting a method from these exposed scores. CG-STVG/gamma/corruption remain user-paused, additional CG-HC cancelled; no original frozen method or paper changed.
