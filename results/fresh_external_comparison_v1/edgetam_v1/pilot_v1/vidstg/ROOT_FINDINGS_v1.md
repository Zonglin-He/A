# vidstg: fixed128 EdgeTAM spatial comparison

All128queries/128parents sealed before independent official CPU labels, followed by independent full packed-mask boxes/reference/input/SHA/means audit. The four rows share each query’s actual prepared-current Ours time interval. This is a conditional spatial comparison, not independent full-tube temporal inference.

|Spatial trajectory|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|SourceMatchedTime|35.202435|14.521920|18.750000|7.812500|
|ExpertInterpolation|35.202435|14.865927|18.750000|8.593750|
|EdgeTAM|35.202435|17.071386|22.656250|10.937500|
|Ours|35.202435|15.339259|19.531250|10.156250|

Parent-paired10000-bootstrap pointwise95% CIs, in percentage points:

- EdgeTAM-SourceMatchedTime: v+2.549466, CI[0.09059048549545681, 5.219976261252901]; 57 improve/48 harm/23 tie; severe t/v/union 0/19/19, rescue union35.
- EdgeTAM-ExpertInterpolation: v+2.205459, CI[1.3545275315759677, 3.1976936981901343]; 71 improve/29 harm/28 tie; severe t/v/union 0/2/2, rescue union21.
- Ours-EdgeTAM: v-1.732127, CI[-2.8449461660157476, -0.7448140187394539]; 29 improve/71 harm/28 tie; severe t/v/union 0/15/15, rescue union4.

Short/long strata (GT event duration<=5s/>5s), same parent CI:

- short_le5s, n78, EdgeTAM-SourceMatchedTime: v+1.741244, CI[-0.014883929606550057, 3.5069735246969373].
- short_le5s, n78, EdgeTAM-ExpertInterpolation: v+1.444494, CI[0.6131398624955413, 2.3784181538663205].
- short_le5s, n78, Ours-EdgeTAM: v-1.187225, CI[-2.2056623539672144, -0.3135444639366557].
- long_gt5s, n50, EdgeTAM-SourceMatchedTime: v+3.810293, CI[-1.8697976022609988, 9.988637599995725].
- long_gt5s, n50, EdgeTAM-ExpertInterpolation: v+3.392564, CI[1.6810632281061135, 5.428169757016442].
- long_gt5s, n50, Ours-EdgeTAM: v-2.582175, CI[-4.939353346613623, -0.4702647262622826].

EdgeTAM exceeds the current method and interpolation in both fixed cohorts on average; all harmed cases remain, and no direction/checkpoint/configuration is selected from these scores. Same eight expert BOX prompts are used, while current correction also consumes the original expert masks/features; same observations are not equal compute or identical information interfaces. EdgeTAM uses an extra13.9M-parameter propagation model and the full native frame grid, all eight conditioning prompts before forward propagation from frame0, no reverse fusion and no current-method DP. Observed output boxes are exactly the original prompts; empty propagated masks are zero area without fallback.

Conditional query mean 9.325454s, request mean 8.090321s, peak query GPU 577460736bytes. This includes cached Source/current-expert inputs and raw checkpoint/provenance work; it is not complete uncached deployment latency. All query costs, actual worker/model loads, two extra successful controls and failed partial attempt remain in cost lineage. Nested stages are not additive; missing failed-request clock is not zero. The250 inherited current-expert inputs retain acquisition cost, and6 actual new current-expert calls were shared preparation.

Only128parents per domain, not all12205queries. Resource feasibility and prediction correctness support a later separately frozen full extension; they do not authorize selecting a method from these exposed scores. CG-STVG/gamma/corruption remain user-paused, additional CG-HC cancelled; no original frozen method or paper changed.
