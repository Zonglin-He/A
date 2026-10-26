# Single-expert complete pipelines: fixed128 parents per domain

Each branch has one external pretrained expert. Source-prompt EdgeTAM uses only original Source boxes and Source native time, without ReferDINO, current posterior/time or current DP. ReferFormer substitutes the expert and rebuilds its own mask geometry, E-ROI features, posterior, event and final DP. The current method uses ReferDINO. These interfaces and compute are different; complete-tube differences do not isolate spatial propagation. The old serial ReferDINO-to-EdgeTAM result is an additional-model extension and is excluded from this single-expert table.

## hc2

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|
|ReferDINO_current|54.968548|31.791298|50.781250|22.656250|
|Source_box_EdgeTAM|46.509478|26.227517|45.312500|16.406250|
|ReferFormer_current|54.769180|27.552349|43.750000|17.187500|

Current-minus-Source-box-EdgeTAM v+5.563781pp, parent-paired95%CI[2.7409141146104608, 8.377100511933943]; ReferFormer-current minus ReferDINO-current v-4.238950pp, CI[-6.94380071135964, -1.6846865614959465].

## vidstg

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|
|ReferDINO_current|35.202435|15.339259|19.531250|10.156250|
|Source_box_EdgeTAM|32.121535|15.257178|19.531250|9.375000|
|ReferFormer_current|34.603561|15.639079|20.312500|10.156250|

Current-minus-Source-box-EdgeTAM v+0.082081pp, parent-paired95%CI[-2.2593527542132277, 2.372827096676974]; ReferFormer-current minus ReferDINO-current v+0.299820pp, CI[-1.9986399694270065, 2.604791396154297].

Full signed/tails/short-long parent CIs are retained in each study. All CIs are pointwise; the fixed cohorts have prior development exposure. A CI crossing zero is not equivalence or expert-independence proof. No model/checkpoint/threshold/prompt direction is selected from these results. Source-only EdgeTAM has unchanged Source temporal readout by design, whereas the other complete methods have their own temporal inference. Single-expert count is controlled, not pretrained knowledge, processed frames, input resolution or compute.

Both256 small groups are complete; no full12205 completion or complete uncached deployment latency is claimed. CG, gamma and corruption remain paused by the user; additional CG-HC cancelled.
