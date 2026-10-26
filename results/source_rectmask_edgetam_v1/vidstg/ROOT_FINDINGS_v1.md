# vidstg: Source-rectangular-mask-prompt EdgeTAM, single-expert full tubes

All128unique parents independently scored and all-mask/input/SHA/means audited after whole prediction seal. Each complete pipeline uses its own available event. EdgeTAM receives only rectangular masks rasterized from original Source boxes at Source-selected eight positions, and uses Source native time. The current branch uses ReferDINO and its own temporal update/finalDP.

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|
|CurrentSimple|32.121535|13.867903|17.968750|7.812500|
|CurrentOurs|35.202435|15.339259|19.531250|10.156250|
|BoxPromptEdgeTAM|32.121535|15.257178|19.531250|9.375000|
|EdgeTAM|32.121535|15.104594|19.531250|7.812500|

Pointwise parent-paired10000-bootstrap95% CIs:

- EdgeTAM-BoxPromptEdgeTAM: t+0.000000pp CI[0.0, 0.0]; v-0.152584pp CI[-0.7694517787520182, 0.43068694377981603]; severe t/v/union 0/8/8, rescue union4.
- EdgeTAM-Source: t+0.000000pp CI[0.0, 0.0]; v+1.781691pp CI[1.2097438929799336, 2.3644236307867144]; severe t/v/union 0/1/1, rescue union23.
- CurrentOurs-EdgeTAM: t+3.080900pp CI[0.7959085296291982, 5.611244478097894]; v+0.234665pp CI[-2.0849989074697657, 2.540723124835495]; severe t/v/union 21/26/34, rescue union55.
- CurrentOurs-CurrentSimple: t+3.080900pp CI[0.7959085296291982, 5.611244478097894]; v+1.471356pp CI[0.2315213952099042, 2.817800827449867]; severe t/v/union 21/13/24, rescue union47.

GT-duration short/long reporting is observational and not used for inference/selection:

- short_le5s, n78, EdgeTAM-BoxPromptEdgeTAM: v+0.265476pp CI[-0.22526302043448126, 0.7779927903446882]; t+0.000000 CI[0.0, 0.0].
- short_le5s, n78, EdgeTAM-Source: v+1.419786pp CI[0.7300442588896231, 2.12365224592795]; t+0.000000 CI[0.0, 0.0].
- short_le5s, n78, CurrentOurs-EdgeTAM: v-1.303755pp CI[-3.3914601247167515, 0.7118100936850316]; t-0.552505 CI[-2.401443869986275, 1.3012908728656365].
- short_le5s, n78, CurrentOurs-CurrentSimple: v-0.246146pp CI[-1.3495918038329964, 0.7845087915369285]; t-0.552505 CI[-2.401443869986275, 1.3012908728656365].
- long_gt5s, n50, EdgeTAM-BoxPromptEdgeTAM: v-0.804758pp CI[-2.1944028128055573, 0.43076477961977055]; t+0.000000 CI[0.0, 0.0].
- long_gt5s, n50, EdgeTAM-Source: v+2.346264pp CI[1.3931252122509097, 3.355595973956356]; t+0.000000 CI[0.0, 0.0].
- long_gt5s, n50, CurrentOurs-EdgeTAM: v+2.634601pp CI[-2.5078480383048496, 7.743229909971911]; t+8.749011 CI[4.203408202871081, 13.965947699051359].
- long_gt5s, n50, CurrentOurs-CurrentSimple: v+4.150660pp CI[1.464663153371928, 6.932496408754038]; t+8.749011 CI[4.203408202871081, 13.965947699051359].

The explicit BoxPromptEdgeTAM arm is the already independently verified original Source-box prompt result on these exact parents/Source input/event. New EdgeTAM tracking here uses only Source-derived rectangular masks. No ReferDINO enters either EdgeTAM branch; boxes are kept exactly on observed frames.

This controls the number of external experts, removing the old ReferDINO-to-EdgeTAM serial relationship. It does not equalize pretrained knowledge, input resolution, frame processing, FLOPs, mask-versus-box evidence, or temporal modules. Any current-minus-EdgeTAM difference includes prompt quality and temporal inference differences, and cannot isolate spatial propagation quality. All128tracking requests/domain were new Source-prompt executions; none of the old teacher-prompt predictions were inherited. The old stacked result remains an additional-model extension only.

Actual conditional query sum1287.786468s; request sum1174.108066s; processed native frames13502, actual image encoder calls14548; peak allocated query GPU609871360bytes. Model load is one actual PID/start shared across domains. Original Source acquisition remains nonzero; its Source-only clock fields are kept separately, and no old ReferDINO acquisition is charged to the Source-Edge branch. Cached-input query/request/IO/stage clocks overlap and are not complete uncached deployment latency. Missing inherited clock fields remain null, not zero. No scored-case exclusion, parameter/prompt/threshold/direction selection, main-method change, or full12205 result is claimed.
