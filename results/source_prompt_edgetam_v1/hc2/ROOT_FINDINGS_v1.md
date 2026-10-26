# hc2: Source-original-box-prompt EdgeTAM, single-expert full tubes

All128unique parents independently scored and all-mask/input/SHA/means audited after whole prediction seal. Each complete pipeline uses its own available event. EdgeTAM receives only original Source boxes at Source-selected eight positions, and uses Source native time. The current branch uses ReferDINO and its own temporal update/finalDP.

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|
|CurrentSimple|46.509478|26.879620|45.312500|15.625000|
|CurrentOurs|54.968548|31.791298|50.781250|22.656250|
|EdgeTAM|46.509478|26.227517|45.312500|16.406250|

Pointwise parent-paired10000-bootstrap95% CIs:

- EdgeTAM-Source: t+0.000000pp CI[0.0, 0.0]; v+1.103264pp CI[0.637409457476999, 1.5526544328782232]; severe t/v/union 0/1/1, rescue union12.
- CurrentOurs-EdgeTAM: t+8.459071pp CI[5.507059352429634, 11.49512358151015]; v+5.563781pp CI[2.7409141146104608, 8.377100511933943]; severe t/v/union 18/23/31, rescue union76.
- CurrentOurs-CurrentSimple: t+8.459071pp CI[5.507059352429634, 11.49512358151015]; v+4.911678pp CI[2.908269145459872, 6.924963412010876]; severe t/v/union 18/16/22, rescue union74.

GT-duration short/long reporting is observational and not used for inference/selection:

- short_le5s, n42, EdgeTAM-Source: v+0.859253pp CI[0.08265414492715985, 1.6881069413345002]; t+0.000000 CI[0.0, 0.0].
- short_le5s, n42, CurrentOurs-EdgeTAM: v+2.611989pp CI[-1.0024200275333295, 6.491925922293496]; t+3.124530 CI[-1.6311748558741703, 7.81248523796055].
- short_le5s, n42, CurrentOurs-CurrentSimple: v+1.355058pp CI[-2.1842523244173018, 4.798607515702112]; t+3.124530 CI[-1.6311748558741703, 7.81248523796055].
- long_gt5s, n86, EdgeTAM-Source: v+1.222432pp CI[0.6523321271514754, 1.7720984309835874]; t+0.000000 CI[0.0, 0.0].
- long_gt5s, n86, CurrentOurs-EdgeTAM: v+7.005354pp CI[3.3041191920185753, 10.696479290302412]; t+11.064312 CI[7.3082225729666765, 14.90077110091356].
- long_gt5s, n86, CurrentOurs-CurrentSimple: v+6.648633pp CI[4.22969959043269, 9.151529164869348]; t+11.064312 CI[7.3082225729666765, 14.90077110091356].

This controls the number of external experts, removing the old ReferDINO-to-EdgeTAM serial relationship. It does not equalize pretrained knowledge, input resolution, frame processing, FLOPs, mask-versus-box evidence, or temporal modules. Any current-minus-EdgeTAM difference includes prompt quality and temporal inference differences, and cannot isolate spatial propagation quality. All128tracking requests/domain were new Source-prompt executions; none of the old teacher-prompt predictions were inherited. The old stacked result remains an additional-model extension only.

Actual conditional query sum1048.969198s; request sum770.339113s; processed native frames8320, actual image encoder calls9344; peak allocated query GPU537671168bytes. Model load is one actual PID/start shared across domains. Original Source acquisition remains nonzero; its Source-only clock fields are kept separately, and no old ReferDINO acquisition is charged to the Source-Edge branch. Cached-input query/request/IO/stage clocks overlap and are not complete uncached deployment latency. Missing inherited clock fields remain null, not zero. No scored-case exclusion, parameter/prompt/threshold/direction selection, main-method change, or full12205 result is claimed.
