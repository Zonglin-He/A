# hc2: Source-rectangular-mask-prompt EdgeTAM, single-expert full tubes

All128unique parents independently scored and all-mask/input/SHA/means audited after whole prediction seal. Each complete pipeline uses its own available event. EdgeTAM receives only rectangular masks rasterized from original Source boxes at Source-selected eight positions, and uses Source native time. The current branch uses ReferDINO and its own temporal update/finalDP.

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|
|CurrentSimple|46.509478|26.879620|45.312500|15.625000|
|CurrentOurs|54.968548|31.791298|50.781250|22.656250|
|BoxPromptEdgeTAM|46.509478|26.227517|45.312500|16.406250|
|EdgeTAM|46.509478|25.360726|43.750000|17.968750|

Pointwise parent-paired10000-bootstrap95% CIs:

- EdgeTAM-BoxPromptEdgeTAM: t+0.000000pp CI[0.0, 0.0]; v-0.866791pp CI[-1.7328153275641787, -0.15670499787214198]; severe t/v/union 0/11/11, rescue union3.
- EdgeTAM-Source: t+0.000000pp CI[0.0, 0.0]; v+0.236473pp CI[-0.6100171267835606, 0.929634908701269]; severe t/v/union 0/7/7, rescue union10.
- CurrentOurs-EdgeTAM: t+8.459071pp CI[5.507059352429634, 11.49512358151015]; v+6.430573pp CI[3.5457681216079635, 9.314575103065401]; severe t/v/union 18/23/31, rescue union76.
- CurrentOurs-CurrentSimple: t+8.459071pp CI[5.507059352429634, 11.49512358151015]; v+4.911678pp CI[2.908269145459872, 6.924963412010876]; severe t/v/union 18/16/22, rescue union74.

GT-duration short/long reporting is observational and not used for inference/selection:

- short_le5s, n42, EdgeTAM-BoxPromptEdgeTAM: v-1.178538pp CI[-3.264993640299019, 0.12043521029419524]; t+0.000000 CI[0.0, 0.0].
- short_le5s, n42, EdgeTAM-Source: v-0.319285pp CI[-2.4421789693178213, 1.0915250700310763]; t+0.000000 CI[0.0, 0.0].
- short_le5s, n42, CurrentOurs-EdgeTAM: v+3.790527pp CI[0.029135829878888804, 7.7950032208811075]; t+3.124530 CI[-1.6311748558741703, 7.81248523796055].
- short_le5s, n42, CurrentOurs-CurrentSimple: v+1.355058pp CI[-2.1842523244173018, 4.798607515702112]; t+3.124530 CI[-1.6311748558741703, 7.81248523796055].
- long_gt5s, n86, EdgeTAM-BoxPromptEdgeTAM: v-0.714543pp CI[-1.4979341067971519, 0.04285548488986705]; t+0.000000 CI[0.0, 0.0].
- long_gt5s, n86, EdgeTAM-Source: v+0.507889pp CI[-0.18710680240642819, 1.1937847600310938]; t+0.000000 CI[0.0, 0.0].
- long_gt5s, n86, CurrentOurs-EdgeTAM: v+7.719897pp CI[3.9236924025706963, 11.563256666983579]; t+11.064312 CI[7.3082225729666765, 14.90077110091356].
- long_gt5s, n86, CurrentOurs-CurrentSimple: v+6.648633pp CI[4.22969959043269, 9.151529164869348]; t+11.064312 CI[7.3082225729666765, 14.90077110091356].

The explicit BoxPromptEdgeTAM arm is the already independently verified original Source-box prompt result on these exact parents/Source input/event. New EdgeTAM tracking here uses only Source-derived rectangular masks. No ReferDINO enters either EdgeTAM branch; boxes are kept exactly on observed frames.

This controls the number of external experts, removing the old ReferDINO-to-EdgeTAM serial relationship. It does not equalize pretrained knowledge, input resolution, frame processing, FLOPs, mask-versus-box evidence, or temporal modules. Any current-minus-EdgeTAM difference includes prompt quality and temporal inference differences, and cannot isolate spatial propagation quality. All128tracking requests/domain were new Source-prompt executions; none of the old teacher-prompt predictions were inherited. The old stacked result remains an additional-model extension only.

Actual conditional query sum1060.093196s; request sum788.365373s; processed native frames8320, actual image encoder calls9344; peak allocated query GPU572798464bytes. Model load is one actual PID/start shared across domains. Original Source acquisition remains nonzero; its Source-only clock fields are kept separately, and no old ReferDINO acquisition is charged to the Source-Edge branch. Cached-input query/request/IO/stage clocks overlap and are not complete uncached deployment latency. Missing inherited clock fields remain null, not zero. No scored-case exclusion, parameter/prompt/threshold/direction selection, main-method change, or full12205 result is claimed.
