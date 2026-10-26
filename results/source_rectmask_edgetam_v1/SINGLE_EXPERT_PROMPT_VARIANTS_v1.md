# Single external-expert complete tubes: eight Source prompts and expert substitution

Same original confirm128parents/domain, all source/current scalar rows crossjoined exactly. Source-box and Source-rectangle-mask EdgeTAM both use original Source event/eight positions, the exact same Source boxes, RGB/grid, checkpoint, direction and observed-box writeback. The mask version rasterizes boxes into binary rectangles and uses official add_new_mask, never referring-expert masks. It tests prompt representation, not extra segmentation annotation. Current and ReferFormer each have their own temporal update. One external expert count is controlled, not pretraining, resolution, processed frames, compute or isolated propagation quality. Old ReferDINO-to-EdgeTAM stacked extension is retained separately, not placed in this table.

## hc2

|Complete pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|
|ReferDINO_current|54.968548|31.791298|50.781250|22.656250|
|Source_box_EdgeTAM|46.509478|26.227517|45.312500|16.406250|
|Source_rectangle_mask_EdgeTAM|46.509478|25.360726|43.750000|17.968750|
|ReferFormer_current|54.769180|27.552349|43.750000|17.187500|

EdgeTAM-BoxPromptEdgeTAM: v-0.866791pp parent-paired10000-bootstrap95%CI[-1.7328153275641787, -0.15670499787214198]; t+0.000000CI[0.0, 0.0].

CurrentOurs-EdgeTAM: v+6.430573pp parent-paired10000-bootstrap95%CI[3.5457681216079635, 9.314575103065401]; t+8.459071CI[5.507059352429634, 11.49512358151015].
## vidstg

|Complete pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|
|ReferDINO_current|35.202435|15.339259|19.531250|10.156250|
|Source_box_EdgeTAM|32.121535|15.257178|19.531250|9.375000|
|Source_rectangle_mask_EdgeTAM|32.121535|15.104594|19.531250|7.812500|
|ReferFormer_current|34.603561|15.639079|20.312500|10.156250|

EdgeTAM-BoxPromptEdgeTAM: v-0.152584pp parent-paired10000-bootstrap95%CI[-0.7694517787520182, 0.43068694377981603]; t+0.000000CI[0.0, 0.0].

CurrentOurs-EdgeTAM: v+0.234665pp parent-paired10000-bootstrap95%CI[-2.0849989074697657, 2.540723124835495]; t+3.080900CI[0.7959085296291982, 5.611244478097894].

All signed effects/severe union/rescues/shortlong and actual cost are preserved in the two per-domain root reports. Rectangular masks do not provide native object contours; no second expert or ground truth produces them. Pointwise CIs crossing zero do not prove equivalence. Cohorts previously exposed; no score-selected checkpoint/prompt/direction/threshold, no full12205 result or complete uncached deployment speed. Model load one actual PID/start shared by domains, old Source acquisition nonzero, old ReferDINO acquisition not charged to the two Source-only EdgeTAM branches. Stacked-extra-capacity experiment remains separately available. Main profile/paper unchanged.
