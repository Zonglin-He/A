# Same-support reference geometry × configuration32

Historical exposed HC16/Vid16,38 common anchors in19 sources;13 unsupported no-op sources retained. Identical support/weights/Source/Adam reset/seeds/final intervals. Short cells are exact sealed reuse; original-package cells are newly executed. All episodic, no LN carry. Historical current5 is not current paper Lite. Original package changes lr/sigma/tau/steps together; not a steps-only ablation or an online-stream reproduction. All new predictions and independent math/output ROOT seals preceded GT scoring.

| Domain | Arm | vIoU % | dense sIoU % | tIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|F|32.587386|53.922657|54.448394|50.000000|31.250000|
|hc2|T|34.066991|53.922657|60.145183|50.000000|31.250000|
|hc2|S_WF_common_SHORT|33.117666|54.685281|54.448394|50.000000|31.250000|
|hc2|TS_WF_common_SHORT|34.643557|54.685281|60.145183|50.000000|37.500000|
|hc2|S_DINO_common_SHORT|33.241644|54.698718|54.448394|50.000000|37.500000|
|hc2|TS_DINO_common_SHORT|34.741339|54.698718|60.145183|50.000000|43.750000|
|hc2|S_WF_common_ORIGINAL|26.821628|45.519128|54.448394|37.500000|25.000000|
|hc2|TS_WF_common_ORIGINAL|28.066390|45.519128|60.145183|37.500000|18.750000|
|hc2|S_DINO_common_ORIGINAL|31.844646|51.864243|54.448394|50.000000|31.250000|
|hc2|TS_DINO_common_ORIGINAL|32.799812|51.864243|60.145183|50.000000|31.250000|
|vidstg|F|9.701443|33.353589|33.628222|12.500000|0.000000|
|vidstg|T|9.966696|33.353589|34.185907|12.500000|0.000000|
|vidstg|S_WF_common_SHORT|10.371622|34.813034|33.628222|12.500000|0.000000|
|vidstg|TS_WF_common_SHORT|10.640229|34.813034|34.185907|12.500000|0.000000|
|vidstg|S_DINO_common_SHORT|10.630311|37.009485|33.628222|12.500000|0.000000|
|vidstg|TS_DINO_common_SHORT|10.924565|37.009485|34.185907|12.500000|0.000000|
|vidstg|S_WF_common_ORIGINAL|11.027306|31.879553|33.628222|12.500000|6.250000|
|vidstg|TS_WF_common_ORIGINAL|11.242133|31.879553|34.185907|12.500000|6.250000|
|vidstg|S_DINO_common_ORIGINAL|12.845797|39.955245|33.628222|18.750000|6.250000|
|vidstg|TS_DINO_common_ORIGINAL|13.001229|39.955245|34.185907|18.750000|6.250000|

Primary interaction: (DINO_original−DINO_short)−(WF_original−WF_short). Same final temporal evaluation support within each S or TS readout.

| Domain/readout/metric | Interaction pp | Paired source t95 pp | + / − / neutral / unknown |
|---|---:|---|---|
|hc2/S_/m_vIoU|4.899040|[-8.026599875585598, 17.824680364023223]|7/2/7/0|
|hc2/S_/m_sIoU_dense_GT|6.331678|[-8.542458113240492, 21.205813839226945]|7/3/6/0|
|hc2/TS_/m_vIoU|4.635640|[-8.560797227878604, 17.83207808864519]|7/2/7/0|
|hc2/TS_/m_sIoU_dense_GT|6.331678|[-8.542458113240492, 21.205813839226945]|7/3/6/0|
|vidstg/S_/m_vIoU|1.559803|[-0.8109787837132858, 3.9305838574835468]|3/1/12/0|
|vidstg/S_/m_sIoU_dense_GT|5.879240|[0.017464330978111953, 11.74101627825496]|6/1/9/0|
|vidstg/TS_/m_vIoU|1.474760|[-0.8590680774406625, 3.8085885290812564]|4/1/11/0|
|vidstg/TS_/m_sIoU_dense_GT|5.879240|[0.017464330978111953, 11.74101627825496]|6/1/9/0|

| Domain | TS contrast | vIoU delta pp | Paired source t95 pp | + / − / neutral / unknown |
|---|---|---:|---|---|
|hc2|TS_WF_common_SHORT_minus_T|0.576566|[-0.39961322843378755, 1.5527452119927743]|6/3/7/0|
|hc2|TS_DINO_common_SHORT_minus_T|0.674348|[0.06633600689677355, 1.2823604584564559]|7/2/7/0|
|hc2|TS_DINO_common_SHORT_minus_TS_WF_common_SHORT|0.097782|[-0.9408864401293235, 1.1364509219235663]|6/3/7/0|
|hc2|TS_WF_common_ORIGINAL_minus_T|-6.000601|[-11.140541806235891, -0.8606597393691091]|1/8/7/0|
|hc2|TS_WF_common_ORIGINAL_minus_TS_WF_common_SHORT|-6.577167|[-11.591432083881065, -1.5629014452829217]|1/8/7/0|
|hc2|TS_WF_common_ORIGINAL_minus_TS_DINO_common_SHORT|-6.674949|[-12.105038811409564, -1.2448591995486646]|1/8/7/0|
|hc2|TS_DINO_common_ORIGINAL_minus_T|-1.267178|[-14.845932912390003, 12.311576709345834]|4/5/7/0|
|hc2|TS_DINO_common_ORIGINAL_minus_TS_WF_common_SHORT|-1.843744|[-15.711146643164358, 12.023658456561204]|4/5/7/0|
|hc2|TS_DINO_common_ORIGINAL_minus_TS_DINO_common_SHORT|-1.941526|[-15.248256740706607, 11.365204072309206]|4/5/7/0|
|hc2|TS_DINO_common_ORIGINAL_minus_TS_WF_common_ORIGINAL|4.733423|[-9.146272289922821, 18.61311763248365]|7/2/7/0|
|vidstg|TS_WF_common_SHORT_minus_T|0.673532|[-0.37450747711090604, 1.7215723952028965]|3/2/11/0|
|vidstg|TS_DINO_common_SHORT_minus_T|0.957868|[-0.03498536229437155, 1.9507221791186131]|5/0/11/0|
|vidstg|TS_DINO_common_SHORT_minus_TS_WF_common_SHORT|0.284336|[-0.9107041277925001, 1.479376026524751]|3/2/11/0|
|vidstg|TS_WF_common_ORIGINAL_minus_T|1.275437|[-3.295795387891025, 5.846668845833968]|2/3/11/0|
|vidstg|TS_WF_common_ORIGINAL_minus_TS_WF_common_SHORT|0.601904|[-3.4563237697275886, 4.6601323095785405]|1/4/11/0|
|vidstg|TS_WF_common_ORIGINAL_minus_TS_DINO_common_SHORT|0.317568|[-4.212182161166758, 4.84731880228546]|2/3/11/0|
|vidstg|TS_DINO_common_ORIGINAL_minus_T|3.034533|[-1.2627387051080587, 7.3318045134238465]|5/0/11/0|
|vidstg|TS_DINO_common_ORIGINAL_minus_TS_WF_common_SHORT|2.361000|[-1.5916964094825152, 6.313697299706313]|4/1/11/0|
|vidstg|TS_DINO_common_ORIGINAL_minus_TS_DINO_common_SHORT|2.076664|[-1.5248078133175373, 5.678136804809084]|3/2/11/0|
|vidstg|TS_DINO_common_ORIGINAL_minus_TS_WF_common_ORIGINAL|1.759096|[-1.515238566303663, 5.033430916676508]|3/2/11/0|

All pairwise source deltas, signs, unknowns and intervals for all metrics are in RESULTS.json; t95 is descriptive on this exposed small development sample. The support is conditional on historical DINO admission. This factorial tests geometry × configuration package, not an isolated step-count mechanism, stronger-MLLM claim, current-B superiority or long-stream online state. Sparse-reference acceptance is not GT correctness. Teacher-reference geometry quality was evaluated previously and is not rerun here. Actual pixel review is engineering inspection, not completion of the512-window semantic panel. Production CURRENT and frozen paper registry remain unchanged.

Execution closure note: all three scientific subprocesses and the controller main exited0, with the last score stage sealed before controller exit. The user systemd manager later reported a90-second stop-cleanup timeout after inotify watch allocation failures. This is an explained post-main cleanup warning, not an invalid prediction/ROOT/GT score or a four-hour compute timeout. The successful stage sum excludes that cleanup wait; remote publication was verified independently afterward. No result was rerun or rewritten.
