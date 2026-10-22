# Actual shared-episode positive event reading

Exposed HC64 development, not independent confirmation. NativeG6 boundaries define a common at-least49-node window BEFORE actual positive first/last text-conditioned inference. No chronology constraint, likelihood ratio, query update or confidence rejection. All52 supported descriptions receive EpisodeFull/First/Last and matched-width shifted First/Last. Twelve unsupported descriptions remain native. Original visual feature values, global-offset positions, two-pass native inference, physical MBR and complete-box DP stay fixed. Cached motion still contains original clip context.

Text encoding reused within query across context variants and explicitly counted. Odd-start windows reorder local parity while retaining original frame IDs/positions. All64 global Full text/hidden/posterior/Source boxes restore bitwise. Source/Native/archived GlobalClause aliases exact. All64 seal before independent CPU GT scoring.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|EpisodeClauseFusion|31.1091|58.1532|31/21/12|11|
|GlobalClauseFusion|31.3453|58.4307|30/21/13|11|
|EpisodeFullFusion|29.5192|55.6926|16/13/35|6|
|EpisodeSelfHullFusion|30.3416|57.2473|33/17/14|5|
|ShiftedEpisodeClauseFusion|28.2983|52.8638|24/28/12|23|
|DirectEpisodeClauses|31.0635|57.8245|29/23/12|16|
|NoComposition|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−G6 t+2.1919pp CI[-0.2210125927074555, 4.586755332026386], v+1.4140pp CI[0.09328244914740215, 2.83218474808308]; Source t+3.3302, v+4.2153pp.
Source shortt-4.7402pp; longt+5.3873pp.
Primary−GlobalClauseFusion t-0.2775pp CI[-1.4189594934839262, 0.8215619401768285].
Primary−EpisodeFullFusion t+2.4606pp CI[0.11929904006828819, 4.723890913300001].
Primary−EpisodeSelfHullFusion t+0.9059pp CI[-0.6625227549393798, 2.5075218456747215].
Primary−ShiftedEpisodeClauseFusion t+5.2894pp CI[0.8233994920727946, 10.212393694179458].
Primary−DirectEpisodeClauses t+0.3287pp CI[-1.154154651880398, 1.9522857176984025].

Not qualified; no window/alpha scan or posthoc promotion of controls.
Worker 71.37s inclloading/research recording; calls{'text_forwards': 168, 'fusion_offsets': 648, 'native_replay_queries': 324, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. This is not deployment end-to-end cost. Official/dense checks1152 maxerror6.66e-16.

[Signed outcomes](signed.tsv), [severe declines](severe_declines.tsv), [Source groups](source_duration_groups.json), [protocol](protocol.json).
