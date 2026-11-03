# Offline spatial support, expert quality and risk: hc2

Existing independently audited frame IoUs and scalar outcomes are reused. New event-membership masks, query/form/quality joins, counts, means and paired CIs passed separate-process independent recomputation. No new predictions, model, Source, expert, optimizer, media access or GT metric scoring. GT intervals/anchor quality are offline labels only, never an online gate.

|Support / nearest-observation distance|Queries|Parents|Annotated frames|Ours IoU %|Interpolation IoU %|Paired delta IoU pp [95% CI]|
|---|---:|---:|---:|---:|---:|---|
|all_unobserved/all_distances|1902|227|38695|52.94278680777401|47.33687743624276|+5.605909 [+4.779928, +6.481026]|
|all_unobserved/bin0|1894|226|11050|54.356542454811574|54.443898929065384|-0.087356 [-0.808779, +0.631346]|
|all_unobserved/bin1|1894|226|15154|52.992386985620406|46.97387961618057|+6.018507 [+5.102124, +6.966805]|
|all_unobserved/bin2|1715|223|10662|51.45624616337421|40.572757381505525|+10.883489 [+9.563179, +12.209912]|
|all_unobserved/bin3|561|180|1829|50.021272812815354|38.031488254078944|+11.989785 [+9.377648, +14.678963]|
|all_unobserved/bin4|0|0|0|None|None|not scorable|
|predicted_event/all_distances|1731|222|27334|53.8860107413703|48.03375431976207|+5.852256 [+5.079912, +6.658582]|
|predicted_event/bin0|1662|221|9122|55.79922581670453|54.679811572480446|+1.119414 [+0.496489, +1.739980]|
|predicted_event/bin1|1693|222|11240|53.96450833752224|47.31893851676127|+6.645570 [+5.757577, +7.576376]|
|predicted_event/bin2|1429|214|6203|52.303052076326296|40.567410377756595|+11.735642 [+10.510977, +13.035105]|
|predicted_event/bin3|314|141|769|52.239514632637416|37.282916492392374|+14.956598 [+11.770961, +18.191611]|
|predicted_event/bin4|0|0|0|None|None|not scorable|
|GT_event/all_distances|1902|227|38496|52.92795076282372|47.355784931901916|+5.572166 [+4.743033, +6.445149]|
|GT_event/bin0|1894|226|11009|54.33394719649821|54.420667586600295|-0.086720 [-0.811650, +0.633368]|
|GT_event/bin1|1893|226|15073|52.92881592595304|46.93310752386517|+5.995708 [+5.084803, +6.953662]|
|GT_event/bin2|1714|223|10600|51.45651615804472|40.62739129222405|+10.829125 [+9.501847, +12.168427]|
|GT_event/bin3|558|180|1814|50.00120154328132|38.098407142441594|+11.902794 [+9.252854, +14.612577]|
|GT_event/bin4|0|0|0|None|None|not scorable|

Distance bins: [0,.5), [.5,1), [1,2), [2,4), [4,infinity) seconds. Frames must be actually annotated native sampled UNOBSERVED frames, with physical nearest-anchor distance / fps. Prediction membership uses exactly the Table10 current joint-MAP interval; GT membership is half open as in the official scorer. Empty cells are explicitly unscorable; primary query-equal per-bin means and parent CIs are distinct from descriptive frame-weighted means. This diagnoses where spatial gains occur; it cannot uniquely decompose full dense vIoU or causally explain query form/quality effects.

|Query form / expert-quality third|Queries|Spatial path minus interpolation v pp [95% CI]|Unobserved IoU delta pp [95% CI]|
|---|---:|---|---|
|all/all|1902|+2.134753 [+1.814042, +2.473029]|+5.605909 [+4.779928, +6.481026]|
|all/low|409|+1.684329 [+1.187875, +2.239225]|+7.204357 [+5.706258, +8.814925]|
|all/medium|459|+3.082107 [+2.345330, +3.808384]|+7.653524 [+5.954223, +9.291898]|
|all/high|1012|+1.895004 [+1.494193, +2.322774]|+3.943191 [+2.856550, +5.044758]|
|all/unscorable|22|+1.771747 [+0.111024, +3.643650]|+9.653669 [+2.801736, +16.806953]|

Expert quality is the original mean IoU on GT-supported expert anchors only. Fixed thirds [0,1/3), [1/3,2/3), [2/3,1], no score-selected quantiles; missing remains unscorable. Declarative/interrogative labels are exact catalog joins, not punctuation heuristics.

|Risk group|Queries|Delta v pp|Gain >+1 pp %|Loss <-1 pp %|Loss <=-10 pp %|Any negative %|
|---|---:|---|---|---|---|---|
|all|1902|+6.644365 [+5.902588, +7.408451]|+59.095689 [+56.729610, +61.434690]|+23.922187 [+21.976337, +25.889603]|+9.200841 [+8.038563, +10.378921]|+27.129338 [+25.083968, +29.166667]|
|Source_v_ge_0.5|170|-4.899389 [-7.433090, -2.334076]|+41.764706 [+34.285714, +49.411765]|+54.705882 [+47.282362, +62.275449]|+33.529412 [+26.923077, +40.251777]|+56.470588 [+49.112426, +63.975776]|
|Source_v_ge_0.8|0|not scorable|not scorable|not scorable|not scorable|not scorable|
|Source_t_ge_0.8|137|-1.155072 [-4.891685, +2.589071]|+43.065693 [+34.931507, +51.163248]|+52.554745 [+44.055944, +60.870382]|+29.197080 [+21.640746, +36.879433]|+54.014599 [+45.636644, +62.307692]|

All thresholds predeclared; strong Source groups are vIoU>=.5, vIoU>=.8 and tIoU>=.8. Query equal, parent10000 pointwise95% CIs, not familywise. No new method adoption or parameter selection. Inherited acquisition costs nonzero, offline cached runtime is not deployment latency.
