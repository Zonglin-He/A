# Current kappa0 actual deployment timing

Actual192 measured Source calls and96 same-eight-observation expert requests, plus4Source/2expert warmup calls. Every output was bitwise equal to the current audited NoShrink prediction and its MAP interval. These are new current-kappa0 measurements, not renamed legacy kappa64 times. Root actually reran the read-only timing export and verified its seal, all-call TSV and paired statistics.

|Domain|Independent parents/queries|Repeats|Source sec/query|Full sec/query|Extra sec [95% CI]|Full/Source ratio [95% CI]|
|---|---:|---:|---:|---:|---:|---:|
|hc2|16|3|3.245172747|4.127469027|0.882296280 [0.842934798, 0.931518604]|1.271880 [1.254950, 1.288263]|
|vidstg|16|3|2.853141015|3.842596420|0.989455405 [0.859296858, 1.164938863]|1.346795 [1.311159, 1.391501]|

Each query first averages its3repeats;10000 paired-parent bootstrap draws use16 independent units/domain, not48 independent queries. Confidence intervals are pointwise. Same RTX5090/i5-14600KF, native224FP32/two offsets/eight observations and fixed input cohort. Models resident with original CPU/GPU offload. No inter-query RGB/feature/capture/teacher/prediction cache. Own CPU experimental groups were paused during timing and resumed afterward; shared-host conditional measurement, not a full-domain throughput guarantee. Original startup and warmup are separate. Prediction-reference verification IO is outside the timer; decode, parse, transfers, native Source, complete expert request, geometry, auxiliary DP, actual ROI repooling, prototype update, jointMAP, final DP and cleanup are inside the query wall.

## Mean stages in complete-method calls

|Domain|RGB decode|Native Source|Complete expert request|Expert forward (nested)|Geometry|Aux DP|ROI pooling|Prototype|MAP readout|Final DP|Finish wall (overlapping)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|hc2|0.727934|2.011605|0.655342|0.183734|0.088841|0.001151|0.002850|0.001143|0.000017|0.001157|0.006322|
|vidstg|0.188461|2.140540|0.645240|0.184816|0.113018|0.001454|0.003565|0.001269|0.000021|0.001498|0.007813|

Times are seconds. Expert forward is inside expert request, and several stages are inside finish wall; these columns must not be summed. The all192callTSV also retains parsing/transfers/per-call variation. Current Source and Full calls share the same fixed hardware, warmup and no-query-cache rules.

Hardware and complete paired results are in paired_timing.json; original startup/warmup and nested stage means are in stage_costs.json. No new model or GT scoring in root export.
