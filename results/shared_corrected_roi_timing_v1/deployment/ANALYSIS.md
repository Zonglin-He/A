# Actual shared-corrected-box ROI deployment timing

Completed192measuredSourcecalls/96actualexpertcalls, plus4Source/2expertwarmupcalls excluded. All measured outputs exactly match independently audited selected-method nativeSource/FullMAP predictions and intervals. Worker exited0; only owned CPU-sync group was paused and finally resumed. Root read the complete measurements, seals, all-call TSV and repeat-averaged paired bootstrap; this export performs zero model/Source/expert/optimizer/GT calls.

|Domain|Source seconds|Selected Full seconds|Paired overhead seconds [95% CI]|Full/Source ratio [95% CI]|
|---|---:|---:|---:|---:|
|hc2|3.216442663|4.033143537|+0.816700875 [0.779585152, 0.853670150]|1.253914 [1.226061, 1.281727]|
|vidstg|2.778438133|3.704309338|+0.925871205 [0.824130545, 1.055843559]|1.333234 [1.306987, 1.364570]|

Three repeats are averaged within each query before16paired-parent bootstrap10000(seed20260920) per domain. Repeats are not independent queries. CIs are pointwise, not familywise. Fixed16queries/parents per domain are the original panel, not selected by outcomes and not full-domain throughput. Source is original native per-offset MAP plus physical hull.

## Measured path and cache rules

Original224FP32two-offsetSource, sameeightjoint expert observations, shared correction, complete geometry, actual direct corrected-box ROI pooling, one kappa0/beta1 prototype update, jointMAP interval and final exact trajectory DP. The first auxiliary DP is absent and its measured selected-method cost is exactly0 on all96Fullcalls. Wall includes RGBdecode/parse/CPU-GPU transfers/Source/complete expert request/geometry/pool/prototype/MAP/finalDP/cleanup. Weights resident with original offload; no interquery RGB/feature/teacher/prediction cache. Reference verification I/O and explicit resource guards outside query timer. Startup and warmups separately reported. Shared-host conditional measurement.

## Nested components

The expert request includes preprocessing, transfers, IPC and forward; the forward alone is not complete extra latency. Source/pipeline/finish/request subintervals can be nested and must not be added.

|Domain|Expert request s|Expert forward s|Geometry s|First auxiliary DP s|E-ROI pool s|Prototype s|Final DP s|Finish wall s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|hc2|0.621549828|0.184683771|0.089488941|0.000000000|0.002868461|0.001152331|0.001244825|0.005286851|
|vidstg|0.610848174|0.185009590|0.110962018|0.000000000|0.003543580|0.001231996|0.001534134|0.006334193|

## Hardware, startup and evidence

{
  "GPU": "NVIDIA GeForce RTX 5090, 32607 MiB, 580.178.04",
  "CPU": [
    {
      "field": "Architecture:",
      "data": "x86_64"
    },
    {
      "field": "CPU(s):",
      "data": "20"
    },
    {
      "field": "Model name:",
      "data": "Intel(R) Core(TM) i5-14600KF"
    },
    {
      "field": "Thread(s) per core:",
      "data": "2"
    },
    {
      "field": "Core(s) per socket:",
      "data": "14"
    },
    {
      "field": "Socket(s):",
      "data": "1"
    }
  ],
  "OS": "Linux-7.0.0-34-generic-x86_64-with-glibc2.39",
  "python": "3.11.16",
  "torch_threads": 4,
  "native_Source": "Original224FP32twooffsets",
  "same_expert_observations": 8
}

Full per-query3repeat means are in paired_timing.json; every measured call and all recorded components are in all_measured_calls.tsv. All startup/checkpoint-load/warmup calls are in report.json. Raw query IDs/captions/media remain private. Original two-DP timing is preserved as historical data: the two methods were measured in separate runs, not a direct interleaved old-vs-new experiment. Differences between absolute run means cannot be attributed entirely to auxiliary-DP removal, especially because Source baselines also changed and the old auxiliary DP itself was a small nested term.
