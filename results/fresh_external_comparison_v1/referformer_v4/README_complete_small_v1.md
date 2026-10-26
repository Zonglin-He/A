# Single-expert replacement: both fixed128-parent domains complete

|Expert and inference|HC tIoU / vIoU %|Vid tIoU / vIoU %|
|---|---:|---:|
|ReferDINO + native-time simple interpolation|46.509478 /26.879620|32.121535 /13.867903|
|ReferDINO + current structured inference|54.968548 /31.791298|35.202435 /15.339259|
|ReferFormer + native-time simple interpolation|46.509478 /23.073061|32.121535 /14.299784|
|ReferFormer + same structured inference, own event|54.769180 /27.552349|34.603561 /15.639079|

ReferFormer structured-minus-own-simple v+4.479288pp CI[2.720726,6.300918] and+1.339296 CI[.288504,2.469619]. Alternative-minus-current complete pipeline v-4.238950CI[-6.943801,-1.684687] on HC, +.299820CI[-1.998640,2.604791] on Vid. Between-expert differences in the within-expert v gain are-.432390CI[-1.657878,.838089] and-.132060CI[-1.255899,.971186]. Positive gains on both fixed cohorts do not establish expert independence, equivalence or performance on the full12205 targets.

Both short-event gain intervals cross zero; Vid short gain-.320108, while long+3.927966CI[1.865728,6.205912]. Severe t/v union once versus own simple25HC/23Vid, rescues67/44. All negative/positive signed pairs, t/v/R.3/R.5 and internal spatial/boundary diagnostics, short-long reports, every model/load/failed-forward/CPU worker cost accompany each domain. No frozen config/checkpoint/prompt direction was selected from these scores.

Actual257 successful alternative expert forwards consist of256logical checkpointed outputs plus one initial unsaved successful forward stopped by a missing save helper. Its original output SHA does not exist and recovery is not claimed; actual failed-load/wholewall and missing forward-only clock retained. Original numerical code is unchanged, execution helper completed in independent versions; two zero-model preflight failures preserved. Source and current-expert acquisition is inherited nonzero. Cached query/request/compatibility and independent auditing elapsed are not complete uncached deployment latency and nested times are not additive.

Each alternative inference uses only original Source plus ReferFormer boxes/masks: current ReferDINO arrays are separate baseline checks, never alternative input. One external expert per branch still does not equalize compute or knowledge. Old ReferDINO->EdgeTAM is only an additional-propagator extension. New Source-original-box-prompt EdgeTAM independently uses Source native time and no ReferDINO evidence; its actual HC first-two full raw/RGB/Source-box/packed-cold bridge passed, but its full256 accuracy results remain pending.
