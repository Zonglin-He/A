# HC2 official full: native temporal prior + Native2D

**Complete:**1902 official queries /227 parent sources, queryequal aggregation. All predictions sealed before independent CPU official labels.

**Acceptance: TARGET NOT MET.** Full-Source mean tIoU +0.747072pp is below the user-fixed +0.9pp threshold by0.152928pp. This remains false despite a positive parent-clusterCI. No parameter selection, inference replay or method promotion follows this full result. Vid full continues with exactly the same frozen method.

| Method | m_vIoU (%) | vIoU@.3 (%) | vIoU@.5 (%) | m_tIoU (%) |
|---|---:|---:|---:|---:|
|Source|24.9948|37.8023|12.8812|48.6473|
|Interpolation|25.1065|38.2755|13.9853|48.6473|
|OursS|26.8147|40.9043|16.0358|48.6473|
|Global6|26.7597|40.4311|15.0894|48.5808|
|Full|27.1561|41.3775|17.0347|49.3944|

OursS uses original native temporal readout with the same complete-box spatial framework. Global6 is only the imported archived baseline, NOT the prior in Full. All HistoricalOPD reference fields remain in report.json with the protocol audit; it is historical project OPD, not third-party genericTTA.

## Paired contribution

| Comparison | delta complete vIoU pp [95%CI] | delta tIoU pp [95%CI] |
|---|---|---|
|Full-Source|+2.1613 [+1.6605,+2.6746]|+0.7471 [+0.5127,+0.9876]|
|Full-OursS|+0.3414 [+0.2018,+0.4843]|+0.7471 [+0.5127,+0.9876]|
|Full-Global6|+0.3964 [+0.1967,+0.6017]|+0.8136 [+0.5095,+1.1196]|

Bootstrap uses10000 parent-source cluster resamples, queryequal estimate. Full-Source spatial/complete-v gains cannot all be credited to temporalfeedback. Full-OursS isolates native-prior Native2D temporal feedback: complete-v+.3414pp and time+.7471pp.

## Four module rows

| Configuration | complete vIoU (%) | delta Full (pp) | unobserved sIoU (%) | tIoU (%) |
|---|---:|---:|---:|---:|
|Full|27.1561|+0.0000|52.7793|49.3944|
|NoLocal|26.6983|-0.4578|51.9662|49.3944|
|Independent|26.8750|-0.2811|52.1976|49.3944|
|OursS|26.8147|-0.3414|52.7883|48.6473|

NoLocal and Independent keep Full time and the exact eight observations. NativeTemporal=OursS; its spatial process uses native time normally. Shared initialization is reported as a mechanism stage, not a deletion ablation.

## Duration and damage

| Group | queries | delta Source tIoU (pp) | delta Source full-v (pp) |
|---|---:|---:|---:|
|short_le5s|544|+0.1050|+1.9101|
|long_gt5s|1358|+1.0043|+2.2620|

Relative Source, time improves845queries / harms558 / unchanged499. Time decline>5pp:148/1902; complete-v decline>5pp:245/1902; union341/1902 (17.9285%). Relative OursS (same spatial framework), severe time148 and severe v60, union148/1902. Every signed and severe record is retained, with explicit baseline.

Official-vs-dense checks 17118, max absolute error 1.33e-15. Source and imported Global6 aggregate metrics exactly recover the archived official full. Parent/effective-query counts, fixed support, observed/unobserved metrics, complete-vs-mask candidate analysis, physical-distance curve and natural-cache worker segments are in report.json.

## CPU scorer repair, no inference change

The original frozen scorer exited before its first metric because its severe-TSV loop named a local variable metric, shadowing the global metric function throughout run(). Original source/log/plan/seal are preserved privately. The independent tools/score_adapter_nativeprior.py only alpha-renames4 AST Name nodes inside that loop to metric_field; reverse editing recovers the entire original AST exactly. Formulas, official labels, samples, predictions, bootstrap and thresholds are unchanged. All1902 prediction SHA values are identical before/after scoring; no GPU/model/expert forward was executed. See CPU_SCORE_ADAPTER.json for hashes/proof. The frozen original script is not edited.

- [Report, all metrics and paired intervals](report.json)
- [All severe pair records](severe_declines.tsv)
- [Union-risk summary](RISK_SUMMARY.json)
- [Source+.9 acceptance](ACCEPTANCE.json)
- Full signed tables: signed_01.tsv and signed_02.tsv
