# GroundingDINO-T / GroundingDINO-B checkpoint comparison32

Completed, measured, independently checked; original NH16/NV16 exposed development parents, one query each. Frozen original TA-STVG Source, full-image four physical observation positions, target context/span, processor, NMS/admission/E source calibration/finalMBR unchanged. Official HF B checkpoint replaces T; no training, OPD, new Source or teacher ensemble. No promotion or automatic next run.

All32 predictions sealed before new offline GT. All parents retained, unknown annotations not fabricated. The five unresolved Vid what queries still have0 calls; B performs108 forwards, not128. Four fresh T NH001 forwards are a numerical/input and small timing control; T scientific results otherwise come from exact cached baseline.

## Full-method results

Mean changes are percentage points, parent-macro; intervals are paired10000 bootstrap, seed20261018.

| B E − T E | HC | Vid |
|---|---:|---:|
|v|+0.5488 [-4.9203, +6.8732]|+1.4291 [-0.7389, +5.0716]|
|t|+0.0000 [+0.0000, +0.0000]|+0.0000 [+0.0000, +0.0000]|
|s|+2.2659 [-5.4339, +11.8119]|-0.2368 [-14.3569, +14.8464]|
|observed_actual_s|+1.0844 [-9.5197, +13.1794]|+7.0128 [-2.7488, +23.7593]|
|unobserved_actual_s|+2.3927 [-5.0747, +11.7793]|-0.0115 [-13.8772, +15.1502]|

| Absolute complete vIoU (%) | HC | Vid |
|---|---:|---:|
|E_MBR|35.30548|15.75593|
|B_E_MBR|35.85431|17.18508|

## Actual teacher and E transfer

Actual top follows the unchanged branch:100 context calls use the highest target-score candidate after mixed-score NMS/top3;8 phrase-fallback calls use their original mixed-score top1 singleton. It is not a GT oracle. Paired top changes average known shared called positions within query, then average parents with that diagnostic support; this support is narrower than the main16.

| Actual top B − T (IoU pp) | HC | Vid |
|---|---:|---:|
| change |+4.1506 [-5.6939, +15.0753]|+15.4981 [+0.6199, +39.3879]|

### hc2

| Teacher | Called / no call | Accepted | Parents with references | Known called / selected | Wrong admitted IoU<.3 / <.5 |
|---|---:|---:|---:|---:|---:|
|T|64 / 0|53|15|49 / 43|8 / 12|
|B|64 / 0|56|16|49 / 46|8 / 12|

| E transfer on own known selected support | T | B |
|---|---:|---:|
|selected_minus_top|+4.0533 [-7.8103, +16.1454]|-5.6479 [-16.9436, +0.0000]|
|propagated_minus_selected|-4.6253 [-9.6773, -0.6360]|-3.7493 [-6.8208, -1.0057]|

Own-selected supports can differ: these are conditional stage diagnostics, not a paired whole-method comparison. Rejection reason counts and all128 planned slots, signed/unknown frame results and all32 parent outputs are saved in JSON. Empty B pools use Source via zero E correction, never an implicit T fallback.

### vidstg

| Teacher | Called / no call | Accepted | Parents with references | Known called / selected | Wrong admitted IoU<.3 / <.5 |
|---|---:|---:|---:|---:|---:|
|T|44 / 20|42|11|17 / 16|4 / 5|
|B|44 / 20|39|11|17 / 12|1 / 2|

| E transfer on own known selected support | T | B |
|---|---:|---:|
|selected_minus_top|-3.5722 [-34.0907, +23.3742]|-13.6624 [-40.9873, +0.0000]|
|propagated_minus_selected|-1.9533 [-5.0750, +1.0826]|+0.4100 [-2.3024, +3.0170]|

Own-selected supports can differ: these are conditional stage diagnostics, not a paired whole-method comparison. Rejection reason counts and all128 planned slots, signed/unknown frame results and all32 parent outputs are saved in JSON. Empty B pools use Source via zero E correction, never an implicit T fallback.

## Signed cases and cost

| Parent | B E − T E vIoU pp | Unobserved actual sIoU pp | Accepted T / B |
|---|---:|---:|---:|
|NH001|-0.38603|+5.30151|4 / 4|
|NH002|-1.32161|-1.88259|4 / 4|
|NH003|-25.75847|-31.75060|3 / 4|
|NH004|-2.35696|-2.32270|4 / 3|
|NH005|-0.01660|-0.07337|4 / 4|
|NH006|+0.59686|+1.47733|3 / 4|
|NH007|-1.07423|-2.30702|3 / 2|
|NH008|+0.06515|-0.00874|4 / 4|
|NH009|+0.79430|+10.40953|3 / 2|
|NH010|+1.14332|+0.66480|3 / 3|
|NH011|+0.01157|-0.17548|4 / 4|
|NH012|+37.50577|+59.36942|0 / 4|
|NH013|+1.33896|+1.62064|3 / 3|
|NH014|-0.48342|-0.64767|4 / 4|
|NH015|-0.30628|-0.54450|3 / 3|
|NH016|-0.97103|-0.84742|4 / 4|
|NV001|+0.00000|-72.72386|4 / 3|
|NV002|+0.00000|-0.16425|4 / 4|
|NV003|+0.00000|+0.00000|0 / 0|
|NV004|+0.00000|+0.02212|4 / 4|
|NV005|-3.19966|-3.82427|3 / 2|
|NV006|+0.00000|+0.00000|0 / 0|
|NV007|-2.75404|-10.72161|4 / 3|
|NV008|-0.09274|+0.02131|4 / 4|
|NV009|+0.00000|+0.00000|0 / 0|
|NV010|+0.00815|+0.01039|4 / 4|
|NV011|+0.00000|+0.00000|3 / 3|
|NV012|+26.32300|+83.45293|4 / 4|
|NV013|+0.00000|+0.00000|0 / 0|
|NV014|+0.02033|+0.05459|4 / 4|
|NV015|+0.00000|+0.00000|0 / 0|
|NV016|+2.56131|+3.68844|4 / 4|

GPU allocation conservative upper bound 78.9470s includes load, numerical replay and final weight/hash checks plus3s guard. B108 synchronized forwards total 13.6785s; observation pipeline 17.9525s; peak allocated 2.1528GiB. T4 NH001 forward 0.9146s vs B NH001 0.4887s; T peak 1.6907GiB. First forwards include cold-start effects; this four-frame timing is not a full matched benchmark. All-T historical forward time 10.7923s is reported separately, not used to claim a controlled speed ratio. Official HF download/verification 305.14s and old cache acquisition are separate. No student updates or model training.

## Interpretation and limits

The checkpoint replacement changes backbone capacity and pretraining together. Equal thresholds are a fixed protocol, not proof of calibration equivalence; no target-GT calibration or prompt sweep was done. This experiment tests actual B teacher capability and unchanged E transfer, not OPD. Direct readout failure cannot be called OPD failure. Dataset-specific decisions and any remaining potential must be stated from measured contrasts; no B/Direct/Fixed/Live continuation is automatic. All developer exposure remains disclosed.

See RESULTS_PUBLIC.json, FRAME_DIAGNOSTICS_PUBLIC.json, ROOT_INDEPENDENT_AUDIT.json, MODEL_PROVENANCE.json and COST_BREAKDOWN.json.

## Explicit decision and representative mechanisms

Do not replace the current T configuration or start OPD from this result. B has actual localized advantages, especially on the eight Vid parents with known called diagnostic support, but the complete-tube gain is not established at2–3pp and both main paired intervals cross zero. This is not an OPD experiment and cannot establish OPD failure.

- NH012: T admits0/4 despite high-IoU top boxes at several positions; B admits4/4. Full v improves+37.50577pp (3.38430%→40.89007%). This combines improved disambiguation/admission with candidate changes, not just tighter coordinates.
- NH003: B admits4/4 but all four actual tops have IoU≤.109; complete v falls−25.75847pp. A larger checkpoint can reinforce the wrong target under the same text/span and thresholds. High score is not correctness.
- NV012: T actual tops are already near.90 but E chooses zero-IoU alternatives. B selects its actual tops and restores the complete trajectory, +26.32300pp full v. This is not evidence that T lacked the correct object; changed scores/candidate geometry affect the estimator's choice.
- NV011 position21: B actual top IoU.956369 is present and admitted; E selects alternative index2 with IoU0, and output remains0. NH001 position42 likewise top.847178→selected0. These are exact retained-evidence selection losses, not missing teacher observations.
- NV001: fixed temporal overlap is0, so complete v stays0 while all-known unobserved actual s drops−72.72386pp. Preserve this spatial damage; do not hide it by restricting the main actual-node metric to the predicted interval.

Signed full-v counts: HC7 improve/9 worsen; Vid4 improve/3 worsen/9 unchanged. Sensitivity only: omitting NH012 gives the remaining15 HC mean−1.91496pp; omitting NV012 gives the remaining15 Vid mean−.23044pp. Main results keep every parent. The mean positives are concentrated and do not justify an automatic independent confirmation or checkpoint promotion.

The concrete surviving finding is that actual B evidence can be lost at reference selection, while B also creates severe errors. Any later OPD proposal would still need reliable object/reference supervision and matched Direct/Fixed/Live evidence; this completed comparison does not authorize that next experiment. See POST_SEAL_ANALYSIS.json and DECISION.json.
