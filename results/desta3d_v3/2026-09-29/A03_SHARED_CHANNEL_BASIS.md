# A0.3 Shared Channel Basis Feasibility — completed / independently audited

Train128/95 original source-training parents alone estimated the basis; Dev64/16 already exposed diagnosis parents only evaluated it. The cache targets are source-GT native gradient oracles. No learned predictor was fitted, no fresh31/388 or target was read.

## Shared channel structure

Each uncentered covariance is normalized by the full field energy before equal-query averaging. The diagnostic ranks and random seed were fixed before measurement. Per-query-optimal SVD is an upper bound, not the shared basis.

|Rank|Train shared mean energy %|Dev shared mean %|Dev shared median %|Dev optimal mean %|Dev shared/optimal mean %|Dev random mean %|
|---|---:|---:|---:|---:|---:|---:|
|1|40.795371|42.609007|41.867668|51.824762|83.100250|0.303167|
|4|65.503054|69.283949|69.821697|78.766523|88.018777|2.041218|
|8|71.858165|75.062082|76.272116|86.018153|87.185651|4.166372|
|16|78.548167|80.668708|81.323412|91.370581|88.238046|6.621292|
|32|84.757064|85.794654|86.193600|95.182519|90.110142|12.368396|
|64|90.462894|90.528371|90.753104|97.730633|92.620136|23.210195|

The independent Dev rank32 median 0.861935998931 passed the locked .75 resource gate. Full energy/cosine/retention mean, median and ranges for both splits and controls remain in the JSON.

## Conditional native screen

Frozen PTD4B/B1/union256, same exposed Dev64, two new projected oracle arms, zero backwards/optimizer. Each projection was renormalized to the full original coefficient-field norm, mapped through the original union basis, and inserted into both native passes at the original radius .13545580427763146. B1 and Full Oracle were reused after sealed hash plus actual pixel/preprocessing/feature-context identity checks; no redundant baseline native decoding. All256 predictions sealed before scoring.

|Arm|tIoU %|sIoU %|vIoU %|Δt vs B1 pp|Δs pp|Δv pp|
|---|---:|---:|---:|---:|---:|---:|
|B1|41.024269|45.421144|29.723147|+0.000000|+0.000000|+0.000000|
|oracle|50.529483|53.708186|42.840472|+9.505214|+8.287042|+13.117325|
|Shared-R16|50.051387|54.647613|40.661526|+9.027118|+9.226468|+10.938378|
|Shared-R32|50.098197|55.844256|42.254575|+9.073928|+10.423112|+12.531428|

Parent-macro is primary; each parent has four queries so the query means coincide here. sIoU uses the existing scorer's valid GT-frame support. No metric or invalid-output filtering changed.

|Arm|Δv paired parent 95% CI pp|v query harm >5pp|v query gain >5pp|v B1-good retained|t B1-good retained|Full Oracle v-gain retained %|Native gate|
|---|---|---:|---:|---|---|---:|---|
|oracle|[+6.465339, +19.519180]|8|34|13/17|21/24|100.000000|True|
|Shared-R16|[+5.670503, +16.695126]|6|32|14/17|23/24|83.388789|True|
|Shared-R32|[+6.537717, +18.867999]|5|35|13/17|21/24|95.533408|True|

All per-query positive/negative/zero changes, temporal/spatial tails, format failures, invalid geometry, source-parent intervals and both rank-vs-Full comparisons are retained in the anonymous JSON. B1-good means score>.5; continuous damage can remain even when a threshold is retained. Bootstrap intervals are descriptive and unadjusted on this repeatedly exposed panel.

## Decision and limits

Locked route: **fixed_rank16_factorized_predictor_candidate**.
The native gate requires positive mean vIoU, nonnegative tIoU and sIoU, and no registered systematic branch collapse. Rank16 has priority when both pass. This selects only the next candidate; no factorized predictor, extra rank, full training, expert, trust gate or OPD was run.
Shared low-rank oracle feasibility does not establish predictability from ordinary inputs or learned-predictor advantage. This is a source-GT oracle and not a deployable correction. A basis fitted on Train and evaluated on exposed Dev is not fresh confirmation. Neither gain retention nor energy is a safety guarantee; all harm and retention failures remain reported.

## Verification and resources

Four synthetic CPU controls passed. All192 raw cached fields were independently reconstructed in Torch FP64: scalar/summary max 4.88498130835e-15; direct projection cosine/norm max 2.91988655476e-14. Native128 factor fields were independently checked and actual FP32 delta hashes reconstructed; delta error 7.45058059692e-09, norm relative error 2.64793150739e-08.
All256 predictions /768 geometric metrics had scalar-versus-tensor max 4.4408920985e-16; reused scores matched exactly. Independent root 472 parent/CI/tail/retention/gate checks max 0.
The original CPU field auditor failed exact delta hashes because it forced OpenBLAS4 while inference inherited OpenBLAS20. Coefficient hashes all matched; across128 fields,23 had28 FP32 elements different, max7.450580596923828e-9. Isolated audit v2 restored the original20-thread execution only for hash reconstruction: every actual delta hash then matched, with independent Torch checks and tolerances unchanged. Original failed auditor/pins/FAILURE/diagnosis remain; no GPU rerun, prediction/metric change or sample removal. Exact duration of the first failed CPU audit was not captured by its original failure handler.
Corrected-F hashes and full physical replay/frozen parameters are worker observations; full stock F not persisted again. Prior B1/Full Oracle reused after sealed hash and exact current input/context identity, no extra baseline native generation.
Actual native worker 201.902199s + nonoverlapping wrapper 1.427986s; cumulative allocation/wrapper 72772.75626609707s, cap=null. CPU primary 12.951026s and independent 4.913663s counted separately; remaining CPU audit times in JSON. Worker exited; free disk 129229410304 bytes above8GiB. No scientific artifacts removed.
