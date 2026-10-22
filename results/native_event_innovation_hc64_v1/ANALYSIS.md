# Context-induced information versus unchanged full-context logits

Isolated fixed follow-up on exposed HC64; spatial method unchanged and full paused.
The previous context rule lowered tIoU6.8349pp; native mean interval8.648s became
7.293s. Same-information full hull calibration was even worse. RC059 lost68.90pp
tIoU when its end moved320frames earlier; RC056 lost46.05pp with inward endpoints.
This motivates separating actual context changes from the confidence already in
original logits, rather than selecting another likelihood exponent.

For the same two fixed windows, take all six original full-context endpoint logits,
slice at exactly the same physical nodes/parity and construct the same native hull
posterior bW. This uses no model call. Compare actual cropped-context qW with bW,
using the same strictly-interior joint/one-boundary marginal censoring. Primary
q0*exp(mean log(qW/bW)) changes only on context-induced prediction differences.
If cropped-context logits equal the original sliced logits, feedback is exactly zero.

SameInformationContext uses bW/uW with the original structural null and isolates
reusing original confidence. Original context/sharpening/reverse/no-innovation
controls are retained. The main coefficient/windows/MBR/space are fixed, no tuning.
This ratio is not assumed calibrated for an event crossing a crop boundary.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|ContextInnovation|29.4482|55.8017|26/25/13|9|
|SameInformationContext|25.2223|46.7187|9/51/4|39|
|CensoredContext|26.2446|49.1264|14/44/6|33|
|Sharpen|28.7258|53.7518|16/38/10|20|
|ReverseInnovation|28.8754|54.0980|18/31/15|14|
|NoInnovation|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−NativeG6: v-0.2470pp CI[-1.3673886822986836, 0.7530458851678318], t-0.1596pp CI[-1.953137429034035, 1.5334918061635696]. Primary−Source: v+2.5544pp, t+0.9787pp. Primary−SameInformationContext: t+9.0830pp. Shortt-1.1780pp, longt+0.1000pp versusNativeG6.

Matched same-duration/near-native-prior candidate feedback accuracy53.4040% CI[0.5046395644939811, 0.5621722500860818], native52.1362%;721,976pairs. All predictions sealed before scoring; all signed outcomes and severe tails retained.

Primary fails basic broad-improvement screen. Do not promote, scan strength or substitute a control.

All64 NativeG6/Source and reused parent arms bitwise; no-innovation exact no-op64. Official/dense checks1024, maxerror8.88e-16. Original48+434 sources intact. Worker4.90s, entirely CPU cached outputs;0RGB/encoder/decoder/expert/optimizer calls.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).
