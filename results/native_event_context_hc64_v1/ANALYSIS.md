# Overlapping native temporal contexts: HC64

Fixed, exposed development experiment. Full remains paused; no method promotion
from one panel or selection of a favorable control. Full caption and all original
sampled feature values remain unchanged. Two contexts are prefix indices0..48 and
suffix16..64 on the existing65 sampled nodes, each retaining both original offsets.
No image decoding, visual/text encoder, expert call, extra observation or training.
Original crossmodal fusion and native two-pass temporal initialization/decoder run
for each context. This differs from failed full/static, masked-text and temporally
averaged-feature ratios: the evidence comes from actual contiguous feature sequences.

Each cropped posterior is calibrated against its own uninformative zero-endpoint
logit two-offset hull. A hull of uniform endpoint scores is NOT uniform over intervals.
Strict interior start/end endpoints use joint evidence if both visible; if only one
is visible, the other is marginalized. Crop-edge/unavailable endpoints are censored,
not clipped into a negative observation. Mean Prefix/Suffix log evidence reweights
the native posterior, then uses original physical MBR and unchanged complete-box DP.
This is a fixed inference feasibility test, not a calibrated independence claim.

Controls isolate ordinary sharpening, same-information full hull calibration,
reversed context placement and exact uninformative-window no-op. Single-context
arms are retained as diagnostics, not post-hoc candidates.

All metrics percent; improved/harmed/tied time outcomes relative toNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|CensoredContext|26.2446|49.1264|14/44/6|33|
|PrefixOnly|25.3629|47.6026|15/46/3|34|
|SuffixOnly|26.3346|49.9564|17/43/4|33|
|HullCalibratedFull|24.3617|45.4755|6/55/3|38|
|Sharpen|28.7258|53.7518|16/38/10|20|
|ReverseContext|25.1588|45.7753|20/40/4|31|
|UninformativeContext|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−NativeG6: v-3.4506pp CI[-5.656005220264935, -1.5186200554154905], t-6.8349pp CI[-10.238763915195843, -3.7756861337023273]. Primary−Source: v-0.6492pp, t-5.6966pp. Primary−same-information HullCalibratedFull: t+3.6509pp. Shortt-2.4362pp, longt-7.9562pp versusNativeG6.

Same-duration/near-native-prior matched candidate feedback macro accuracy53.9611% CI[0.51740820196327, 0.5613550584739034];721,976 pairs, native52.1362%. These are offline diagnostics after all predictions sealed, not online selections.

All64 Full hidden/posterior and original Source/NativeG6 outputs restore bitwise; uninformative context is an exact no-op. Independent official/DenseTube checks1152, maxerror8.88e-16. All48+434 bound sources unchanged. Actual worker46.09s including source loading, calls{'fusion_offsets': 384, 'native_replay_queries': 192, 'new_expert': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_RGB': 0}; no end-to-end deployment speed claim.

Primary does not pass the predeclared broad improvement screen. Preserve all outcomes; do not scan context weights or substitute the best control as a new method.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).

Context/boundary work motivates the question but uses learned models: [MS-2D-TAN](https://arxiv.org/abs/2012.02646), [CBP](https://ojs.aaai.org/index.php/AAAI/article/view/6897). Neither validates this frozen window likelihood rule.
