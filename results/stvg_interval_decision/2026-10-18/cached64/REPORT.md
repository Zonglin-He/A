# Cached64 temporal decision audit

Original32 exposed parents/domain; fixed Source boxes, original two offsets and physical envelope. Exact expected-tIoU MBR has no Actionness, extra model, threshold or parameter update. Historical Native/projection are replayed exactly.

| Domain | Arm (fused) | vIoU % | tIoU % | Δv vs Native pp |
|---|---|---:|---:|---:|
| hc2 | Native | 22.662572 | 43.574672 | +0.000000 |
| hc2 | projection | 26.417434 | 52.823955 | +3.754861 |
| hc2 | projection_length_native_location | 26.465934 | 52.870719 | +3.803361 |
| hc2 | shuffled_projection | 21.933782 | 42.491792 | -0.728790 |
| hc2 | native_MBR | 26.144050 | 51.378361 | +3.481478 |
| vidstg | Native | 12.742288 | 34.547397 | +0.000000 |
| vidstg | projection | 14.376637 | 38.454197 | +1.634349 |
| vidstg | projection_length_native_location | 14.418102 | 38.553177 | +1.675814 |
| vidstg | shuffled_projection | 15.640817 | 40.868809 | +2.898529 |
| vidstg | native_MBR | 13.501398 | 36.390281 | +0.759110 |

Per-offset metrics, all signed source differences and paired parent bootstrap intervals are in RESULTS. Length is fixed exactly in physical frames before choosing location; this is a diagnostic, not an Actionness-free deployment variant because its length comes from projection. Shuffled values retain their multiset; one fixed permutation is a limited intervention, not a complete null distribution.

MBR optimizes expected physical interval IoU under the Native model posterior, not unknown true IoU. Miscalibration or missing semantics can remain. Its decision rule is standard expected utility (https://aclanthology.org/2022.emnlp-main.754/), not claimed as new theory. No posterior-mass coverage threshold or GT-based choice is used.

Production registries and prior negative findings stay unchanged.
