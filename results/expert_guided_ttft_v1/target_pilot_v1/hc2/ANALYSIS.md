# ROOT target-pilot interpretation

hc2: original128 queries/128 distinct parents. Both complete source-selected TTFT prediction and saved-state/official/manual/input/SHA/means audits passed.

| Method | tIoU | vIoU | R.3 | R.5 | Native unobserved IoU |
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|53.067325|
|SourceTR|54.833053|30.060976|50.000000|24.218750|53.067325|
|TTFT|54.076423|33.979095|53.125000|31.250000|60.666883|
|Current|54.968548|31.791298|50.781250|22.656250|54.637004|

TTFT-Current: vIoU +2.187797 pp, pointwise10000 parent-bootstrap95%CI [0.4652124008995947, 4.0024197184022094]. Full signed outcomes and short/long groups retained.

TTFT-SourceTR: vIoU +3.918119 pp, pointwise10000 parent-bootstrap95%CI [2.5127457781064253, 5.431133860174175]. Full signed outcomes and short/long groups retained.

TTFT-Source: vIoU +8.854842 pp, pointwise10000 parent-bootstrap95%CI [6.554425437872569, 11.17996310183259]. Full signed outcomes and short/long groups retained.

Unobserved IoU uses GT-supported native sampled unobserved frames over the video, equal query means; not the conditional dense event metric from the candidate-decode experiment.

Fixed128 exposed parent panel, not full target lists or unknown confirmation.
Configs selected solely on source validation36 per direction before target predictions.
SourceTR has no expert; its contrast includes added expert information and parameter updates, not a same-information adaptation attribution.
Current Full and TTFT use same8raw expert observations, different complete inference/optimization pipelines and own refined intervals. Not isolated propagation comparison.
Original Current soft E/expert anchor choice unchanged; not validation of an intended hard-anchor variant.
No complete matched deployment timing for TTFT measured.
