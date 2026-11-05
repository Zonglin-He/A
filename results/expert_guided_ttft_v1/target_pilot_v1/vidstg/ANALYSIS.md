# ROOT target-pilot interpretation

vidstg: original128 queries/128 distinct parents. Both complete source-selected TTFT prediction and saved-state/official/manual/input/SHA/means audits passed.

| Method | tIoU | vIoU | R.3 | R.5 | Native unobserved IoU |
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|36.412936|
|SourceTR|35.528603|14.548333|17.968750|7.812500|36.412936|
|TTFT|35.087349|16.094270|19.531250|10.937500|41.417636|
|Current|35.202435|15.339259|19.531250|10.156250|38.207575|

TTFT-Current: vIoU +0.755011 pp, pointwise10000 parent-bootstrap95%CI [-0.03454370508299148, 1.5724415984133873]. Full signed outcomes and short/long groups retained.

TTFT-SourceTR: vIoU +1.545937 pp, pointwise10000 parent-bootstrap95%CI [-0.4057675146330762, 3.525075490711594]. Full signed outcomes and short/long groups retained.

TTFT-Source: vIoU +2.771367 pp, pointwise10000 parent-bootstrap95%CI [0.7857079075336811, 4.836698901845389]. Full signed outcomes and short/long groups retained.

Unobserved IoU uses GT-supported native sampled unobserved frames over the video, equal query means; not the conditional dense event metric from the candidate-decode experiment.

Fixed128 exposed parent panel, not full target lists or unknown confirmation.
Configs selected solely on source validation36 per direction before target predictions.
SourceTR has no expert; its contrast includes added expert information and parameter updates, not a same-information adaptation attribution.
Current Full and TTFT use same8raw expert observations, different complete inference/optimization pipelines and own refined intervals. Not isolated propagation comparison.
Original Current soft E/expert anchor choice unchanged; not validation of an intended hard-anchor variant.
No complete matched deployment timing for TTFT measured.
