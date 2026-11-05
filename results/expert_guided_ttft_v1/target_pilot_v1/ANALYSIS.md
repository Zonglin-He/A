# Expert-guided test-time fine-tuning: fixed128 target panels

Both domains completed sealed predictions, independent GPU replay of all256 adapted weights, official scoring and separate CPU input/feature/interval/manual-score/means audit. ROOT independently reconstructed all absolute means and192 grouped paired fields/CIs. Configurations were selected only on source-domain validation before target prediction.

| Domain | Source vIoU | SourceTR vIoU | TTFT vIoU | Current Full vIoU | TTFT-Full pp (95%CI) |
|---|---:|---:|---:|---:|---|
|hc2|25.124253|30.060976|33.979095|31.791298|+2.187797 [0.4652124008995947, 4.0024197184022094]|
|vidstg|13.322903|14.548333|16.094270|15.339259|+0.755011 [-0.03454370508299148, 1.5724415984133873]|

TTFT has higher HC vIoU with a positive pointwise CI relative to Current Full. VidSTG mean is higher but its CI crosses zero. These outcomes do not support claiming Current Full is more accurate than this baseline in both panels. Full target lists and actual matched deployment timing remain unfinished; do not extrapolate pilot ranks or relabel experimental query wall as deployment latency.

The44.263877-minute VidSTG bridge wait was a ROOT handoff delay, not model computation. It remains in raw stage wall and is explicitly separated in cost reporting.

All data/source calibration/target code and positive/negative summaries in this stage are currently LOCAL; no new GitHub publication receipt exists. Original published evidence remains unchanged.
