# Literal official last-head baseline, aggregate scalar table

Released union checkpoint, B=1, deterministic sampling, available official VAL. GT-class-assisted eligible span, native yaw0 predictions and official dimension clamp. Official original float32 geometry is reported below; the all256 oracle columns are explicitly from audited float64 geometry. No adaptation or retraining.

| Platform / mode | Eligible n / scenes | Literal Top1 mIoU % | Acc25/50 Top1 % | Acc25/50 Top5 % | Acc25/50 Top10 % | Audited64 oracle Acc25/50 % | Audited64 oracle mIoU % |
|---|---:|---:|---:|---:|---:|---:|---:|
| waymo / BBF | 3782 / 93 | 46.550333 | 80.3279 / 49.2597 | 93.4955 / 59.7567 | 95.6372 / 61.4754 | 99.2332 / 63.7758 | 59.526461 |
| waymo / BBS | 3782 / 93 | 46.338959 | 79.9048 / 49.2068 | 93.0460 / 59.5717 | 95.1348 / 61.1846 | 99.2332 / 63.7758 | 59.526461 |
| drone / BBF | 2984 / 2 | 21.931691 | 52.7145 / 9.8525 | 71.5483 / 13.5054 | 78.0831 / 14.7788 | 94.9732 / 19.2024 | 37.161910 |
| drone / BBS | 2984 / 2 | 21.956872 | 52.5804 / 9.8190 | 71.3472 / 13.4718 | 78.4517 / 14.6448 | 94.9732 / 19.2024 | 37.161910 |
| quad / BBF | 2928 / 3 | 27.915245 | 52.1516 / 24.2145 | 78.0055 / 28.3128 | 87.1243 / 29.2350 | 93.5451 / 29.7131 | 39.984957 |
| quad / BBS | 2928 / 3 | 27.949337 | 52.3907 / 24.2145 | 78.0738 / 28.4153 | 87.0219 / 29.2691 | 93.5451 / 29.7131 | 39.984957 |

Waymo has97 listed VAL scenes but93 on-disk scenes:4 listed scene directories are absent. Its3,794 forwarded raw objects include12 objects excluded by the official class-synonym criterion, leaving3,782 eligible. Drone2/2 and Quad3/3 listed scenes are available, with2,984 and2,928 eligible objects. Empty-map counts arezero. This is not a claim of complete97/97 Waymo coverage.

Literal-vs-audited checks cover native Top10 and legacy seven-head Top1, not every candidate under the literal routine. Their maximum pairwise absolute IoU difference is0.0001144944, withzero threshold flips at.25/.5, zero swallowed exceptions andzero nonfinite outputs in that coverage. Root float64 oracle covers all targets/candidates; independent3D oracle validation covers the predeclared64 parents/platform only. All result sets share the same frozen predictions.

GT fields beyond the official first7 are unnamed extras and unused. Multiple targets/frames/heads are not independent scenes; Drone/Quad contain only2/3 scenes. The full-scoring transaction follows all-source sealing, while smoke48 and earlier research labels were already exposed.
