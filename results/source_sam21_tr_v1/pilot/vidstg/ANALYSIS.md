# pilot vidstg: Source-prompt SAM2.1 Hiera-L + Source-box TR

Whole 128 queries/128parents sealed, independent input/mask audit, official two-interval scores and separate manual-score/SHA/means/parentCI audit complete. No new Source/ReferDINO/optimizer. Source/SourceTR/Current scalar references reused exactly.

|Method|tIoU|vIoU|R.3|R.5|Native unobserved IoU|
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|36.412936|
|SourceTR|35.528603|14.548333|17.968750|7.812500|36.412936|
|SAM2Native|32.121535|15.808445|20.312500|8.593750|41.851595|
|SAM2TR|35.528603|17.032514|23.437500|13.281250|41.851595|
|Current|35.202435|15.339259|19.531250|10.156250|38.207575|

SAM2 receives only original Source boxes at the original eight physical positions and all native sampled RGB frames. All native mask boxes are used, including prompted frames, without Source writeback. SourceTR event is inherited from expert-free original Source-box evidence. No ReferDINO/E/Current time/DP enters SAM2. One external model per pipeline is controlled, not evidence identity, frame-forward count, model knowledge or precision. Official BF16 autocast/TF32,1024 resolution; no target tuning.

All signed negative/severe/rescue and short-long paired CIs are retained. Experimental query clocks include checkpoint/hash/audit provenance and cannot be compared directly with matched deployment timings.

SAM2TR-Source: delta vIoU +3.709611 pp, pointwise95%CI [2.195384891614257, 5.468737209995248].
SAM2TR-SourceTR: delta vIoU +2.484181 pp, pointwise95%CI [1.4948086661905764, 3.5547907672261507].
Current-SAM2TR: delta vIoU -1.693255 pp, pointwise95%CI [-3.9636379878485513, 0.521166022542245].
SAM2TR-SAM2Native: delta vIoU +1.224069 pp, pointwise95%CI [-0.03106265916528173, 2.668583750329489].
