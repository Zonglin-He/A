# pilot hc2: Source-prompt SAM2.1 Hiera-L + Source-box TR

Whole 128 queries/128parents sealed, independent input/mask audit, official two-interval scores and separate manual-score/SHA/means/parentCI audit complete. No new Source/ReferDINO/optimizer. Source/SourceTR/Current scalar references reused exactly.

|Method|tIoU|vIoU|R.3|R.5|Native unobserved IoU|
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|53.067325|
|SourceTR|54.833053|30.060976|50.000000|24.218750|53.067325|
|SAM2Native|46.509478|25.383772|41.406250|17.187500|50.287104|
|SAM2TR|54.833053|29.917082|46.093750|25.000000|50.287104|
|Current|54.968548|31.791298|50.781250|22.656250|54.637004|

SAM2 receives only original Source boxes at the original eight physical positions and all native sampled RGB frames. All native mask boxes are used, including prompted frames, without Source writeback. SourceTR event is inherited from expert-free original Source-box evidence. No ReferDINO/E/Current time/DP enters SAM2. One external model per pipeline is controlled, not evidence identity, frame-forward count, model knowledge or precision. Official BF16 autocast/TF32,1024 resolution; no target tuning.

All signed negative/severe/rescue and short-long paired CIs are retained. Experimental query clocks include checkpoint/hash/audit provenance and cannot be compared directly with matched deployment timings.

SAM2TR-Source: delta vIoU +4.792829 pp, pointwise95%CI [2.8001283829253003, 6.880613659999527].
SAM2TR-SourceTR: delta vIoU -0.143894 pp, pointwise95%CI [-1.114860479704206, 0.8274487948376124].
Current-SAM2TR: delta vIoU +1.874217 pp, pointwise95%CI [-0.7423495170406896, 4.477903517688564].
SAM2TR-SAM2Native: delta vIoU +4.533309 pp, pointwise95%CI [2.75720584341349, 6.427829233194489].
