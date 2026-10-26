# vidstg: single-expert replacement, fixed128 parents

All128 predictions sealed, actual GPU exited, independent full raw-output/input/selected-single-query/mask/geometry/Epool/posterior/own-MAP/final-DP repeated-cold audit, official CPU score and independent complete-array/means audit passed. This is an exposed fixed cohort, not an unseen confirmation or full12205 result.

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|
|CurrentSimple|32.121535|13.867903|17.968750|7.812500|
|CurrentOurs|35.202435|15.339259|19.531250|10.156250|
|AlternativeSimple|32.121535|14.299784|18.750000|7.031250|
|AlternativeOurs|34.603561|15.639079|20.312500|10.156250|

Single external expert per branch: Current uses Source+ReferDINO; Alternative uses Source+ReferFormer. ReferDINO never supplies alternative prompts/masks/E/prototypes/posterior/final event. Separately reconstructed current arrays are baseline references only; their names cannot overwrite AlternativeOurs/Simple. Both simple fusions use Source native event, while each structured inference uses its own expert-dependent posterior/event/finalDP. These comparisons do not equalize architectures, model knowledge, precision/resolution or FLOPs.

- CurrentOurs-CurrentSimple: t+3.080900pp CI[0.7959085296291982, 5.611244478097894]; v+1.471356pp CI[0.2315213952099042, 2.817800827449867].
- AlternativeOurs-AlternativeSimple: t+2.482026pp CI[0.2603290301774109, 4.88914494958528]; v+1.339296pp CI[0.28850440640616, 2.469618644902753].
- AlternativeSimple-CurrentSimple: t+0.000000pp CI[0.0, 0.0]; v+0.431881pp CI[-1.5675672921765762, 2.445672024298217].
- AlternativeOurs-CurrentOurs: t-0.598874pp CI[-1.6325655872790064, 0.29247820631324334]; v+0.299820pp CI[-1.9986399694270065, 2.604791396154297].

The between-expert gain-difference CI, severe t/v union-once/rescues and all signed short/long strata are retained in accompanying JSON/TSVs. Neither a positive within-expert gain nor a CI crossing zero proves expert independence. No checkpoint/prompt/parameter is selected from these scores.

Actual128 expert request sum20.115300s, conditional query sum226.853000s, selected structured finish sum37.543711s, baseline compatibility sum37.817848s; peak allocated query GPU3531716608bytes. These nested components must not be added. Source/current-expert acquisition is inherited nonzero and shared; this cached-input query work is not complete uncached deployment latency. The two actual runtime model loads are separate PID/start records counted once globally. One earlier successful forward was lost before checkpoint due to absent save helper and remains an extra paid forward with missing forward-only clock; its model/whole-worker clock is retained, never zero or claimed raw-SHA restored.

Source-prompt-only EdgeTAM is a separate upcoming branch; old ReferDINO-plus-EdgeTAM is an additional-model extension only, not an equal-expert mechanism comparator. CG/gamma/corruption remain user-paused, additional CG-HC cancelled. Whole current authorization remains active.
