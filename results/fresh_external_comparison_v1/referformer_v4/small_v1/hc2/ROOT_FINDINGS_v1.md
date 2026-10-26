# hc2: single-expert replacement, fixed128 parents

All128 predictions sealed, actual GPU exited, independent full raw-output/input/selected-single-query/mask/geometry/Epool/posterior/own-MAP/final-DP repeated-cold audit, official CPU score and independent complete-array/means audit passed. This is an exposed fixed cohort, not an unseen confirmation or full12205 result.

|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|
|CurrentSimple|46.509478|26.879620|45.312500|15.625000|
|CurrentOurs|54.968548|31.791298|50.781250|22.656250|
|AlternativeSimple|46.509478|23.073061|35.937500|13.281250|
|AlternativeOurs|54.769180|27.552349|43.750000|17.187500|

Single external expert per branch: Current uses Source+ReferDINO; Alternative uses Source+ReferFormer. ReferDINO never supplies alternative prompts/masks/E/prototypes/posterior/final event. Separately reconstructed current arrays are baseline references only; their names cannot overwrite AlternativeOurs/Simple. Both simple fusions use Source native event, while each structured inference uses its own expert-dependent posterior/event/finalDP. These comparisons do not equalize architectures, model knowledge, precision/resolution or FLOPs.

- CurrentOurs-CurrentSimple: t+8.459071pp CI[5.507059352429634, 11.49512358151015]; v+4.911678pp CI[2.908269145459872, 6.924963412010876].
- AlternativeOurs-AlternativeSimple: t+8.259703pp CI[5.28065135225317, 11.34186485869759]; v+4.479288pp CI[2.7207257445490267, 6.30091846500522].
- AlternativeSimple-CurrentSimple: t+0.000000pp CI[0.0, 0.0]; v-3.806559pp CI[-6.448386173695205, -1.3980477146137238].
- AlternativeOurs-CurrentOurs: t-0.199368pp CI[-0.9209644041929725, 0.5363202650027828]; v-4.238950pp CI[-6.94380071135964, -1.6846865614959465].

The between-expert gain-difference CI, severe t/v union-once/rescues and all signed short/long strata are retained in accompanying JSON/TSVs. Neither a positive within-expert gain nor a CI crossing zero proves expert independence. No checkpoint/prompt/parameter is selected from these scores.

Actual128 expert request sum23.489117s, conditional query sum348.090787s, selected structured finish sum25.543325s, baseline compatibility sum29.604352s; peak allocated query GPU4553138688bytes. These nested components must not be added. Source/current-expert acquisition is inherited nonzero and shared; this cached-input query work is not complete uncached deployment latency. The two actual runtime model loads are separate PID/start records counted once globally. One earlier successful forward was lost before checkpoint due to absent save helper and remains an extra paid forward with missing forward-only clock; its model/whole-worker clock is retained, never zero or claimed raw-SHA restored.

Source-prompt-only EdgeTAM is a separate upcoming branch; old ReferDINO-plus-EdgeTAM is an additional-model extension only, not an equal-expert mechanism comparator. CG/gamma/corruption remain user-paused, additional CG-HC cancelled. Whole current authorization remains active.
