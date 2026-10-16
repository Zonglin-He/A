# Bounded instance-evidence study: complete measurements, no method promotion

The requested sequence is complete on the native TA-STVG route: exact old candidate-pool12 review, matched complete Temporal clock repair32, one prospective candidate modification with E0/E1/E2 on32 development sources, and one frozen64-source confirmation. Current32 and fixed64 parents are disjoint,96 unique parents total; all have historical project exposure. The12 diagnostic sources are inside32, not12 additional independent test sources. No official fullset, Bootstrap or CURRENT promotion occurs.

## What was tested

Temporal stays512-parameter Temporal-Lite L-BFGS with original8x448 frames/prompts and the corrected fractional-time interface. Spatial stays episodic1792 with HC lr .0045/Vid .027. The sole evidence change is Source-seeded full foreground from three-iteration GrabCut and existing five-frame local DIS; it replaces one last-slot proposal at the same nine-candidate budget. No new detector, tracker, Teacher or hyperparameter search is added.

E0 uses the old candidate pool/current response-guided updater; E1 uses the new candidate/current updater; E2 uses exactly E1 references with original validated OPD. E1-E2 is a whole updater comparison, not a projection-only ablation or a comparison to a separately tuned DINO/online OPD method. LN does not carry across queries. Source state is restored and all204 actual fit outputs receive real complete-model final readout.

## Main evidence

| Fixed64 method | HC mean vIoU (%) | VidSTG mean vIoU (%) |
|---|---:|---:|
| Frozen |22.998670|9.910852|
| Corrected Temporal |26.420683|11.359017|
| E0 old evidence |26.253638|10.901340|
| E1 new evidence |26.331687|11.260686|
| E2 new evidence/original OPD |26.321074|11.335173|

E1-E0 remains positive:+.078/+ .359pp, compared with+.141/+.289pp in development32. However, E1-Temporal is-.089/-.098pp in fixed64; E1-E2 is+.011/-.074pp. The joint model still improves Frozen by+3.333/+1.350pp, mainly through Temporal. This does not establish an additional beneficial Spatial branch or a stable independent updater advantage. Vid's10 positive/3 negative/19 unchanged E1-versus-Temporal sources show why signs alone are insufficient: a few large harms outweigh several small improvements. All positive, negative, neutral and unknown cases remain public.

A narrow interpretation is essential: adding a good nonSource hypothesis is not the same as discovering a new object in the complete pool. In development32 the full-pool IoU .5 coverage including Source changes31→31 of43 known HC windows and10→11 of26 known Vid windows, while nonSource-only coverage changes much more. In fixed64 it changes63→66 of76 HC windows and37→37 of65 Vid windows. Source already supplies many of the usable locations; the new hypothesis is Source-conditioned pixel geometry, not independent referent discovery.

On actual adopted fixed64 anchors with exact GT, E1 reference-minus-Source box IoU averages-9.406pp in HC(24 known/20 unknown) and-16.014pp in Vid(33 known/34 unknown). These are offline observation-level geometric diagnostics, not full-Tube scores or semantic identity labels. Both updater variants still learn from imperfect references. A Source-box-reference control was not part of the three-arm plan and is not silently added; the current result does not isolate pixel geometry from making a Source-like reference eligible.

The inherited main supervised/unsupervised metric names refer to the common four observation positions and all other known GT, including off-grid frames. Separate actual-gradient-support results use each updater's positive-weight training anchors and matched Temporal support. Fixed64 E1 changes on the unsupervised complement are-.177pp HC and-.344pp Vid, with all32 sources known per domain. Actual-anchor changes are-.781pp on13 known HC sources and-.452pp on16 known Vid sources; the remaining19/16 stay unknown. No sparse-GT null is replaced by zero.

## Decision and boundaries

The Source-conditioned foreground modification reduces harm relative to the old evidence system, but has not confirmed a positive additional Spatial contribution. This particular recipe is not promoted or expanded to official fullset or empirical second-backbone efficacy evaluation. Temporal repair and the1792 OPD interface are retained; this finite negative finding does not reject all Spatial branches, pixel evidence or other backbones. No confirmation-driven recipe switch or rescue parameter sweep occurs.

The two local TubeDETR checkpoints have query dimension256 and last-layer LN1536, established only by a CPU state-dict inventory. Existing replay/reinsertion still uses TA-specific ground-decoder/two-offset routing. No empirical second-backbone adaptation result is claimed, and unavailable weights are not used as an excuse: the native recipe failed to establish additional Spatial value before that expansion.

Two material CPU engineering issues are preserved with their original logs/source/outputs: float32 saved-box conversion versus the original float64 readout, and a missing copy of an already locked development-rule citation at final confirmation binding. Separate readers/metadata recovery preserve scientific expressions and thresholds. Zero GPU fits are repeated for either issue. Source optional mask tensor metadata may be absent/stale after prospective replacement; actual mask archives/rendered NPY bindings and geometry readback remain available. The complete raw/public source bindings and failures are recorded in the stage reports.

The independent final audit reaggregates576 primary arm rows/2880 metric values with186 nulls, verifies anonymous exports, source uniqueness and six existing actual HTTP publication receipts. It uses saved GT-derived scores, reads no new raw annotations and does not claim an independent complete backward pass. There are204 actual Spatial fits/readouts,84 new Temporal fits,284 new Temporal caption/alignment pairs plus52 exact corrected reuses, and768 new Spatial WHO requests. Confirmation measured worker stages sum2050.99s, including the failed final CPU stage and its recovery, excluding handoff gaps and inherited cache generation; this is not a fullset ETA.

## Incremental publications

- Whole-pool diagnosis:76646d08, docs/STVG_INSTANCE_POOL12_001.md.
- Matched Temporal repair:a090050a, docs/STVG_INSTANCE_TEMPORAL32_001.md.
- Development E0/E1/E2:ab789f3d, docs/STVG_INSTANCE_EVIDENCE32_001.md.
- Actual gradient support:bbccc183, docs/STVG_INSTANCE_ACTUAL_SUPPORT32_001.md.
- Source-inclusive interpretation correction:fff55dfb, docs/STVG_INSTANCE_POOL_INTERPRETATION001.md.
- Frozen64 full results:b151fc22, docs/STVG_INSTANCE_CONFIRMATION64_001.md.

Each group is published when complete, not held until this closing note. Actual remote bytes/SHA256/Gitblob/mode/tree/ref verification receipts are retained. No media, raw annotation coordinates, weights, tensor caches, real Query-caption/raw Teacher responses, personal conversations or credentials are published.
