# One instance-evidence experiment, corrected time interface

Research only. CURRENT, official full evaluation, Bootstrap, optimizer/prompt/LR searches remain unchanged. Original TA-STVG source checkpoints and frozen STVG-R1-7B remain fixed. HC lr .0045 and Vid .027; Temporal-Lite shared-candidate/no-parameter-L2/full-grid-trust512 and episodic Spatial1792 interfaces stay fixed.

## Locked sources and diagnostics

Development32 contains the already reviewed diagnostic12 plus ten additional parent/query-hash ordered development sources per domain. All are historically exposed development, not independent confirmation. The diagnostic12 remain sign-stratified. Existing confirmation64 parents do not overlap these32; their prior project/result exposure must still be disclosed if used again.

Inspect the exact cached original 48 windows and all displayed Source/nonSource candidates. Verify candidate JSON/masks/RGB/video/ID bindings, then read GT only for offline geometric diagnosis. No historical regeneration is needed. Review all actual private scene-and-candidate panels; ambiguous identities stay unknown. Geometric IoU >= .5 is a descriptive coverage statistic, not proof of instance identity or an online decision rule. All missing GT positions remain unknown.

## Corrected Temporal baseline first

Use the exact original uniform8x448 marked videos, original candidate intervals, caption/alignment prompts, 160-token generation cap and 512-param Temporal-Lite objective. Apply the isolated fractional-seconds correction. First strictly replay the legacy Lite fit in its original inherited BLAS context; use the same context for the new fit and independent CPU replay. Reuse a corrected P0-A response only if physical frames, marked RGB, FPS, source, own Query, complete prompt templates, visual size, model/clock/cap all match. Otherwise issue a new call; never relabel a legacy response as corrected. Preserve original no-four-candidate fallback.

Seal the32 new temporal predictions and exact root replay before task scoring. Score Frozen, legacy Temporal-Lite and corrected Temporal-Lite using identical source geometry. This is interface repair, not a new algorithm or selection over variants. Downstream E0/E1/E2 all share this fixed corrected Temporal output and the same resulting four physical observation centers. Both Temporal and Spatial newly invoked experts use the corrected time version; physical FPS is not stretched.

## Exactly one proposal modification

The full-pool diagnostic has25 exact-GT windows, zero nonSource boxes with IoU >= .5 and seven Source boxes with IoU >= .5. Visual review nevertheless finds a few plausible correct identities at low geometric IoU, so do not declare all nonSource identities wrong. Frequent backgrounds/parts and lack of full-object geometric support motivate a candidate-generation test, not another Full-scene or Pairwise rejection prompt.

Test one **Source-seeded foreground hypothesis**. Source supplies only a generic normalized spatial prior. On the current center RGB resized by the original320 rule: expand its clipped box by1.25; use probable foreground inside Source, probable background in its surrounding support, sure background outside the expanded support, and a central half-size ellipse as sure foreground. Run exactly three original GrabCut iterations. Preserve the whole resulting foreground, without maximum-component removal. Track only this five-frame local hypothesis with the already used DIS/forward-backward routine. Preserve all masks/intermediates privately. No GT or new model is used to construct it.

Keep total candidate budget at most nine. Source remains ID1; retain old IDs2..8; replace the final lowest-ranked old proposal ID9 with the new hypothesis. If fewer candidates exist, append one up to nine. If the new mask is invalid under original minimum32-pixel/positive-area requirements, retain the original pool and record the failure. No search over expansions/seeds/iterations is allowed. This can remove a useful old proposal and cannot reliably recover a referent outside a wrong Source prior; record both limitations. New geometry may remain close to or equal to Source, which is not independent object discovery.

E0 = original proposals, current response-guided Spatial Lite. E1 = new hypothesis pool, exactly the same response-guided Spatial Lite. E2 = exactly E1 teacher references, original validated OPD with identical final parameters. Presentation, original ID-selection prompt, one call per observation window, source reset, noise key and update budget remain matched. No extra verification fields or repeated questioning. Corrected E0 teacher output may be reused only under exact input/model/protocol signature equality; E1/E2 share one teacher result by construction.

If corrected Temporal changes centers, reuse sealed old proposal/mask windows only at exactly identical physical centers with source/pixel bindings. Generate missing centers once as a prospective new input version and seal them. Do not claim exact recovery of unrecorded historical intermediates. E0/E1 share those same prospective base candidates.

## Complete task comparison and confirmation

Run real full Student re-decodes, not replacement teacher boxes. Independently audit original rollout/loss/Adam/response math and actual final boxes. Seal all32 E0/E1/E2 predictions before GT scoring. Report Frozen, corrected Temporal, E0, E1, E2; teacher choices, no-reference/no-update cases, geometry coverage, actual nonzero accepted updates, full Tube, supervised/unsupervised changes, source signs and real cost. E1-E0 tests evidence; E1-E2 tests the updater. Do not attribute evidence gains to projection.

Choose at most one shared spatial recipe after development. If no evidence-qualified improvement emerges, record that outcome instead of forcing confirmation. If a recipe qualifies, freeze all settings and evaluate disjoint fixed sources; report prior exposure. Cross-backbone execution needs an actually available compatible source/checkpoint/readout interface and must be separately qualified; a generic box/RGB API alone is not an empirical second-backbone result. No automatic new backbone download or official full-set launch.

Publish each completed meaningful group promptly with anonymous full positive/negative/neutral/unknown rows, code/config/protocol, costs and engineering recovery. Exclude media/GT coordinates/weights/tensors/actual Query-caption/raw responses/private conversation. Verify remote bytes/SHA/blob/mode/tree/both refs, review actual charts, and archive check-snapshot-check.
