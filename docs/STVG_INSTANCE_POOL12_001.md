# Whole-pool instance-evidence diagnosis

All actual candidate IDs in the twelve previously reviewed development sources are now checked, using cached historical boxes, masks, RGB and teacher ID bindings. No historical Selective Search/GrabCut result is regenerated. The twelve scene/pool panels were actually viewed, including positive, negative and unchanged sources.

Among48 windows,25 have exact physical-position GT and23 remain unknown. No nonSource candidate reaches box IoU .5 on those25 windows; Source reaches it on7. These are geometric coverage statistics, not semantic identity labels: visual review finds a few plausible correct instances at lower IoU, and annotation extent can differ from visible-instance extent. Do not infer that every nonSource candidate is another object or that all48 windows lack the referent.

Frequent nonSource candidates depict background, clothing/body fragments, other entities or mixed regions. Source often supplies the most recognizable instance hypothesis. There are also sources where Source itself follows the wrong person/region; reorganizing existing crops cannot create a missing whole-object hypothesis. All candidate-level scalar IoUs, selected IDs, presence, nulls and window strata are included anonymously. The panel is result-stratified and previously exposed, not a representative confirmation set.

## One prospective change, not a prompt search

The next fixed32-source experiment tests Source-seeded RGB foreground as one additional instance hypothesis within the same nine-candidate budget. The old last proposal slot is replaced, other IDs stay fixed. This is a new prospective candidate, not a reconstructed historical foreground. It uses no new weights, labels, detector or tracker. Original three-iteration GrabCut and local DIS are reused; the complete foreground is retained. Expansion1.25 and central-half sure-foreground seed are locked once, with no sweep.

This hypothesis inherits the Source position prior, can merge background, can delete a useful old last-slot proposal and cannot reliably discover a referent outside a wrong Source region. It is not claimed as independent object discovery or as a paper contribution before evidence. Invalid masks retain the original pool and remain in the denominator. Original masks/rectangles/seeds and new intermediate outputs are preserved privately.

Before E0/E1/E2, the complete Temporal Teacher baseline is being corrected under identical original8x448 sampling, prompts, candidate intervals,512-parameter objective and CPU lineage. E0 and E1 then use the same corrected Temporal/Spatial time version and physical centers; E2 shares exactly E1 references and changes only to original validated OPD. HC lr .0045/Vid .027 stay fixed. All no-reference/no-update/failure cases remain. No further Full-scene/Pairwise gate, Teacher replacement, LR search, Bootstrap, CURRENT change or full official evaluation is launched.

This publication completes the pool-diagnosis group only. The prospective proposal and E0/E1/E2 task effectiveness are not results yet. Private media/annotation coordinates/weights/tensors/Query-caption/raw teacher outputs are excluded.
