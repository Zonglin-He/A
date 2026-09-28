# Isolated source coordinate-normalizer correction

Prior free002 completed the fixed source30+30 steps. Temporal succeeded; spatial
failed native grammar at steps24..30 despite lower coordinate-only CE. Raw box
8 coordinate slot4 emits non-coordinate ID99835; event/ref/anchors and box start/
end markers remain unchanged. Official sample_token_ids uses full vocabulary
argmax, while the preceding diagnostic normalized only1001 coordinate classes.
This is a demonstrated objective/action-space gap, not proof of all oracle failure.

Only change: native spatial four-coordinate CE uses the entire152775-token
vocabulary and the corresponding GT coordinate token IDs. Same GT coordinate
positions/native anchors/mean denominator, source28546/B1, free merger tensor,
fresh AdamW .01/wd0/clip1, exactly30steps, BF16body/head/cast, reference/cache,
no structure loss and no generated-token repair. Save full vocab logits plus
restricted original readout. Every differentiable replay equals native exactly.
The event19368 completed control is hash-reused unchanged. Original failed and
recovered trials remain evidence, not overwritten or relabeled successful.

Before GPU: two CPU controls test unchanged slices/grad scope/exception restoration
and show the excluded competitor's missing gradient. These are not GPU results.
At step30 evaluate original native output and all malformed outputs; no best step.
If both controls qualify, register the frozen-output-span control using this same
branch-specific objective. Otherwise retain the failure; no LR/step/precision grid.
No target data, expert or OPD. Exposed source supervision is diagnostic only.

Same3600s engineering limit,10GiB maximum artifacts including reused links,
8GiB disk floor, single GPU and all measured overhead accounting cap=null.
