# Three native model filmstrip rows and case score columns

The final human sketch requests one row each for TA-STVG, TubeDETR and an
MLLM, real GT/native box overlays, Prediction/GT event bars, and two case
score columns. The subsequent human instruction allows the six native score
labels. They are rounded to three decimals and the columns retain a common
linear zero-to-one scale. Timelines have no timestamps or numbered ticks.

TA-STVG and TubeDETR use HC2-source-only to VidSTG native cross-domain outputs.
The explicitly authorized PTD/Qwen3-VL row is marked joint-trained reference;
it is not source-only cross-domain and is excluded from the strict aggregate.
All three real video/query inputs, frames and GT instances match. Candidate
zero is reused without new model calls. The actual private PNG, editable SVG
and independent PDF raster have been reviewed. Their integrity metadata is
published; dataset RGB, query, GT geometry, raw temporal ranges, prediction
arrays and weights remain local.

This illustration is selected post hoc by a documented deterministic ranking
among five existing cases where all three temporal scores exceed .5 and
spatial scores do not. It illustrates that event coverage does not ensure
instance grounding. It does not estimate prevalence or establish aggregate
spatial-only-error dominance, OPD benefit, expert reliability or an on-policy
mechanism. All 512 original anonymous cross-domain outputs, negative primary
contrasts and bootstrap results remain unchanged and published in
results/stvg_motivation_cross_domain/2026-10-09. The conditional endpoint in
results/stvg_motivation_filmstrip/2026-10-09 remains labeled post hoc.

The scalar records below are case scores, not aggregate quadrant percentages.
See docs/STVG_MOTIVATION_SKETCH_V10.md for the factual paper caption. Run the
preserved independent public primary/conditional scalar audit with:

```bash
python -B scripts/audit_stvg_motivation_filmstrip_public_v7.py
```

Only this finite figure presentation closes here. The P1-P6 paper stages keep
their separate actual root/math/state/dense/view/publication completion gates.
