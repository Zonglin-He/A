# Three fixed component ablations and event-coverage Figure 3

The complete component table combines the preserved P2/P3 controls with three
new interventions: whole observed-video Uniform4, persistent query residual
with per-query Adam reset, and full LayerNorm carry with query/Adam reset.
All 5,376 new formal fits, both directions, conditions and fixed orders sealed
before labels. Seventy-two qualification arrivals / 144 actual full fits stay
excluded from formal predictions. Main source checkpoints, hyperparameters,
frame grid, sparse budget, reward/optimizer and Native WHEN are unchanged.

Read [the actual root review](ACTUAL_ROOT_REVIEW.md), [the component table](COMPONENT_TABLE.md),
all per-condition/order contrasts, parent effects and negative tails in
[ALL_STATISTICS](ALL_STATISTICS.json), and the fixed-bin [coverage analysis](COVERAGE_ANALYSIS.json).
The cohort uses the original 128 historically exposed parents per target; this
is not a fresh population claim. Confidence intervals use 10,000 paired-parent
draws and fixed seed 20261011, without multiplicity adjustment. Coverage is
postseal descriptive grouping, not tIoU or a causal coverage intervention.
Both arms keep their own online LayerNorm trajectory, so their final contrast
also includes inherited effects. Averaging the two fixed orders does not
establish robustness over arbitrary stream orders.

```bash
python -B scripts/audit_stvg_opd_design_ablation_public_v1.py results/stvg_opd_design_ablation/2026-10-11
```

Two historical table display values are corrected from full sealed row
readback. No old score or prediction is rewritten. The original failed CPU
projection, the CPU002 dynamic-status failure and the additive CPU003 correction have separate bindings. The GPU
runtime is preserved. Original P1-P6 closure remains unchanged and separate.
Actual expert calls, reused observations, valid-GT quality denominators and
recorded fitting times are retained; cached fitting costs are not cold latency.
Six completed P3 cases and their display frames were fixed before the new GT
read, then compared with all three new controls. Their historical success/harm
roles are not newly selected positive/negative examples. Only scalar signals
are public; actual RGB comparison sheets remain private. Signal curves use
each arm's admitted evidence, so they do not share the frame-level denominator.
The final signal PNG/PDF uses visual003 to shorten clipped axis labels. All 24
saved curves and case selection are unchanged; the original provisional images
remain locally preserved and bound by the display-only capture receipt.

Private RGB/video, query/caption, labels/box geometry, native outputs, actions,
gradients, optimizer/LN/prefix tensors, weights and raw fit/input caches are
excluded. Only anonymous scalars, code/configuration, hashes and scope/count
receipts are public. Common orchestration transports the same human attachment
hashes as constants instead of publishing private attachment paths; its original
and public file hashes are explicit. Private mathematical/scientific runtime
pins bind the actual execution, not a claim that an asset-free checkout can
rerun the GPU experiment. The portable audit independently checks all public
arithmetic, not the complete decoder Jacobian or CUDA transcendental safety.
The initial source privacy validator rejected its own bare attachment-path sentinel.
The original failed exporter and unsealed stage remain captured; separately pinned
exporter002 rejects complete private paths while allowing the literal guard text.
No result/scientific input or GPU runtime changed.
Final close additionally requires exact remote bytes and archive receipts.
