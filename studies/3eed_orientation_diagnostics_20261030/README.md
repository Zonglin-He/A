# 3EED orientation diagnostics, 2026-10-30

P0 source-only training was cancelled at the user's cost boundary. Zero full epochs,
no source-only checkpoint and no automatic continuation. The estimated 2.35 hours
comes from historical official logs, not a measured RTX5090 run. Official release
contains only a mixed-platform checkpoint. No source-only migration claim is made.

P1-P3 are descriptive CPU diagnostics on 54 historical mixed-model cached parents,
23 scenes (18 Waymo, 2 Drone, 3 Quad). They are not a new confirmation cohort.
GT yaw injection and finite yaw search are offline diagnostics only; the finite
search is a feasible lower bound, not a certified continuous upper bound.

See [Chinese report](REPORT.md), [paired scalar comparisons](results/PAIRED_COMPARISONS.json),
[all primary scalar results](results/P1_P3_SCALARS.json),
[reliability associations](results/P2_RELIABILITY_ASSOCIATIONS.json),
[controlled perturbations](results/PERTURBATION_SUMMARY.json), and
[separate full-box fitting group](results/FULL_REFIT_SCALARS.json).

Only source code, protocols, scalar results, signed tails, cancellation/provenance
and independent audits are included. No data, annotation coordinates, GT yaws,
point arrays, boxes, embeddings, weight archives, media, raw captions, raw frame
IDs, physical scene names, external paper PDFs or local training logs are included.

Source code is byte-exact and retains the original workspace-relative directory
layout. This is a research evidence export, not a standalone dataset installer.
Private cached inputs and original input manifests are not published. Reproduction
needs authorized official data, the prior round2 preparation/source pipeline, and
new local path binding plus a new registration lock; historical locks must never
be overwritten. JSON mirrors remove local absolute paths and hash physical scene
names. Their exported hashes can differ from original evidence hashes; original
source hashes are retained in EXPORT_PROVENANCE.json. No compatibility rerun or
training is launched by opening this bundle. The cancellation receipt is binding.

The optional complete box fitting supplement was locked after the initial historical
diagnostic was scored. It uses the existing frozen fitting API and no GT tuning,
but is explicitly a posthoc capacity extension. No new TTA method was adopted.
