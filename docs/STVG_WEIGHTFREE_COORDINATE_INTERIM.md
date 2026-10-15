# Sequential tuning: interim evidence, expanded 64-source cohort

This is a running study, not a completed confirmation or Table1 result. Each domain has16 development and16 frozen confirmation parents. The first32 already had measured results and are explicitly development now. All64 are historically exposed project sources.

The previous method had temporal ranking and update failures, and spatial pseudo-reference improvement did not guarantee tube improvement. Independent NumPy replay of28 original temporal fits matched every saved grid decode and Adam path (504 checks, max theta error about2.08e-13).19/28 final regularized preference objectives exceeded their initial values.

Each coordinate inherits the previous winner separately per domain; all32 predictions seal before development scoring. Parameters are selected within a fixed finite grid, not proven globally optimal. The additional32 are excluded from selection. The original QA observation clock is held fixed; tuned temporal output does not silently move spatial supervision.

Completed stages:

- 01_lr: 115 new spatial fits.
  hc2: lr=0.005, development joint vIoU=30.799196%.
  vidstg: lr=0.0075, development joint vIoU=9.950818%.
- 02_sigma: 69 new spatial fits.
  hc2: sigma=0.0125, development joint vIoU=30.803494%.
  vidstg: sigma=0.2, development joint vIoU=10.061378%.
- 03_tau: 92 new spatial fits.
  hc2: tau=0.2, development joint vIoU=30.806679%.
  vidstg: tau=1.0, development joint vIoU=10.067195%.
- 04_steps: 92 new spatial fits.
  hc2: steps=5, development joint vIoU=30.824940%.
  vidstg: steps=5, development joint vIoU=10.067195%.
- 05_lr_refine: 46 new spatial fits.
  hc2: lr=0.003535533905932738, development joint vIoU=30.843203%.
  vidstg: lr=0.0075, development joint vIoU=10.067195%.
- T01_lr: 0 new spatial fits.
  hc2: lr=0.0, development joint vIoU=33.485816%.
  vidstg: lr=0.0003, development joint vIoU=10.706815%.
- T02_tau: 0 new spatial fits.
  hc2: tau_T=1.0, development joint vIoU=33.485816%.
  vidstg: tau_T=0.5, development joint vIoU=10.811254%.
- T03_steps: 0 new spatial fits.
  hc2: steps=5, development joint vIoU=33.485816%.
  vidstg: steps=5, development joint vIoU=10.811254%.
- T04_l2: 0 new spatial fits.
  hc2: l2=0.001, development joint vIoU=33.485816%.
  vidstg: l2=0.001, development joint vIoU=10.811254%.

All positive, negative, neutral and missing values for completed stages are included. No confirmation results are available in this interim report. The current user instruction cancels Bootstrap; no resampling or confidence intervals will be computed. A zero temporal learning-rate winner means final-output Projection-only; the original QA head still provides the fixed spatial observation clock. Later parameters at zero lr are algebraically inactive, not evidence that they never matter. The early own-cgroup initialization failure occurred before GPU inference and was resolved in separate002 supervisor; originals retained. The confirmation input-reader hookup and zero-lr Projection-only accounting were corrected before those stages executed. CURRENT and the old queues remain unchanged.
