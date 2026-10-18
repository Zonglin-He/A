# Joint event–trajectory posterior: source96 → exposed target32

Source-only learned prior, fixed finite source/target protocol. No Qwen, no new DINO, no production promotion. Previous target oracle exposure is retained. This is a feasibility pilot, not an independent test.

## Target results

Values in percent; full 16-source denominator in each domain.

| Method | HC vIoU | HC tIoU | HC sIoU | Vid vIoU | Vid tIoU | Vid sIoU |
|---|---:|---:|---:|---:|---:|---:|
| Source | 18.831315 | 45.792579 | 37.714391 | 7.147781 | 31.818184 | 36.753536 |
| Direct | 18.446397 | 45.792579 | 34.238478 | 11.604658 | 31.818184 | 47.854281 |
| joint_diffusion | 4.527597 | 25.318009 | 31.241872 | 1.636205 | 17.498608 | 34.173009 |
| joint_deterministic | 1.885150 | 33.564152 | 17.252934 | 1.419660 | 17.744730 | 36.046170 |
| separated_diffusion | 2.568395 | 26.563097 | 29.076740 | 1.660566 | 17.193702 | 35.043349 |

## Matched contrasts

Paired source-level t intervals, percentage points. Intervals are descriptive on the exposed development cohort.

| Contrast | HC ΔvIoU [95%] | Vid ΔvIoU [95%] |
|---|---:|---:|
| joint_diffusion - Direct | -13.918800 [-25.322381, -2.515219] | -9.968452 [-15.865995, -4.070910] |
| joint_deterministic - Direct | -16.561246 [-27.012268, -6.110224] | -10.184998 [-16.226340, -4.143656] |
| separated_diffusion - Direct | -15.878002 [-26.740441, -5.015563] | -9.944092 [-16.194959, -3.693224] |
| joint_diffusion - joint_deterministic | +2.642446 [-0.050442, +5.335335] | +0.216546 [-0.906405, +1.339496] |
| joint_diffusion - separated_diffusion | +1.959202 [-0.785405, +4.703809] | -0.024361 [-1.054430, +1.005709] |

## Decision

Prelocked transfer gate: joint diffusion must exceed all-anchor episodic Direct mean vIoU AND tIoU in EACH domain. Actual gate: **False**.

Actual posterior-guided Student parameter adaptation is **not triggered** because the prerequisite failed.

The first two requested controls have actual measurements. The third (inference versus parameter adaptation) is conditional and is not reported as measured when the prior fails. This is not a failure of a measured Student update.

## What this implementation does and does not establish

- Whole event and32 trajectory knots are jointly denoised with dynamic video/time ROI, full Query and interval-exterior context. The separated control has exactly zero bidirectional state derivatives; joint derivatives are nonzero. Derivatives demonstrate implemented dependence, not beneficial coupling.
-171,268 parameters per prior, two source directions,60epochs/360AdamWsteps each. Checkpoint selection is source validation only. Deterministic correction has matching architecture/features/steps, but Source/Direct-corruption rather than GT-centered DSM training; it cannot isolate diffusion as a single changed loss term.
- Association head estimates source geometric compatibility, not verified identity. Other/nonmatch are marginalized together; event-exterior source anchors are unknown, not negative. No distance-to-Source gate. Sparse experts are observations, not forced targets.
- Frozen map pooling8x8,32context times and32event knots may discard detail. Spatial training supervision is interpolated only between real source event-interior annotations, never invented outside the event. Final exterior predictions retain Source.
- Four fixed-seed samples/16prior evaluations each; latent averaging can mix identities. Observation guidance uses a bounded approximate noisy-state likelihood, not an exact clean posterior/DPS implementation. No automatic claim of SDA theoretical guarantees.
- Small source training set and cross-domain shift are unresolved risks. A negative pilot does not establish that all diffusion/data-assimilation approaches fail. No extra data, seeds or noise searches were run.
- All target predictions were sealed before this run’s target scoring, but these32 sources were exposed previously. No independent confirmation, official full-set claim or novelty claim.
- R@.3/.5, full signed per-source deltas, fixed-Source-time geometry, source validation/cost and unknown supports are in machine-readable results. Pixel panels are private and were actually viewed; no independent human identity labels were added.

Primary sources: [SDA](https://arxiv.org/abs/2306.10574), [official SDA code](https://github.com/francois-rozet/sda), [Diffuser](https://diffusion-planning.github.io/), [Switchable Constraints](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf).


## Source validation and failure localization

The frozen posterior also failed on held-out SOURCE validation, so the current failure is **not explained by cross-domain transfer alone**. No target Student update was attempted. Source validation vIoU percentages:

| Source domain | Direct | Joint diffusion | Deterministic joint | Separated prior diffusion |
|---|---:|---:|---:|---:|
| hc2 | 46.093364 | 18.551165 | 23.370473 | 21.901716 |
| vidstg | 17.919784 | 14.948784 | 2.354189 | 10.810612 |

Denoising validation loss used for checkpoint selection is not evidence that an approximate conditional sampler improves tubes from actual Student starting states. This prototype has not established a useful source posterior, let alone transferable feedback for adaptation. No extra epochs, posterior noise search or target-based checkpoint reselection followed.

A representation-only32-knot readout changed Direct vIoU by HC **−0.090804pp**, Vid **−0.010515pp**, far smaller than the joint posterior loss. This excludes trajectory regridding ALONE as a sufficient explanation; it does not exclude information loss from pooled visual features or poor learned prior/inference. Fixing Source spatial boxes and using only the inferred time interval gives HC **10.089822%**, Vid **2.175157%** vIoU; using inferred spatial boxes with Source time gives **15.020859%**, **6.640224%**. These are CPU output-accounting readouts, not independently trained T-only/S-only methods.

Source-HC geometric association validation AUROC .86 transfers to target-Vid .202797 (24 precise-GT anchors); source-Vid .790909 transfers to target-HC .721739 (43 anchors). These are descriptive geometric-compatibility diagnostics, with within-query correlated anchors; they are NOT independent visual identity labels.39 admitted event-exterior anchors remain unknown. Source validation discrimination does not establish target association reliability.

The inferred joint event intervals retain45/106 original admitted observations;8 of30 nonempty queries have no observation inside their final event. This is a limitation of this implementation's uninformative exterior likelihood, not proof those exterior observations are wrong. It presently lacks an event-exterior identity model.

The factorized control removes cross-state dependence from the LEARNED PRIOR; all arms retain the same observation model, which can itself couple event time and trajectory. It is not a claim of complete posterior independence. The deterministic control also differs in its training corruption law. Consequently the pilot does not establish diffusion necessity or universally beneficial temporal/spatial coupling.

All signed results are retained: joint-minus-Direct HC3 improved/12 worsened/1 equal; Vid1 improved/12 worsened/3 equal. Paired mean vIoU difference HC−13.918800pp [−25.322381,−2.515219], Vid−9.968452pp [−15.865995,−4.070910]. The limited positive examples do not pass the prelocked transfer gate.

The implementation consumes the existing privately bound Source/DINO feature roster. The public export provides code, exact numerical configuration, hashed anonymous cohort, all scalar results, training histories and validity receipts; private videos, annotations, raw text/responses, tensors and learned weights remain excluded. It is not a turnkey data download or a new deployable TTA method.
