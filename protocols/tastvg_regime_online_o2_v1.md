# O2: Regime-Coherent Online Transfer

Question: can the unchanged O1 slow state transfer sparse expert preferences to
future non-expert queries within a persistent deployment corruption regime?
O1 had zero task transfer; O1.1 changed many decisions after readout scaling but
none of the three amplified scales improved mean vIoU. This test changes stream
construction only, with a smaller 16-source screen explicitly requested by user.

Use original C0.5/C3 first16 VidSTG source parents (ordinal0..15), one query/source,
all historically exposed. Sort by the SAME O1 source-only hash order, restricted
to these16, identical across five streams. Conditions: frame_drop_5,
frame_freeze_5, motion_blur_5, occlusion_5, exposure_5. 5 means the existing
5% full-input-video random burst protocol, NOT newly tuned corruption strength.
Seed0, source-hash burst start, exact old pixels/frames/preprocessing, no query/GT
in corruption/order. 80 arrivals are only16 unique sources, repeated five times.

Same frozen official Vid-source TA-STVG and UniversalVTG best+PE as O1/C3; same
max8 candidate generation and raw start/end/span-mean hidden features (768D),
exact two-offset native-envelope base score. Same FP64 CPU w768+b, zero start,
SGD .001, one strict-pair logistic step per expert, alpha1, no normalization,
gate, memory/prototype, LR/alpha tuning, H or model updates. Bias cancels exactly.
State resets at the BEGINNING OF EACH stream; never leaks across conditions.
Positions1,5,9,13 are expert (4/16=25%);12 nonexpert/stream,60 cells primary.
Fast output is expert-best, emitted before write; nonexpert uses arriving state.

Four unchanged arms: Frozen, Budgeted Rerank, Online Slow-Fast, Full Rerank.
Read only20 assigned cached experts before sealing all five online streams;
read remaining60 Full reference caches only AFTER online seal. Cached evidence
counts as logical calls (20/20/80 for Budgeted/Online/Full), not new computation.
All expert caches already exist from C3; verify pixels/candidate order/checkpoints.
Seal all four outputs before reading GT for original16 keys. GT scoring only.

Primary: Online minus Budgeted at nonexpert arrivals, tIoU and corrected vIoU.
Report each condition (12) then equal-five-condition macro. Identical source order
means macro is also average over12 parent-level five-condition averages; bootstrap
those12 parents, NOT60 independent observations. All-arrival secondary uses16
parent clusters. Descriptive paired10000 bootstrap seed20260929 is conditional on
these fixed shared states, not resimulated online trajectories. Retain all cases,
selection changes, state norms/losses, expert agreement and negative tails.
Readout-scale diagnosis is observational at alpha1; no O1.1 sweep on O2 states.

Positive mean transfer motivates further testing, not proof locality is the sole
cause: O1 uses32 sources/eight writes and heterogeneous source assignments, while
O2 has16/four writes. Zero transfer with tiny decision effects is inconclusive
about representation; O1.1 does not establish adequate O2 scale. No universal
Slow-Fast impossibility or automatic prototype follow-up. Production unchanged.

New frozen capture80 cells only for missing hidden features; reuse exact existing
native outputs via pixel/logit/box identity assertions. Serial GPU cap1800s, >=8GiB
free floor, <=2GiB new artifacts; cached teacher and online loops CPU. Independent
NumPy state/gradient reconstruction, reset/chronology, zero-native identity, dual
metric routines and public scalar aggregation audit. Archive code/results/negative
findings and publish sanitized GitHub export with remote byte/hash readback.
