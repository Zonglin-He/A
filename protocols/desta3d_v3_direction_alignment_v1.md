# Saved-residual direction alignment audit v1

Status at registration: CPU-only, no new model execution. The user instruction requests this audit before any prototype-direction oracle.

Question: do the previously successful frozen-output-span residuals and scalar
mask residuals differ in alignment with the same initial native objective?
Stop further scalar-mask tuning. Preserve THW/dual readers as a candidate, not
as a demonstrated general correction mechanism.

Use both pre-existing outcome-selected source cases, PANEL indices 7 (event)
and 3 (spatial), exact B1 and frozen PTD4B. No additional cases, GT pools, target
inputs, forward/backward, optimizer, or new native predictions. Read only sealed
artifacts. Event objective is mean CE over native endpoint time support; spatial
objective is mean full-152775-vocabulary coordinate CE on annotated native
anchors. NTP/MTP or restricted-coordinate gradients must not substitute for it.

Primary g_F is the stored *unclipped* initial free-merger STEP_01 gradient from
actuation_full001 (event is the preserved actuation_free002 result). Initial
span gradient is in 128-dimensional QR coordinates, not a full merger gradient.
Check its relation to sqrt(2560/128) g_F Q and report actual error without
silently projecting/replacing g_F. Require exact initial native/pixel/query/grid/
time/adapter identity, forward/native distribution equality, target and valid
support identity. Verify every consumed raw file against its original seal and
pin its original manifest/COMPLETE before numerical readout.

Directions: fixed step-30 span and free residuals, plus all eight stored matched
mask directions per case: late/early correct/wrong from location001, large
correct/wrong from large_mask002, context correct/wrong from context001. Do not
choose a favorable intermediate optimizer step. Keep small-vs-large magnitudes
explicit. Compute norm, cosine to span residual, cosine to -g_F and -g_F dot
delta (positive means local loss descent). Also report dot normalized to the
span norm. Zero directions have undefined cosine (null), never fabricated zero.
Independent NumPy float64 chunk reductions and PyTorch float64 reductions must
agree at atol=1e-9, rtol=1e-10. CPU synthetic tests cover signs, scale, zeros,
invalid support and QR-coordinate chain rule. No claim of exact finite BF16
program derivative, finite native utility, or causal sufficiency follows.

Predeclared branch decision: the attachment's strong sign diagnosis requires
span local descent > 0 and BOTH original late-correct and context-correct local
descent <= 0, with signs robust to the independent arithmetic tolerance. Large
correct is reported separately but is a positive rescaling of late-correct.
Report each branch separately; only if both branches meet this diagnostic will
the conditional prototype residual be registered. If mixed, sign-opposite, or
near-zero, report inconclusive/contradicted in this limited scope and do not
silently relax to an arbitrary cosine cutoff. Low cosine to a 30-step endpoint
alone does not establish misalignment with the initial gradient. A later
prototype can still be an explicitly qualified feasibility test, not a proven
root-cause fix.

Resources: CPU only, 4 threads, 900-second engineering ceiling, new evidence
under 32 MiB, free disk >=8 GiB. Do not copy large raw tensors. Record CPU wall
time separately; new GPU allocation is zero and prior cumulative GPU seconds
remain 42493.15652965409, cap=null. Write once to direction_alignment_v1, preserve
failures and original artifacts. Publish code/protocol and anonymous numeric
aggregates only, then archive check -> completed snapshot -> check.
