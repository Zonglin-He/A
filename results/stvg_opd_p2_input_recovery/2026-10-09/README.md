# Fixed P2: strict legacy input compatibility and real continuation

P1 has completed its actual root, visual, anonymous publication and archive gate.
All P1 results, including the inconclusive VidSTG contrast against DINO Refine,
remain in `results/stvg_opd_p1_complete/2026-10-09`.

The original P2 four arms and all source checkpoints, rosters, parameters,
physical conditions, order resets, inherited LN, query/Adam resets and final
readout are unchanged. The historical engineering002 launch and its full
96-arrival/192-real-fit qualification remain in
`results/stvg_opd_p2_launch/2026-10-09`.

That controller subsequently failed before accepting any formal P2 prediction:
the older P0 clean-input header omitted `source_model_state_sha256`, although
it already referenced the original baseline NPZ by its exact cache hash. The
failure occurred before fitting. Original code, locks, logs, qualification
receipts and pre-repair status snapshots are preserved; there is no dead
failed-fit serialization to compare.

The additive input-schema003 bridge verifies the original hash-bound cache,
its original runtime receipt, exact pixels/frame IDs/interval/native boxes and
the complete admitted expert evidence before reading its recorded source hash.
No input bytes or old receipts are rewritten. All 256 legacy inputs were read
back, four old/current pairs passed and 48 deliberately wrong variants failed.
Unknown legacy schemas fail closed; source provenance is not guessed and no
input equality check is relaxed.

Both first-formal Direct fits were executed and fully repeated on the real GPU:
four new fits, exact equality against the previous real qualifications, original
math/state/reset/writeback readback, zero qualification predictions accepted,
zero new DINO calls and no GT. Formal acceptance then required exact equality
against this new qualified first fit. A new single finite controller actually
resumed the original formal stream. The bounded root receipt records an actual
accepted immutable prefix, not a projected future total or phase seal.
An initial unsealed CPU root-helper namespace initialization error was also
preserved and repaired; the subsequent complete prefix readback passed without
changing any GPU fit, prediction or receipt.

This completes a metadata recovery and resumption gate only. All P2 arms and
both directions must globally seal before GT scoring. Actual root statistics,
math/state/dense/cost/cases, plot inspection, anonymous publication and archive
closing are still required before P3. P2 and the paper remain incomplete. EATA
and unrelated old queues remain paused. The old controller PID and old running
text are historical; use the latest schema003 launch and dynamic stage status.
Private media, query/caption, GT geometry, weights, box/actions, qualified or
formal fit payloads, gradients and Adam arrays are excluded.
