# Spatial-DeCoTA critic P0 execution

The user authorized this fixed paired episodic experiment. Only the new
`artifacts/tastvg_decota_critic_p0_v1` queue may execute. Historical C1,
TA-STVG A, DTA, full-query and other paused queues remain closed.

1. Run the six CPU objective contracts; prepare and hash-lock the old C1
   cohort, cached expert support, Frozen H and current registries.
2. Run one original clean no-GT full-backbone parity/reinsertion smoke per
   dataset. Both paths must match all eleven cached fit states exactly.
3. After `SMOKE_ROOT_ACCEPTANCE=pass`, start the finite controller once.
   VidSTG then HC2 each calculate 288 unique paired Direct/Critic fits.
   Duplicate-order reads reuse the same episodic result.
4. Globally seal all 576 payloads before CPU GT evaluation. The controller
   runs independent objective/Adam/state/scorer audits and creates reports.
5. Root visually inspects the two figures and reads paired source intervals,
   clean, harm, empty/zero-gradient and proxy/task mismatch evidence. Publish
   the implementation, frozen protocol and all anonymous results, verify
   remote bytes, update the research archive and save `FINAL_COMPLETION`.

Commands (authorized controller; not a recurring task):

```bash
.conda/tubedetr/bin/python -B -m pytest -q tests/test_tastvg_decota_critic_p0_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_decota_critic_p0_v1.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_decota_critic_p0_v1.py smoke
.conda/tubedetr/bin/python -B scripts/continue_tastvg_decota_critic_p0_v1.py
```

No scientific setting, support definition, temperature or cohort may be
changed because of outcomes. Preserve engineering failures and pin a
revision for necessary implementation repairs. Do not overwrite or
silently re-run a completed prediction. Ordinary IoU's disjoint-box zero
gradient is measured behavior, not permission to substitute GIoU.

`completed_pending_root_visual_publication` is not final completion.
No automatic production promotion, LN-persistence P1, cross-domain,
preservation, gate or sweep follows this P0.
