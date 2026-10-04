# P0 execution and handoff

The user authorized the staged mechanism in attachment f7175e26. Only P0 is
implemented here. Read the protocol and saved STATUS, not this plan, for actual
completion. Old queues and CURRENT_METHOD remain untouched.

1. `python -B scripts/test_tastvg_privileged_p0_v1.py`
2. `python -B scripts/run_tastvg_privileged_attention_p0_v1.py prepare`
3. `python -u -B scripts/run_tastvg_privileged_attention_p0_v1.py run`
4. Verify GLOBAL_PREDICTION_BARRIER and both no-GT full-model parity records.
5. `python -u -B scripts/score_audit_tastvg_privileged_p0_v1.py run`
6. `python -B scripts/report_tastvg_privileged_p0_v1.py`
7. Root inspects plots, result signs, evidence coverage, positive/negative cases,
   private arithmetic audit and public aggregate audit; save ROOT_VISUAL_REVIEW.
8. `python -B scripts/publish_tastvg_privileged_p0_v1.py prepare` builds a reviewed
   exact public Git tree/manifest. Publish that manifest with authenticated GitHub
   Git-data APIs; preserve the parent, use a non-forced main ref update.
9. `python -B scripts/publish_tastvg_privileged_p0_v1.py verify <remote_commit>`
   independently reads every raw remote file, checks bytes/SHA256/Git blob SHA,
   verifies main/tree/parent and reruns the public CPU checker in the checkout.
10. Archive check/snapshot/check and FINAL_COMPLETION close this P0.

This fixed P0 failed the predeclared all-four-panel positive-mean gate. No OPD,
LN consolidation, expert swap or strength search follows automatically. Weak
positive cases are retained; the outcome does not prove the route impossible.

GPU code and defaults were locked before GT. CPU scoring/audit code has a separate
pin. Any engineering repair must preserve failed records and revision hashes;
never overwrite a sealed prediction or alter the gate based on outcomes.
