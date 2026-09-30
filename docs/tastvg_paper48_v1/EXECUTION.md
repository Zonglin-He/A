# Paper48 execution

User replaced all earlier queued plans. Old B1 stopped at temporal22822, spatial86352 completed; no old partial scoring. No total deadline under the latest user authorization; EXECUTION_POLICY.json supersedes the preserved original TIME_BUDGET.json.

P0 Raw-RKL only (480) first. P1 experts then online8040 then CPU score/report. P2 experts/4096/score. P3 shared experts/2304/score. P4 uncached100. P5 required HC-STVG-v2 (up to128 sources, one order, six conditions); no preparation cutoff or time-based skipping. P0–P4 alone does not complete the task. No baselines/Pairwise/mixed/cross-domain/TubeDETR. Current implementation/run state comes from STATUS and QUEUE_STATUS, not this plan. Follow protocols/tastvg_paper48_v1.md. Completed phases require archive and verified GitHub publication. Monitor must never restart B1 or old paper queue.

P5 executable continuation: `continue_tastvg_paper48_p5_v1.py` prepares/validates the locked128-source roster, runs spatial then temporal specialists (192 each), native online768, then CPU score/audit. HC2 official checkpoint and checkpoint-relative support are used; no Vid-trained warm state is inherited. `P5/COMPLETION.json` plus passing audit are required before the parent queue reports P0–P5 execution complete. GPU P5 integration is deferred until P4 finishes on the single GPU; failures preserve receipts for root recovery.
