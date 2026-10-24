# CG-specific calibration and published Tent/SAR follow-up — complete

All authorized measurements on the two available domains are complete. CG-STVG was run last. The third benchmark is **deferred by the user**, and Fig1 is excluded. Production SimplePath/Native2D/eight observations and all earlier source/result roots remain unchanged. The original negative CG transfer results are preserved.

## Fixed CG confirmation: calibration helped HC relative to old settings, but did not establish a consistent improvement

Development used64 parents/domain, two frozen10-profile stages. All20 profiles, including losses, are in `cg_dev/A` and `cg_dev/B`; no posthoc combinations. The selected parameters were bound before confirmation: **HC gamma0/link.25/context1.5/kappa64/beta.25**; **Vid gamma.5/link.25/context1/kappa64/beta1**.

Confirmation was metadata-fixed before new development scoring: all35 remaining HC parents and128 hash-selected remaining Vid parents, excluding all192 previous CG parents/domain. One query/parent. They are new to CG evaluation but were historically exposed to TA-STVG evaluation, **not historically unseen data**. Compare all configurations on this SAME new cohort; do not compare these means with a different historical cohort as a treatment effect.

|Direction|Parents|SourceMBR vIoU|Original transferred Full|Selected Full|Selected−original (pp,95%CI)|Selected−Source (pp,95%CI)|
|---|---:|---:|---:|---:|---|---|
|HC|35|28.737788%|24.530040%|26.420382%|+1.890341 [-1.057037,5.228521]|−2.317406 [-7.803496,3.241156]|
|Vid|128|9.192995%|11.879313%|11.530333%|−.348980 [−.935209,.148858]|+2.337337 [.412902,4.357343]|

SourceMBR→selected temporalIoU: HC43.406607→45.304006% (Δ+1.897398pp CI[−1.302417,5.449316]); Vid21.093027→22.862605% (+1.769578 CI[−.115706,3.675745]). SelectedFull−Space vIoU HC+.306535pp CI[−1.662525,2.241110]; Vid+1.371367 CI[.281782,2.571649]. All CIs are pointwise parent bootstrap, not familywise-adjusted. The new profile is not established superior to original settings in either direction; Vid has a positive selectedFull−Source joint-quality contrast on this cohort, HC does not.

Severe Source damage (t or v decline>5pp, union once): HC selected9/35 vs original15/35 (Space7/35); Vid selected22/128 vs original21/128 (Space8/128). All individual t/v flags, signed changes, short/long groups and CIs are in `cg_confirm`, including all6 arms. The original nativeMAP Source is separate: MAP→MBR readout gains are not external feedback gains.

## Mechanistic conclusion, updated by confirmation

The DEV diagnosis in [cg_mechanism/DIAGNOSIS.md](cg_mechanism/DIAGNOSIS.md) identifies shared-correction damage and weak candidate-score/quality alignment. It is an offline analysis, never a GT inference gate. In the new HC confirmation, gamma0 makes SharedCorrection equal Source up to native clamping roundoff, eliminating the global correction change. Yet fixed-support unobservedIoU Source67.832981→Space63.982395→Full63.029979%. Thus residual damage is now localized to spatial candidate/path selection, not a remaining global correction. Compared with original Full59.971831%, selected Full recovers3.058149pp CI[.816119,6.456549], but remains4.803002pp below Source (CI crosses0).

Vid unobservedIoU Source40.143025→Space40.728221→Full40.602997%, selected−original+1.051453pp CI[−.532163,2.707836]. The combined gamma/context change loses.882525pp temporalIoU versus original, CI[−1.842741,−.085210]. Since both factors changed, this comparison cannot attribute the loss solely to context. No confirmation-driven reselection or expanded grid follows. Further changes to appearance scoring would be a separate mechanism study requiring a new direct request.

## Published existing TTA algorithms, explicit STVG adapter

Tent(ICLR2021) and SAR(ICLR2023) use unmodified official SAR repository20f6e24b17525f34503510afccedc0629b67b7c4 code (including its LN-compatible Tent comparator). Original Tent provenance e9e926a668d85244c66a6d5c006efbd2b82e83e8 is also retained. Sources and completeSHA are in `reference_audit.json`; [Tent paper](https://arxiv.org/abs/2006.10726), [SAR paper](https://arxiv.org/abs/2302.12400). This is a disclosed final-time-decoder STVG adaptation, not a full-network or unchanged classification-benchmark reproduction. Actual native seven-argument final-pass contexts and Source boxes stay fixed; existing normalization affines update, all other parameters frozen; dropout eval, episodic reset, after-update original native hull/MBR. SAR retains its reliability filtering/SAM/recovery and exclusions. No expert is used. Earlier custom Adam entropy/PhotoConsistency/PseudoBox controls are not relabeled as these published methods.

64 development parents/domain, Tent6/SAR12 trial grids: SGD momentum.9,LR1e-4/1e-3/1e-2 ×1/5 steps, SAR margin.4/.8logN, original SAMrho.05. Both methods/directions selected LR1e-4/five steps, SARmargin.8logN, before fixed128-parent/domain confirmation. All cohorts were historically TA-exposed.

|Direction|Source vIoU|Tent vIoU|SAR vIoU|Main Full reference vIoU|Tent/SAR−Source(pp)|
|---|---:|---:|---:|---:|---:|
|HC|28.044780%|28.054833%|28.054833%|30.689863%|+.010053 CI[0,.030160]|
|Vid|14.102136%|14.102136%|14.102136%|15.205425%|0 CI[0,0]|

Tent norm states change128/128 episodes/domain; SAR116/128HC and117/128Vid, the rest exactly unchanged. Native MBR outcomes: HC1win/127ties, Vid0win/128ties for both;0numerical failures, severe Sourceunion0 for both. MainFull severe union25HC/28Vid is retained. This restricted adapter yields minimal discrete-readout changes, not a conclusion that every TTA objective or update scope fails. Full precision signed statistics, reliability-empty counts, conditional optimizer timing, all trial losses and paired comparisons are in `tta`.

## Verification, costs and retained artifacts

- CG developmentA/B:1,280 configuration-query predictions each,6,400 official/common/dense comparisons each; transferred default and selectedA→Bbase arrays bitwise. No new model/expert for cached development.
- TTA development:2,304 predictions/6,912 real optimizer iterations/32,512 decoder calls,519.637654s worker wall. Confirmation:512 predictions/2,560 iterations/10,240 calls,212.969515s. All seals precede independent CPU labels. Postseal4,608+1,024 prediction/trace SHA checks, exact resets, no numerical failures. These cached-context optimizer times exclude original Source acquisition/loading/early context; no zero-cost deployment claim.
- CG confirmation:163 predictions×6arms=978 official/common/dense comparisons, maxerror2.7755575615628914e-15. Postseal815 prediction/capture/receipt/acquisition/teacher payload SHA checks,163 unique same-ReferDINO joint8 contexts,0cache hits.167 completed Source forwards=163 ordinary+4 native reference extras.832.543170s worker wall. Source capture290.918614s includes hooks/transfers and reference extras, not pure GPU forward; pipeline sum803.277753s, nested component costs are not added. Source model loads6.716334sHC and5.985836sVid once each; teacher load4.026299s once. Every PID/start segment retained.
- New follow-up total16,594 official metric comparisons. All owned controllers/predictors/scorers/teacher exited and sharedGPUlock is available. Original frozen Source420FP32/single-grid/cross-source EMA/native sampling and expert evidence remain unchanged. Prior fullGPU executable OOM/storage-schedule recovery limitations remain documented in the earlier suite; no claim of an unmodified fullGPU reference executable.

The private `outputs/paper_followup_v1` holds manifests, predictions, acquired features, media references and audit receipts. Public exports contain code/protocol/scalar and signed results only; no captions, GT coordinates, media, masks, features, weights or credentials. Earlier PseudoBox data remain a self-designed optional box-residual control, not a published baseline. All old roots, previous positive/negative findings, original production acceptance records and ancestors are preserved. Final publication uses both existing refs with fresh CAS and all file/mode/bytes/SHA/tree/ancestor verification; only then is the same heartbeat paused.
