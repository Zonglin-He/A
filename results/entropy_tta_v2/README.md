# Paired entropy-minimization TTA diagnostic (Fig. 1)

Completion status and numerical verification are recorded in verification.json once available. This is an actual optimization experiment; no quality result is claimed before the complete 256-query seal and separate CPU scoring.

## Fixed experiment

The user explicitly authorized a real optimization experiment after rejecting a conceptual-only figure. This is independent of the frozen SimplePath main method and never resumes the cancelled coupled ablation.

Each target domain contributes 128 parent videos and one query per parent, chosen by deterministic SHA-256 with seed 20261024 from the complete existing official input manifests. No label, confidence, loss reduction or grounding result selects cases. The official datasets have already been exposed in previous work; this is a new paired optimization measurement, not an unseen benchmark.

**Loss:** arithmetic mean Shannon entropy of the native start/end distributions, averaged across both original offsets, in nats. **Adaptation:** Adam, lr0.001, default betas0.9/0.999, eps1e-8, no weight decay, exactly five steps; last state only. Only the original temporal decoder's 9,728 LayerNorm affine scalars update. Every query starts from the original parameters and a fresh optimizer. Dropout and all modules remain in evaluation mode. Encoders, fusion, initial gating, second-pass inputs, spatial boxes and all other parameters remain fixed. There is no new expert, label feedback, prototype target, learned additional module, or parameter sweep.

The cached boundary is the original final native second-pass TimeDecoder. This is an episodic temporal LayerNorm entropy-minimization adapter, **not canonical BN-Tent, full-network TTA, or an end-to-end rerun of adapted first-pass gating**. The entropy principle and Adam defaults are checked against official Tent; five steps, LN scope and episodic reset are explicit implementation choices fixed before results. Source initial logits/prior/boxes/intervals must match the completed Source outputs bitwise. Original tensor strides are restored for decoder replay. Parameters are restored exactly after each query, and post-reset logits must again match bitwise.

## Measured axes and inference/label separation

Each real point is one video-query pair:

- Horizontal: loss_before - loss_after, nats, the **same optimized loss** evaluated before and after all five updates.
- Vertical: 100 * (official complete vIoU_after - vIoU_before), percentage points.
- Before/after use identical Source spatial trajectories and original two-offset physical-hull/MBR temporal readout.

All 256 predictions and complete loss paths seal before independent CPU official-label scoring. No quality is read during optimization or used to choose steps. The scorer checks original Source scores against previously saved official results and compares the official implementation with an independent dense evaluator.

The primary statistic is the fraction of grounding declines among queries whose surrogate loss improves. Ties in vIoU stay in the denominator; a loss-improvement tolerance1e-6 sensitivity is also reported. Both domains, every selected case, ties and both signs are retained. Mean gains/losses, >5pp severe declines, and bootstrap uncertainty are saved. The plot title remains a question regardless of outcome. No desired number of negative points is an experimental success criterion. A single fixed objective/update scope cannot establish that all TTA methods fail.

## Engineering preflight

The first v1 attempt reproduced full native Source exactly, then failed mandatory no-update replay because cloned tensors did not retain original strides. It stopped before any optimizer step, adapted prediction or GT scoring. Its code, plan, logs and failure state remain preserved. V2 changes only actual seven-input layout restoration; selection, objective, optimizer, update scope and steps are unchanged. The five independent CPU checks cover entropy values/gradients and expanded/strided restoration; see cpu_tests.json. Official source provenance is recorded in official_source_audit.json. Production code/configuration/plan and earlier scientific outcomes remain immutable.

Final paired scatter figures and numerical summaries are stored per domain as hc2/fig1.svg, vidstg/fig1.svg, report.json and paired.csv. ANALYSIS.md records the observed outcome after completion. Additional TimeDecoder-call counters cover replay/optimization/reset, excluding the Source reconstruction internals counted separately as Source_forwards.
