# v0 execution outcome

The attachment-bound v0 has been implemented and tested. Its complete HC64 pre-adaptation stage ran and was sealed before GT. **The prelocked history-benefit gate failed, so no OPD, FixedKD or Direct parameter adaptation was executed on real queries.** No adapted scores are claimed. The controls below are real untrained propagation/decoding predictions, not new OPD results.

The user specification orders candidate/feedback/history diagnostics before fitting, then says to run lightweight comparisons after those checks pass. Before opening GT, this run made the qualitative history-benefit check executable: MemoryGreedy must not lower either mean full vIoU or unobserved-node sIoU relative to NoMemoryGreedy, and must improve at least one. This is a development feasibility rule using point estimates, not a statistical claim that negative effects are established across all videos. No threshold or method was changed after seeing the diagnostic.

## Completed implementation

- Three candidates from separate reliability-preserving forward/backward predicted-mask components, with original B0 always retained.
- Zero six-weight initialization, original prior0.7, and exact initial greedy B0 output.
- Student features exclude teacher support u. Teacher feedback uses support, connection, bounded memory and bounded geometry, without the student prior.
- Confidence-normalized forward KL plus0.05 prior KL; lr0.1, gradient clip1, memory write at most0.2*c*u.
- Student-state OPD, fixed teacher-state KD and full-gradient soft Direct; maximum3 rounds of2 paths. Updated greedy fixed-score check with full Adam rollback and stop.
- Exact additive-score candidate DP and independent memory/no-memory controls. All final decoding is a real candidate path.

Thirty-two synthetic CPU tests pass, including full Direct finite differences, nonempty Adam rollback, source-feature interface, loss normalization and DP versus brute force. The original70 engineering tests also pass. These tests do not substitute for an actual adapted-method result.

## Completed HC64 evidence

| Setting | Full vIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|
| Initial ERef + Strong |31.0918|52.2680|50.7055|
| Saved geometric OPD reference |31.3868|53.1654|51.2469|
| MemoryGreedy |28.8461|48.1802|47.9424|
| NoMemoryGreedy |28.9642|47.5745|48.3152|
| EvidenceDP |29.3919|47.4528|49.1405|

The candidate oracle does contain useful alternatives: mean fixed-unobserved-node IoU gain is5.4833pp, 95% parent interval[3.3802,8.0853];248/1365 nodes have a valid alternative more than5pp better than B0. The denominator includes545 original-only fallback nodes. GT nodewise candidate readout gives34.3111% complete vIoU, +3.2193pp over B0. This is an offline availability diagnostic, not executable output, not a guarantee of achievable gain and not an exact global trajectory upper bound.

Feedback's macro pair accuracy is54.9747%, interval[49.4629,60.5565]; macro Spearman0.06137, interval[-0.05592,0.18449]. These pass the minimal prelocked point-estimate screen but do not establish reliable feedback quality. The matching confidence and region score remain uncalibrated proxies.

MemoryGreedy minus NoMemoryGreedy is -0.1181pp full vIoU, interval[-0.3554,+0.0604], and -0.3727pp unobserved sIoU, interval[-0.9755,+0.0424]. Neither whole-panel interval establishes a global negative effect, but the point estimates do not satisfy the required net-benefit condition. At nodes farther than half the median anchor gap from an observation, the unobserved difference is -0.9762pp, interval[-2.5708,-0.0436]. Every fallback/unchanged node with GT remains in these denominators.

The fixed-score readback is also material: EvidenceDP raises the proxy score in62 queries, while38 of those queries have lower true vIoU;15 of64 have drops greater than5pp relative to B0. A fixed-score non-decrease check alone therefore cannot certify localization improvement. This is a mismatch in this candidate/evidence construction; it is not a measured failure of the unrun OPD adaptation.

## Scope and costs

All64 parents are the same exposed HC2 development panel, one query each. There was no online GT, per-query selection, new observation, RGB read or heavy-network/expert/decoder call. The teacher masks are ReferDINO predictions, not ground-truth segmentations. Both offsets and Strong temporal intervals were preserved; all reported sIoU uses fixed actual nodes with GT.

The stage1 prediction/seal run took18.45s. Mean cached costs per query are0.0949s fixed evidence construction,0.0616s component-bank construction and0.0358s for the initial diagnostic paths/controls. The complete query loop median is0.2307s and P90 is0.2539s. These do not include generating Source/Strong/ReferDINO caches. End-to-end latency of an adapted v0 pipeline was not measured because the prototype stopped before fitting; no cached timing is substituted for it.

The source/config/input fingerprint is38b838b3d74cea3aacccc5b300c59e0ac96b277ecff0f83c44f4b674b06af5d6. All64 assets and original baselines were revalidated before and after GT;384 official/independent dense cross-checks agree to below1e-15. The full signed anonymous results, ranking coverage, bootstrap intervals and distance bins are in diagnostics.json. control_audit.json is a readback of sealed paths and saved metrics, without new rollouts or labels.

The decision is **stop before adaptation, preserve this completed v0 screen**, without a post-GT parameter sweep or an expansion to other backbones. The previous registered single-ReferDINO initialization + geometric spatial OPD + Strong method is unchanged. Formal VidSTG and CG remain paused. Source, data, masks, tensors, private identifiers and weights remain local.
