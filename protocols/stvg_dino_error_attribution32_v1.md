# Saved DINO reference / trajectory error attribution

Status: completed CPU-only diagnosis of previously sealed HC16 + Vid16 development
sources. Baseline evidence commit: `8968cce0b2ccd9c3994eeb35b33621117ab54b82`.

Question: do harmful updates coincide with wrong-object/role or extent errors in
admitted DINO references, or with degradation outside reliable supervision?
Both can coexist. No new model, Teacher call, optimizer run, identity model,
critic, full stream or hyperparameter search is authorized by this diagnosis.

1. Keep all32 sources and all signs/unknowns. Direct: Source versus final episodic
   Direct under fixed Native time. OPD: inherited Before versus final P7-A under
   fixed projection time. Do not interpret their difference as an objective-only
   comparison, or attribute inherited errors to the current Query update.
2. Verify saved payload receipts and recompute official metrics. Partition exact
   annotated frames into admitted observation frames, unobserved native Decoder
   frames, and benchmark interpolation frames. Their signed contributions must
   exactly sum to each source's vIoU and dense-sIoU delta. Keep the final temporal
   support fixed in each contrast.
3. Compare admitted reference geometry only where exact GT exists. A missing box
   is unknown; distinguish outside-event frames from sparse annotation holes.
   Geometry thresholds do not establish object identity.
4. Actually inspect all32 private panels: four planned observation positions and
   up to two largest-absolute-change unobserved native positions per method.
   Verify original video SHA, domain-specific frame clock and original DINO RGB
   hashes. Read original full Query privately. Identity/role labels require pixels
   and target evidence, not low IoU alone. Keep merged objects, partial extent,
   annotation disagreement and semantic uncertainty separate. These judgments are
   explicitly model-assisted; they are not independent human semantic annotation.
5. Inspect saved OPD rounds for observed-geometry improvement with unobserved
   native degradation. IoU>=0.5 is a disclosed geometric diagnostic, never a new
   online acceptance or GT routing rule. Preserve positive final outcomes even
   when individual rounds show interference.
6. Recompute critic time-versus-space oracle headroom and same-time false rejection
   counts. Shared-source pairs are dependent. Oracle choices remain post-GT
   diagnostics and must not be deployed as a selector.
7. Export anonymous metrics, review judgments, uncertainty and the implementation.
   Exclude frames, original Query, media, GT/reference coordinates, tensors,
   weights, credentials and personal conversation. Verify remote content and both
   refs, then archive the receipt. Do not change CURRENT/MAIN/PAPER, resume paused
   Qwen/full queues, or close the separate512-window semantic panel.

Entrypoints, in order (new immutable artifact directory required for reruns):

```bash
.conda/tubedetr/bin/python -B scripts/analyze_stvg_dino_error_attribution32_v1.py prepare
.conda/tubedetr/bin/python -B scripts/analyze_stvg_dino_error_attribution32_v1.py analyze --render
.conda/tubedetr/bin/python -B scripts/audit_stvg_dino_error_attribution32_v1.py
```

Actual visual judgments and original RGB binding are independent review steps,
not fabricated by an automatic label generator. The export script requires their
saved receipts:

```bash
.conda/tubedetr/bin/python -B scripts/report_stvg_dino_error_attribution32_v1.py
```

Decision: keep the fixed critic stopped. The HC negative cases support auditing
role/reference reliability before another architecture. Sparse-update interference
also exists, while the Vid extent example prevents an identity-only explanation.
No causal reference-removal or propagation intervention was run. Any follow-up
must be separately authorized and matched; the diagnosis itself does not start it.
