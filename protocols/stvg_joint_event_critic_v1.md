# Conditional tube update: finite source-supervised pilot

Question: can a source-trained frozen visual event critic identify the actual
consequences of a sparse DINO-driven Student spatial correction across domains?
This is an additional source-supervised preparation setting, not training-free TTA.

The restored historical DINO P7-A remains CURRENT, MAIN and PAPER. Qwen/official
full evaluation and all old queues remain paused. No tracker, new MLLM, diffusion,
Actionness, cross-query LN writeback, Prompt or hyperparameter search is added.

## Fixed data and adaptation

- Targets: the same historical exposed 16 HC2 + 16 VidSTG parents used by the
  DINO restoration pilot, all denominators retained; not fresh confirmation.
- Sources: 48 official-train parents per direction, chosen by fixed parent SHA
  order from the existing 64-parent source manifests after excluding every
  target pilot parent. First 36 train, last 12 source validation. The two critics
  train in their respective Student source domains; target labels never train
  either critic or select its checkpoint. These source pools were used in old
  development work; no claim of globally untouched source validation.
- Source Student, original frame grids/two offsets, same frozen tiny DINO,
  Native Uniform4 observations/admission/Top1 and 1792 interface unchanged.
- Direct correction: original 5*L1 + 2*(1-GIoU), Adam betas(.9,.999), eps1e-8,
  HC .01/40 steps and Vid .03/10 steps, matching restored P7-A update budgets.
  Source candidates use the destination direction's same LR/steps, so the
  critic sees that correction protocol before transfer. Reset Query/LN/Adam for
  every Query; final iterate, no best-GT or LN carry.
- B0 and B1 are complete actual Student decoder readouts, with both decoder
  passes recomputed after direct adaptation. They are not DINO interpolation.
- Three times: Native original envelope first, then highest Source-native legal
  interval mass from the equal offset mixture, with different Uniform4 frame
  sets. Duplicate physical intervals are summed. No action scores or Expand/
  Contract rules. Six hypotheses have a fixed uniform prior; joint argmax of
  critic scores is equivalent to q proportional to exp(C) with tau=1.

## Critic, locked before any new target diagnostic GT

Frozen pre-adaptation source appearance/motion grids, ROIAlign2x2 and full-frame
context, actual full-Query contextual tokens, box coordinates and physical-time
positions are inputs. All sampled frames outside the proposed event remain.
Only ROI sampling intersects image boundaries; output boxes are never clamped.
Per-frame 1033 -> 64, full-Query cross attention, two Transformer64/4-head/FF128
layers, inside/context pooling and 128->64->1 head. Dropout0, seed20261017.
Same-Query strict quality pairs use softplus(-(Cbetter-Cworse)); true ties do not
become artificial negatives. AdamW .001/weight_decay .0001, batch4 Queries,
80 epochs, source-validation pairwise loss selects one checkpoint (ties earliest).
No target-based architecture, seed, epoch, temperature or threshold selection.
Source GT labels candidates only after source model outputs seal; raw GT boxes,
event masks and annotation IDs are never network inputs.

## Order, diagnostics and stopping

1. Freeze this implementation/data/algorithm; generate all target direct tubes,
   features and time candidates before any new target labels. Post-seal oracle
   diagnoses whether the six candidates offer real headroom. Oracle is an
   explicitly labelled target-development diagnostic, never an online method.
2. If each direction has at least one genuinely better full-tube candidate,
   generate the source candidates, source labels, train and freeze both critics.
   Otherwise stop this pilot before training. This checks opportunity, not a
   statistical proof that the candidate family is good.
3. Score all target hypotheses without target GT, seal selections, then evaluate
   Frozen/Native, Direct+Native, Direct+conditional time, joint state/time,
   independent-maxima diagnostic and oracle. Report all signed per-parent deltas,
   intervals, bad updates, DINO agreement vs vIoU, ranking, negative tails and
   uncertainty. Independent maxima is only an error-mode diagnostic, not method.
4. Advance to a matched same-critic OPD comparison only if source-heldout pair
   accuracy and target strict-pair accuracy exceed chance in both directions and
   joint selection improves Direct+Native mean vIoU in both. This is a resource
   gate for this fixed pilot, not a universal significance/validity criterion.
   Failure stops before an online stream or full official evaluation. Positive
   source-training fits alone cannot pass. If advanced, OPD gets the same DINO,
   episodic1792, domain LR/steps and interval set; its trajectories must also be
   frozen and evaluated by this same critic. Wall compute must be reported,
   equal optimizer steps are not an exact compute match.

Resource cap: one serial GPU worker, 4h total pilot wall budget from preparation;
no deadline reset, disk>=30GiB, host available>=6GiB, no signals to other services.
Partial outputs preserved on failure/cap; no automatic scientific retries or
source/target substitution based on GT. Immutable per-Query receipts, complete
GT barriers, source/teacher hash checks, actual pixel inspection, anonymous
positive/negative/neutral/unknown export and verified GitHub sync are required.
