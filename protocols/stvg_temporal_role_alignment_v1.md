# Fixed question-role temporal alignment ablation

P0 reuses the current actor-centric caption and current alignment. P1 changes only alignment for the 28 registered VidSTG question sources. The other 36 VidSTG declarative sources and all 64 HC-STVG sources retain P0. All sources are exposed development data, one query per source. No independent-confirmation claim is made.

The exact fixed prompts are in `scripts/stvg_temporal_role_alignment002.py`. Query and description are quoted data, never execution instructions. R1 is frozen. Alignment uses the actual next-token A/B/C logits, normalized over those labels, with reward P(A)-P(B). The model loads once for the P1 text-only batch.

There are 22 question sources with four inherited captions each: 88 alignment calls and zero new caption calls. Three inherited captions are nonempty but truncated; the original P0 used them, so P1 preserves their use and flags. Six original-input fallbacks each have three actual candidates. This gives 106 actual candidate slots, 18 unknown slots and six uninstantiated positions in the nominal four-candidate budget. Unknown feedback is not zero and no source is dropped.

Candidate intervals, source captures, 512-parameter Offset-Trust fitting, decoded time mapping and spatial observation clocks remain fixed. The current Offset-Trust implementation is replayed on CPU using saved model captures; it is not a new full-backbone GPU forward or a change to temporal Adam. Twenty-two original P0 fits are checked for exact interval restoration before P1 execution. New P1 temporal fits are independently repeated with the same interface. Native spatial boxes remain unchanged.

All 64 VidSTG predictions and source/probability/CPU-replay readbacks are sealed before official annotation access. Official metrics are checked with the existing independent implementation. Report query-macro vIoU, tIoU and dense spatial IoU, all per-source signed changes, fallback/null evidence, and candidate-pair reward ranking on common valid support. GT-quality ties are counted separately; reward ties receive half credit. No GT candidate selection, Oracle fit or bootstrap is performed.

P2 is one fixed relation-aware caption plus the P1 alignment on the same visual inputs. It activates only if the full 28-question group has known vIoU and tIoU and P1-minus-P0 mean vIoU is positive while mean tIoU is nonnegative. Otherwise no P2 or additional prompt search runs. P1 and P2 are separate from the spatial parameter search, whose original temporal lock is preserved.

Raw captions, queries, videos, boxes, model states, weights, annotation files and private runtime input paths are excluded from public exports. This public export covers the finite P1 comparison, not completion of the ongoing 37-slot spatial coordinate search or promotion to CURRENT/full evaluation.
