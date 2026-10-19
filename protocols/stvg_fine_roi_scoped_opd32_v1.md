# Fine ROI scoped OPD32: one newly authorized finite experiment

User explicitly resumed only this experiment after pausing overnight research. Keep heartbeat PAUSED and all other branches paused. Old Fine direct gate=false, Coarse ROI gate=false, Tent and later negative results remain. This combination was not run: actual layer3 features plus ordered3x3 ROI feedback plus original independent-center two-round OPD. Hard direct selection is a recorded mechanism control, not a gate that skips actual fits.

## Locked inputs and mechanism

Same exposed NH16/NV16, one query per parent, cross-domain frozen EMA TA-STVG, existing E/finalMBR, retained four DINO references and same reference admission. No new experts/DINO/backbone/feature extraction/head/model training/weights/network backward. Existing layer3 cache is stride16,1024channels, versus old input_proj stride32,256channels. This is not a pure resolution causal comparison: layer semantics and channels differ too.

Teacher uses original vg_tta/stvg_roi_layout_opd32_v1.py: 3x3 bilinear grid at candidate and DINO ROI, feature L2 per bin, matching corresponding bins, weighting common valid support, equal anchor mean. Original q=softmax(log Gaussian proposal prior+reward/.1). No new map, learned projection or score calibration. ROI contains background and is not a mask; score distribution is not calibrated DINO identity probability.

New actual Fine_fixed/Fine_live/Fine_shuffle: per-frame independent2D external state starts0, E width/height and four planned nodes fixed; old alias2x17 noise, scale.25 E width/height, categorical-KL block step.5, final2. Fixed proposal centered0 both rounds; Live second center actual updated state; Shuffle uses original permutation of target features AND masks with templates fixed. Same first Fixed/Live step exactly. No temperature/step/layer search, no GT checkpoint selection. All three arms really fit even when direct is worse; no-reference and constant feedback no-op, preserved in denominator.

## Pure support control

For increasing decoder frame IDs d, a node's conservative interpolation support [previous d,next d] is relevant if right>=MBRstart and left<MBRend. End nodes use own d at outside boundary. All relevant nodes plus necessary boundary neighbors retained; four planned nodes excluded from updates. Others exactly E. This depends only on prediction interval and IDs, never GT presence/event/quality. No triangle/tent multiplier, no change to exploration or retained-block step. This scope exactly matches the old published ROI partition. It is conservative at zero-weight endpoints.

Reuse old Coarse Direct/Fixed/Live/Shuffle by restoring excluded output nodes to E. Per-node independence implies retained-block states are unchanged. Independently verify all normalized coordinate interpolations throughout integer predicted interval unchanged, and final complete-v unchanged. Full old Coarse outputs/metrics retained for transparency. Fine direct uses34 initial candidates pooled from same two17 batches, same pure scope; Fine actual fits use same budget.

## Seal and evaluation

Verify all32 cache receipts/frame IDs/E/references and current registries; no private raw data in public export. Before new GT access, seal all32 contexts, candidate/feedback/gradient/state paths and9 output arms. Cache existing target exposure cannot be undone. GT only offline scoring/diagnostics; unknowns null, not failures/successes. Official full-v, tIoU, ALL known unobserved actual-frame sIoU with original denominators, interpolation s, negative tails, parent paired10000 bootstrap fixedseed20261018 and strata. Extra in-scope conditional s is explanation only, not replacement main denominator.

Main decision: both domains Fine_live−E full-v and all-unobserved actual-s positive, Fine_live−Coarse_live same endpoints positive, and Fine_live full-v above Fine_fixed/Fine_shuffle/Fine_direct. Report all paired intervals and tail cases; passing is development eligibility only, no auto expansion/promotion. Fine and Fixed gains without Live advantage support teacher feedback, not unique on-policy contribution. Three arms no benefit closes this exact combination, no sweep rescue.

CPU only,4threads, wall cap900s for construction+fits, MemoryMax20GiB/SwapMax1GiB, resource guard host6GiB/disk30GiB. No GPU lease needed because CUDA disabled and no model invoked. Preserve partials and never overwrite old predictions. One bounded stages measure→all32 seal→score; independent CPU numerical audit and existing GPT-6.1 Sol review; publish code/protocol/full anonymous signed and unknown results with verified dual-ref CAS and archive. No new automation. Current/MAIN/PAPER unchanged.
