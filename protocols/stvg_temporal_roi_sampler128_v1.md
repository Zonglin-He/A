# Object-conditioned source temporal sampler, HC128, 2026-10-20

Autonomous temporal exploration, spatial outputs/config frozen; no new models,
experts, RGB observations, training or spatial-state optimization. Reuse RC128
exposed development, same caches and source checkpoint. Previous decoder
actionness Full/Fluct on current cohort: best NativeFluct +.634825pp, CI crosses
zero. Object attention prior plus final action head did not establish gain.
This probe changes the trained feedback interface, not that prior's strength.

Official TA-STVG TemporalSampling first adaptive-average-pools the fused motion
feature map across the entire image, then applies two frozen text cross layers
and a foreground head. Its existing logits_f_m uses source event BCE. Rebuild
the identical text and motion tensors from cached temporal decoder memory:
text=all tokens before the final h*w tokens, motion=those final tokens reshaped
to original [T,256,h,w]; text query retains original full description. Do not
read verb/category annotations. NativeGlobal calls the original t_temporal_clas
on the original motion map. An equivalent prepooled native call must be exact.

Other conditions only change spatial pooling, before that same learned
classifier: ValidGlobal averages valid image cells, E_ROI and three frozen
Live_ROI average fractional box/cell-intersection-weighted valid cells;
ReferROI uses existing eight native boxes interpolated at actual feature times;
ShiftedReferROI shifts center horizontally by half-image modulo1. No new
mapping, normalization, resized pixels, learned weights or context-strength
sweep. Each normalized pool sums1. Empty clipped ROI falls back to ValidGlobal.
Full-image ROI must equal ValidGlobal. Padding exclusion is explicit and has a
separate global-valid control, so its effect is not called target semantics.

Use original native endpoint logits as the SAME fixed prior for every condition.
Only foreground evidence changes. Historical Full/Fluct projection rules from
foreground128 remain exactly the same, including .1 endpoint logprior, MAD and
physical-cell integration, peroffset argmax and min/max physical envelope.
No sigmoid threshold or event-probability claim. Original finalMBR stays base.
Eligible: NativeGlobalFull/Fluct and ReferROIFull/Fluct, all reported; primary
ReferROIFluct. E/Own/Shift/ValidGlobal are controls, not perquery switches.
Whole-cohort highest fullv parent seedmean with lexical ties selects exploratory
winner, mean>=2pp and CI lower>0 required for independent confirmation. Not OPD.

Max900s lease, original shared lock/resource floors, first2 noGT cost check.
2048 primary temporal classifier offset forwards plus256 equivalent-pool native
checks, zero Source encoder/decoder or expert calls. Source pretrained state
hash unchanged, old cache/config/teacher/spatial SHA pinned. All128 predictions,
raw foreground/endpoint logits, pooling diagnostics and intervals sealed before
GT. Exact old baseline, official/independent fullv/t/s/boundary/tails and paired
parent CI, all parents retained. Correct pool behavior tests establish geometry,
not semantic correctness. Positive readout still needs confirmation and later
matched Direct/Fixed/Live before attributing any benefit to OPD. All outcomes
archive/public-export with exclusions and remote byte receipt; no registration
or spatial change.
