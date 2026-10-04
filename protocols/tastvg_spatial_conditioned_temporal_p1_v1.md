# P1: Soft spatial conditioning of frozen temporal evidence

User-authorized question: does full-context-preserving spatial guidance improve
the temporal teacher, compared with the failed hard A-box crop implementation?
This is one matched technical-factor experiment, not DTA or a method promotion.

The fixed parent is P0 (39e060b), with the same 288 expert cells: 240 corrupt,
48 clean, from both orders of six conditions on VidSTG/HC-STVG-v2. The design has
32 search and 16 confirmation sources per dataset; actual scheduled expert
sources are Vid 16/8 and HC 14/7. All are historically exposed, not fresh test.
Do not change sources, corruption pixels, expert positions, A pre/post states,
the original media observations, nearest-original-observation 2Hz grid, source
checkpoint, query text features, or raw confidence top1 rule.

## One locked intervention

Keep the complete original image and stock PE-Core-L14-336 transform: bilinear
 squash resize to 336x336, no center crop. The final layer has 577 post-ln_post
tokens (one CLS plus 24x24 patches, width1024). Convert the saved **pre-update**
A pixel-xyxy box to this grid; mask each patch by fractional box intersection.
The frozen learned pooling query, Q/K/V/out, residual MLP, and visual projection
are unchanged. Add log prior to pooling attention logits only:

\[
\alpha=0.5,\qquad h_{CLS}=1,\qquad
h_{patch}=0.5+0.5 M_{patch}.
\]

All patches remain in the complete-frame transformer and pool; background prior
is strictly positive. No raw patch mean, new feature dimension, normalization,
training, token deletion, pixel crop, or parameter update. No alpha sweep. Empty,
invalid or clipped-outside box uses the original global pool. Alpha0 and full
mask explicitly use the original pool, with exact parity verified in smoke.
This is a frozen box-derived attention prior, **not a reproduction of the
jointly trained CoSD static/dynamic architecture**.

Full cached teacher is primary baseline; P0 A_ROI is a cached secondary
comparison, not recomputed. New Soft features can change both raw proposal
support and ordering. An oracle decrease combined with top1 increase does not
prove same-candidate reranking. Do not run GT-ROI or an additional expert.

## Execution and barriers

Freeze code/assets/inputs and existing GT file hashes before inference. Offline
models only, one GPU lease, FP16 stock batch16, no shared asset mutation.
First clean cell per dataset is preselected for no-GT format/resource smoke:
stock vs decomposed global feature exact parity, alpha0/fullmask exact parity,
historic FP16 feature tolerance<=.02, proposal frame error<.1, confidence
error<1e-4 and Full top1 identical. Root must accept smoke before 288 Soft cells.

Reuse identical feature input hashes; preserve unique grid and inverse repeats.
No GT, scored ROWS/SUMMARY, original annotation files, or quality-driven rule
changes during deployment inference. Seal all Soft receipts before temporal
GT scoring. Old Full and crop assets remain immutable. Disk floor8GiB.

CPU scoring reports teacher continuous tIoU, tIoU>.5, disjoint rate, raw support
oracle, Soft−Full, Soft−A_ROI, clean, both orders, severe negative tails and
source concentration. Same-source macro with paired10000 source bootstrap,
seed20261004. No pseudo-independent condition/order bootstrap. Report new
feature relative displacement, mask coverage, calls and worker wall time;
wall time is not GPU kernel time. These are teacher-only results, not final
STVG vIoU or persistent adaptation gains.

Teacher-only GO requires all four corruption panels Soft−Full lower95CI>0.
Otherwise stop this UVTG soft/crop coupling line as a resource decision. Neither
outcome authorizes DTA integration, another strength, crop sweep, expert,
candidate-support change, or old queue. No universal impossibility claim.

## Original-paper support and boundary

CoSD CVPR2023 Eq.(2) uses object-query spatial cross-attention to modulate a
dynamic 3D representation with a residual and LayerNorm. Its Table3 56.1→57.4
m_tIoU is jointly trained HC-STVG-v2 validation; it is not a frozen-plugin gain
guarantee. Original paper:
https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Collaborative_Static_and_Dynamic_Vision-Language_Streams_for_Spatio-Temporal_Video_Grounding_CVPR_2023_paper.html
https://zanglam.github.io/files/Collaborative_Static_and_Dynamic_Vision-Language_Streams.pdf
An official runnable implementation was not located in the bounded check.

All code, protocol, anonymous results/negative cases, audit and figures must be
publicly synchronized to Zonglin-He/A and remotely verified. Update local
RESEARCH_HISTORY, check/snapshot/check; preserve private media, annotations,
weights, raw caches and personal archive outside public export.
