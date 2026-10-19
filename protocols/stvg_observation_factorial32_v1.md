# S / T / S×T observation capability32, 2026-10-19

User authorizes all three candidates after the analysis-only pause. This finite
round actually evaluates spatial RGB re-detection, temporal full-coverage dense
reading, and their factorial joint output. It is NOT three completed OPD methods.
No Fixed/Live distillation is silently claimed. The question here is whether the
existing models gain usable information by observing differently. No branch is
skipped because another branch is negative. No automatic follow-up or training.

NH16/NV16 existing exposed development parents, one query per parent. Existing
frozen cross-domain TA-STVG EMA (HC target/Vid source, Vid target/HC2 source) and
GroundingDINO-T checkpoint unchanged. Original E, final-MBR, four observation
positions, text/target-span parser, admission/NMS and source-calibrated E estimator
remain fixed. No repaired what prompts, target labels, new experts/backbone/head.

S: around each ORIGINAL E box at the original four slots, take a 2x width/height
RGB context, floor top-left/ceil bottom-right, intersect image. Empty raster is
explicit no-query fallback. Use native DINO processor 800/1333, same original
context/span or original fallback phrase, original top3/NMS/target gate. Lift
actual detected boxes to original normalized coordinates. Accepted crop pools
replace that slot's original pool; rejected/unsupported slots retain original
pool. Re-run original E estimator from Native Source, not from E (no double
correction), old scale/cutoff/beta1/16. This fallback rule is prelocked, not GT.
Unresolved queries stay unresolved. This uses detected geometry, NOT crop token
energy/gradient and not fixed-box IoU reward. All raw outputs remain private.

T: union original physical frame IDs with one integer midpoint per adjacent pair,
deduplicate. Same full coverage/endpoints, original two interleaved offsets. New
length <=2N-1; each actual offset must fit original model MAX_VIDEO_LEN, otherwise
stop with unsupported status, never extend/interpolate positional embeddings.
Original text and trained motion/appearance/decoder rerun on true RGB frames;
no old-feature interpolation/GT crop or actionness label. Raw temporal logits
FP32 -> existing probabilities -> independent-CDF exact final-envelope mass ->
same final_mbr. Only <1e-12 negative CDF roundoff/illegal mass may be zeroed and
normalized; total illegal/normalization CDF roundoff tolerance is1e-10, individual
negative cells must be above-1e-12; save L1 correction. No cross-grid logit interpolation. E spatial
nodes unchanged; dense TA box output is never substituted.

Outputs: E=(B_E,I_E), S=(B_crop,I_E), T=(B_E,I_dense), ST=(B_crop,I_dense).
Also GridOnly=(B_E,I_grid): original posterior mass embedded exactly in dense
decision grid, then same MBR, to distinguish new decision support from a changed
dense-run posterior. This does not isolate new RGB evidence from sequence length,
index position embeddings or the changed physical duration of motion windows.
ST is no-extra-call factorial composition, NOT a joint conditional
adaptation loop. Interaction Δv_ST−Δv_S−Δv_T is metric complementarity only.

All32 five predictions, input receipts, detector outputs, probabilities and
intervals sealed before any NEW GT read/score. Formal FH/FV64 excluded. Main all
parents, no-ref/unknown/zero-overlap included. Corrected full v/t/s, original
unobserved actual s, endpoint/length, signed tails and parent paired CI. Temporal
only actual s must exactly match E. GT only postseal diagnostics of actual
admitted detector choices, never pool-oracle output or crop selection.

Budget: shared original nonblocking gpu.lock, total newly held GPU <=900 seconds,
first NH001/NV001 timing pilots are INCLUDED in all32; actual cap, no speed claim.
An owned systemd unit has RuntimeMaxSec880/TimeoutStopSec5/KillModecontrol-group,
providing a hard stop inside900s even during an individual GPU call; it cannot
signal unrelated services. Plan deadline3600s includes preparation, the owned
model worker has the stricter880s wall cap. CPU4threads, memory high16/max20GiB/swap1GiB, disk>=30GiB,
hostavailable>=6GiB/GPUfree>=4GiB before calls, no signal to unrelated services.
Max128 new DINO image calls (less for old unresolved conditions),32 dense TA
queries/64 offset forwards; no old Source/history replay or parameter updates.
Cost includes failures and loading; raw decode/metrics/audit separate. If early
support/resource/numerical checks fail, keep all partial outputs and do not call
engineering noncompletion a scientific negative. No configuration search.

If observation outputs show no useful net advantage, close these tested recipes,
not all possible observation-conditioned OPD. If an advantage exists, only a
separately locked same-evidence/direct/Fixed/Live experiment can establish OPD;
current capability results cannot. 2–3pp target means complete cohort vIoU, not
anchor IoU, oracle, selected subgroup or best step. Publish code/configuration,
all signed/unknown results and negative limitations with remote receipts. Never
publish media, original text/IDs, GT boxes, raw tensors, weights or secrets.
