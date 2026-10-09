# P1 allocator-only recovery004: qualified and resumed

The locked HC2-trained TA-STVG → VidSTG P1 run encountered a CUDA out-of-memory
exception after 12,488 of 30,909 accepted arrivals. Video Swin attention requested
2.14 GiB while 2.01 GiB was available; PyTorch also reported 3.51 GiB of reserved,
unused memory. The original failure, code, scientific/runtime locks and all
12,488 prediction files (3,847,482,880 bytes) are preserved. This is an operational
memory failure, not a grounding score or mathematical-audit failure.

The separately pinned continuation sets only
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It uses the original full
200-frame input for the first missing query, order2/arrival2185/query2685, and
the exact predecessor state. It changes no frame sampling, query, steps,
hyperparameters, Gaussian objective, detached feedback, optimizer, LayerNorm
writeback, output selection or chart-revision003 arithmetic audit. The original
GPU runtime remains byte-identical. HC2's sealed 10,446 arrivals are not rerun.

Six actual GPU fits passed before formal resumption: two complete ten-round
fits for the failed query and original/revised pairs for two predefined ordinary
controls (a no-feedback no-op and an informative ten-round fit). The two complete
failed-query fits are bitwise equal over 334 tensors, 75,076 coordinates and
322 scalar fields. Ordinary controls match their previously sealed fits.
The exact full-frame native capture also matches the existing native input for
this query from the earlier order. No qualification prediction was accepted,
no new DINO call was made and no GT was read.

Root independently re-read the saved qualification and its input receipt,
verified all original 200 frame IDs, and recomputed the complete saved
Gaussian/IoU/gradient/Adam/state arithmetic on CPU. Its audit dictionary matches
exactly. The original failed native capture was not serialized: these checks
establish actual reproducibility and preserved-input equality, not comparison
with lost process memory or an independently reimplemented full decoder
Jacobian. Qualification peak counters were reset inside the fit and are not a
measurement of the entire native encoder's peak usage. Future memory safety
is not established by this bounded repair.

Formal resumption compares the first missing complete fit with the actual GPU
qualification before acceptance. Root then re-read that accepted prediction,
its SHA256/bytes/receipt and complete fit, checked its exact predecessor link,
independent LN writeback and per-query residual/Adam resets, and re-hashed all
12,488 original prediction and receipt files. No original prefix was rewritten.
The saved progress snapshot is 12,688/30,909; later dynamic STATUS takes priority.

This closes only the allocator repair, qualification and missing-suffix
resumption. Both directions and all 41,355 P1 arrivals must globally seal before
GT scoring; actual root efficacy/state/dense/tail/case/figure review, publication
and the original P2–P6 suite remain. EATA and historical queues remain paused.
The newly registered cross-domain Figure 1 diagnostic waits for this P1 GPU
release and does not inspect active P1 prediction payloads.

Public export contains inspectable code and safe counts/hash/scope receipts.
Frames, captions, annotations, weights, raw predictions and fit/optimizer/
gradient tensors remain private. No P1 efficacy or whole-paper completion is
claimed. Readback scripts require the original locked research workspace.

Evidence: artifacts/stvg_opd_paper_hc2_revision_v2/recovery/P1_cuda_memory_004/
{CAPTURE_RECEIPT,REVISION_RUNTIME,CPU_CONTRACTS,GPU_QUALIFICATION,
ROOT_QUALIFICATION_READBACK,FIRST_FORMAL_FIT_BITWISE,ROOT_RESUME_READBACK}.json.
