# Fixed-DINO shared target-memory adaptation32

Research candidate, not deployed Current and not OPD. Same exposed NH16/NV16 cohort as 520484a5, one query per parent, original opposite-source TA-STVG checkpoints. All weights including LN/heads remain frozen. Only a disposable 256-dimensional offset is optimized, reset per query.

Insert after fused-memory normalization into the text slice read by both second-pass spatial and temporal decoders. Select the prepended parsed subject tokens and a uniquely occurring original referent noun via the actual RoBERTa fast-tokenizer offsets. No action/relation tokens or special tokens are directly changed. No generic cosine alignment assumptions. First-pass native frame routing, global anchor/context summaries and initial decoder queries stay fixed, preserving the original native prefix. Thus this experiment is not an end-to-end actionness-free implementation.

E candidate selection and rejection are frozen and reused exactly. Its selected original DINO boxes supervise only their original positions; no GT visibility filtering, no event-positive pseudo labels and no conditional DINO calls. Retain E on the output: centers E+(B(delta)-Source), sizes E*exp(log B(delta)-log Source). Zero offset must reproduce original boxes/logits/MBR exactly.

Two Adam steps, lr=0.01 in both domains (existing HC query-offset rate reused once, no search), betas=(0.9,0.999), eps=1e-8, no decay. Loss is mean over E-selected positions of 5*L1(cxcywh)+2*(1-GIoU), plus 0.001*mean(delta^2). These fixed engineering choices are not DSE-VTG hyperparameter replication. No support or no safely mapped target noun means no-op; no noun is invented. Read step2 only; no GT selection, line search or extra safety gating. Decoder reference detach remains unchanged.

One fit yields four readouts: E/original MBR; E/adapted MBR (primary temporal test); adapted E-retaining boxes/original MBR; adapted boxes/adapted MBR. Each temporal MBR sums the full two-offset physical-interval distribution before expected-tIoU decision. All32 predictions and distributions seal before any target-GT scoring. GT only offline final evaluation.

Report official full vIoU/tIoU/sIoU, actual observed/unobserved decoder-node sIoU (unknown stays null), start/end signed and absolute errors, duration ratio, negative tails, query-paired bootstrap and physical-distribution changes. Query=parent here. Keep both directions and all failures/no-ops. Two source checkpoints and all raw data hashes remain bound; public export excludes query text, parent IDs, boxes/GT/media/tensors/weights.

Finite limit: 900 seconds GPU allocation and 3600 seconds total wall time including engineering failures; no new model, teacher, cohort, scan, historical replay or automatic OPD. Source prefixes may be recaptured once per query and cached. Reopening differs from spatial-only QR: the input is shared by both decoders and the four readouts isolate actual temporal changes.

Conceptual reference: DSE-VTG https://arxiv.org/html/2609.08850v1 . Only disposable query adaptation is borrowed; its aligned-embedding cosine pseudo labels are not used.
