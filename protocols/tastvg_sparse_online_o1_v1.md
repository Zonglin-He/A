# O1: Sparse-Critic Online Transfer

User attachment 4d43b8b6 authorizes ONE 32-source stream and four arms only.
Question: do past expert preferences help future non-expert arrivals?
Prior C3: rerank +3.2666pp v versus Frozen; current-query one-step OPD did not
beat rerank. Historical spatial online state exists; this is a different temporal
candidate ranking state, not a claim that slow/fast itself is new.

Original exposed VidSTG32 parents, one query/source, each source occurs once.
Conditions ordered [frame_drop,frame_freeze,motion_blur,occlusion,exposure], each
[1,5,10]. Select index int(SHA256('O1-condition-20260929|'+source),16)%15;
order ascending SHA256('O1-order-20260929|'+source), tie by source string.
Use existing seed0 C0.5 full-video random burst, unchanged pixels/severity.
No GT/query/outcome in condition or order. No fresh-confirmation claim.

Same Vid-source TA-STVG and UniversalVTG best+PE checkpoints as C3. Same student
max8 layer/legal-span candidates and confidence*proposal-IoU expert ranking.
Capture final second-pass temporal hidden states before temp_embed; interleave
both offsets onto the original observation grid. phi=[h_start;h_end;mean h_span]
is raw 768D, no learned/fitted normalization or extra features.
Native base score is one half of the largest sum of final per-offset legal-span
log probabilities whose physical envelope equals the candidate. This preserves
exact native argmax with native-first ties at w=b=0. A nearest-endpoint score on
an envelope is not guaranteed to do so. No bias added in favor of native.

Persistent state w(768),b(1), both initially zero; FP64 CPU rank arithmetic.
S=ell_native+phi@w+b. Only w,b persist via existing O-all arrival semantics.
Pairwise logistic loss on every strict teacher preference, same 1e-12 tie epsilon
as C3; SGD LR=.001, one update per expert arrival, no momentum/weight decay,
no scheduling/tuning, no normalized gradient, no replay/EMA/gate/prototype.
LR is a fixed first implementation choice, not selected using any outcomes.
Bias has exactly zero gradient because it cancels in every score difference.
All model weights, H, head and decoder remain frozen.

One-based positions 1,5,9,13,17,21,25,29 are the eight expert arrivals.
Frozen: always native. Budgeted Rerank: expert-best there, native elsewhere.
Online Slow-Fast: SAME expert-best output there, then write one SGD update;
other positions use arriving state, no expert read and no update. No current-query
update before output. Save all arrival/after state tensors and hashes, preceding
record hash, selection and write diagnostics. State resets only before stream.
Full Rerank is a separate 32-call reference, not a guaranteed performance upper
bound. Its 24 non-expert teacher outputs are read/computed ONLY AFTER online
predictions/state chain are sealed. Existing expert caches count as logical calls;
report both logical budget and actual newly computed/reused evidence. Historical
features/GT exposure is acknowledged; no GT in any model or adapter update.

Primary: Online minus Budgeted Rerank on 24 non-expert positions (tIoU/vIoU).
Secondary: all32 four-arm mean, 8 expert positions, chronological prefix/block
readout, losses/state norms and harms; blocks differ in source composition and
are not a causal state-pollution test. Paired bootstrap10000 seed20260929 may be
reported descriptively conditional on this realized stream, not as uncertainty
across online trajectories. No extra seeds/orders or follow-on experiment.
All outputs, including Full Rerank, sealed before GT scoring; only original32 GT.
No new benchmark severity, C0.6, spatial, reliability or current-query latent tuning.

Serial single GPU capture/teacher capped cumulatively 1800s; tiny online loop CPU.
8GiB free floor and <=2GiB new artifacts. Verify frozen outputs, native zero-state
identity, state chronology, expert-access split, SGD and two task metric routines.
Archive complete positive/negative outcomes, publish sanitized code/results and
verify GitHub remote. Production CURRENT unchanged.
