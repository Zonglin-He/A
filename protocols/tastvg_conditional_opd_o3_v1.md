# O3: Conditional Online OPD (supersedes KNN memory)

Latest user b231c75b replaces the previous O3 KNN proposal and requests temporal
only. KNN CPU results already existed before this steering and are archived as
superseded, not deleted or rerun. S0 RVOS download is paused, no S0 inference.

Question: does a conditional nonlinear student absorb sparse critic preferences
more usefully than O2's global linear ranker, and does reverse-KL help compared
with matched pairwise loss? Use EXACT O2 five16-source homogeneous5% streams,
same order/candidates/hidden/expert caches,20/80 expert arrivals,60 nonexpert.
Four arms: Budgeted Rerank; saved O2 Linear Slow; Conditional Pairwise;
Conditional Reverse-KL. No new baseline trajectory or teacher calls.

Follow the attachment's final concrete768->128->1 architecture. Input is the
same raw [h_start;h_end;mean(h_span)] produced by the query-conditioned multimodal
TA-STVG. Do not concatenate a new g/score vector (would change input to1025), fit
normalization or sweep width. Native score enters output s=native+MLP(phi), lambda1.
ReLU hidden; PyTorch fixed seed20260929 Kaiming first layer; output layer zero.
All initial selections exactly native; Pairwise and Reverse-KL get identical
initial state. Reset model and replay at start of every condition. FP64 CPU.

One plain SGD LR1e-3 per expert position, no momentum/weight decay/scheduler;
frequency and step count equal O2. The added nonlinear parameterization and
replay differ from old Linear, so that contrast is a recipe comparison, not
an isolated architecture effect. Pairwise vs Reverse-KL differs ONLY in loss.
Same strict-pair logistic with1e-12 teacher tie rule for Pairwise.
Reverse-KL=sum p*(log p-log q), p=softmax(s/tau_S), q=softmax(r/tau_E), both
fixed tau=1.0, no teacher score calibration, clipping or temperature search.
Do not claim reverse-KL removes calibration dependence. MLP has98,561 params.

Replay: up to4 most recent PRIOR expert-labeled candidate sets; current loss +
1.0*mean(past losses), no sampling. Emit expert-best output BEFORE updating,
then append current set, evict oldest if needed. Only4 expert arrivals/stream
means maximum past set count3: capacity4 eviction is not exercised here.
Nonexpert: use arriving student, no teacher access/no write. Historical feature
sets are frozen proposals of the student backbone; describe as distillation on
student-generated fixed candidate support, not full-policy regenerated rollouts.
No GT involved in replay or updates. Cache tensors detached.

All new online predictions/state chains seal first, independently reconstruct
both MLP forward/SGD gradients in NumPy, then read O2 cached GT-derived candidate
metrics and old Linear choices. No new GT files, new backbone or expert, GPU0.
Primary Online−Budgeted nonexpert t/v per condition then equal-five macro;
secondary Reverse-KL−Pairwise, current loss and replay loss, changes/harm tails,
old Linear, all-arrival scores. Parent-average five conditions before paired
bootstrap12 nonexpert parents (all16 secondary),10000 samples seed20260929,
conditional on fixed shared trajectories.60 cells are not60 independent sources.
No novel stable-transfer or universal failure claim from one tiny development
screen. No numeric equivalence margin is specified, so similar scores do not
prove equivalence or justify promotion by narrative preference alone.

Resource: CPU <=10min, <=1GiB state artifacts; no hyperparameter search or extra
stream. Final independent scalar summary audit,3 CPU contracts, archive current/
candidate/update log, sanitized GitHub publication and remote bytes/SHA readback.
Production unchanged. S0 waits for later continuation; no automatic temporal rescue.
