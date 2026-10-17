# Fixed original256 experiment: only I2 ranking changes

Authorized by the human after original256 B diagnosis c71208b9. Same exposed
HC128/Vid128 parents and cross-domain TA-STVG Source checkpoints. Frozen R1,
Query-free caption Prompt/processor/clock, support Prompt, eight-frame budget,
160 generation cap, deterministic generation, reward P(A)-P(B), tau1, zero-init
512 endpoint parameters, original joint-state conditional target, Native-grid
trust1, LBFGS50/Armijo and complete Native decode are unchanged. No external
Actionness, new Spatial/Full stage, candidate/Prompt/parameter sweep or CURRENT
promotion. This is exposed development, not another independent confirmation.

Only I2 physical-interval ranking changes: sum the original independent legal
offset-state Native probabilities over all states merging to each physical
interval; select the largest mass with valid, distinct actual uniform8 samples.
Ties use stable row-major physical endpoints. I1 stays actual original Native,
not the marginal mode. The representative state within I2 uses the original
FP32 log-softmax joint ranking and stable latent row-major tie. Its raw joint
score/features still feed the unchanged fitter. Marginalizing the training
target, changing I1 or modifying the loss would be additional interventions and
are explicitly excluded. Native internal frozen action gates stay unchanged.

Retain independent Native merged-posterior MAP with zero parameter updates as a
separate control. Its already observed free decoder gain is not credited to
Teacher/TTA. Five main arms: Frozen, old T2, original B, marginal no-update,
new B with marginal-ranked I2. Original Frozen/T2/B complete predictions are
reused/rescored exactly, no rerun. New hypothesis inputs and control intervals
are sealed before new feedback/fit. No GT file or saved diagnostic scores may
be opened by input preparation, feedback, fitting or integrated readout.

Exact I1 evidence may be reused from original B after source/query/visual SHA,
fps, Prompt, Teacher and clock binding. I2 evidence may be reused only if its
complete latent pair, interval, actual frame IDs, pixel SHA and metadata match.
Otherwise generate actual new interval video input and fresh frozen feedback.
Actual calls/reused calls, two-domain short-prefix qualifications and original
costs are reported separately. No fallback using GT; invalid/missing/truncated/
equal feedback preserves original Native and all256 denominators.

Real optimized parameters must be inserted into the original temporal head and
all full-grid outputs read back exactly; Source spatial boxes unchanged. Raw
Source prefix caches are reused only with original bindings; two historically
absent prefix receipts require fresh frozen forwards from identical RGB, with
Source logits/hidden/boxes/records/Native exact. No model state capture/refit or
Teacher repetition for that engineering route. Preserve all old artifacts.

Prediction/ROOT and no-update controls all sealed before GT evaluation. Report
vIoU, tIoU, dense known-GT sIoU, common Native-time spatial support, R@.3/.5,
paired source t95 intervals and all signed/unknown values separately by domain.
Primary mechanism comparison new B minus old B; separately compare Frozen,
T2 and marginal no-update. No automatic selection or follow-up based on scores.

One serial bounded unit, 4h execution cap, MemoryHigh16GiB/Max24GiB/Swap1GiB,
host available6GiB/GPU free4GiB/disk30GiB live guards. Reuse original offline
environments. Publication is necessary: anonymous actual config, code, complete
signed/unknown results and limits; no texts/media/GT coordinates/weights/cache.
Actually inspect input samples and result chart; verify remote HTTP bytes/SHA/
blob and fetched mode/tree, both refs with fresh CAS, archive check/snapshot/
check. Stop at this finite group; do not revive historical queues.
