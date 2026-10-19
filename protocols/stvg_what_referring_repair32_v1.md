# What-role referring-expression coverage repair32 v1

Locked before new DINO outputs and new GT evaluation. Existing exposed NH16/NV16,
32 distinct parent videos. Only NV003/NV006/NV009/NV013/NV015 are changed.
Question: can category-free role-preserving conditioning let the existing frozen
DINO produce useful references for queries previously skipped before invocation?
This is coverage repair, not OPD, and is not independent confirmation.

One deterministic prompt: `the object that ... .`, span [4,10] on the inserted
`object`. Unknown wh subject stays subject; object/oblique clauses undo auxiliary
inversion using cached dependency roles, retaining do-support and all source
words including prepositions and negation. Narrow declared parser repair: if the
root is an ADV/ADP particle and the immediately preceding NOUN is attached to the
root's subject, treat that preceding lexical token as predicate for auxiliary
placement. No target category inferred; no prompt alternatives or post-score edits.
Only unresolved_wh_category plus empty legacy phrase may enter this adapter.
Normal branches return an exact deep copy of their original parses.

Existing Grounding DINO tiny snapshot/checksum and processor, same ContextView,
explicit-span scoring, OLD NMS top3, .35 score/.05 distinct margin, same E estimator,
source calibration, beta1/16, outlier handling. Exactly four ORIGINAL Native-I0
Uniform4 positions per affected query (20 new image forwards total). E uses the
original Source trajectory once, not E plus a duplicated correction. Existing
finalMBR/time/interpolation fixed. Other27 parents reuse original E unchanged.
Versioned condition cache includes prompt/span/pixels and source runtime identities;
old negative observations cannot match. No new STVG forward, backbone, model,
training, OPD, observation location, frame budget, temporal tuning or threshold.

All32 predictions sealed and hashed before any new target GT access. Offline
metrics: full tube vIoU, tIoU, observed/unobserved actual sIoU, all parents including
no support/unknown GT/zero temporal overlap, paired bootstrap10000 seed20261018,
query=parent. Report primary NV16 and secondary affected5 plus HC16 no-op checks;
full signed per-query results and severe tails. New references: four receipts per
query, accepted/rejected reasons, known-anchor IoU versus E and retained-pool upper
bound (offline diagnostic only); missing GT stays unknown. Success requires useful
real references and actual E gain, not completion of calls or detection confidence.

Resource cap: shared nonblocking gpu.lock, exclusive own lease, <=900 allocation
seconds, <=3600 wall seconds; disk/RAM guard and >=4GiB free GPU, own unit memory
high16GiB/max20GiB/swap1GiB. Stop own worker on cap; do not signal others. Frozen
DINO state verified before/after. Preserve all partial failures, charge retries.
Raw prompts/captions/dependency tokens/identifiers/coordinates/caches/media/weights
stay private. Public code/protocol/sanitized results and costs/audits only. No
production registration changes or automatic follow-up.
