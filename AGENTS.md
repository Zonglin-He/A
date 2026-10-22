# Main method

Maintain only TA-STVG BoxFixed + Global6 + original physical MBR. Read docs/history.md,
docs/method.md and docs/archive.md before edits. configs/default.json is authoritative.
Do not add research modes, OPD, training, extra teachers, 448, conditional replay or
parameter sweeps without explicit user authorization. Original TA-STVG and ReferDINO
sources are necessary frozen dependencies, not experimental alternatives to remove.

Keep complete native teacher box extents, one expert/eight observations, original ERef
calibration/gamma, input decoding, offset/padding mappings, fixed score and MBR. Never
read GT in inference. Score only sealed predictions, independently on CPU. Cache reuse
must be content bound; no fingerprint exception or cross-code resume.

The completed full job is archived in the separate immutable stvg-opd checkout. Do not edit,
pull, reset, rename or delete its frozen files or mutate its sealed full outputs. Its original plan
fingerprint is 2ebd134dfd055c8e17e01859b706d56d187b15b8907a8129c49669539d84ea6e.
This maintained source tree has a different source fingerprint and must not resume
that job. Both domains have sealed/scored; their public summaries are imported here. Preserve all private saved predictions and old source snapshots.

Run CPU engineering tests and appropriate compatibility checks. No new GPU experiments
or parameter changes are authorized by the cleanup. Record material changes in
 docs/history.md, then python tools/archive.py check, snapshot, check. These are source
and evidence checks, not model experiments.

Publish only an explicit clean source/evidence tree to Zonglin-He/A main and research,
using fresh expected-head checks, preserving ancestry and the research archive branch.
Verify remote tree and file bytes. Never publish media, query text, GT coordinates,
weights, raw masks/features/states/caches, credentials, environments or local asset links.
Never restore the old broad research export over the clean main branch.

Latest user correction (2026-10-22): exploratory HC full is paused_by_user at338.
Never resume it automatically or bypass a failed Source temporal gate by calling
a run exploratory. Continue only authorized temporal development; spatial code,
weights, evidence and Source forward remain fixed. New full requires the frozen
candidate to beat Source on the declared fixed validation. Global6-only improvement
is insufficient. Preserve all paused outputs, receipts and original source snapshots.

Latest temporal admission: require strictly positive Source tIoU gains separately
for GT-duration<=5s and>5s in tuning AND fixed recheck. GT grouping is offline only;
never choose inference parameters from GT duration. V5 MBR3 failed long-event recheck;
no automatic fallback, resume, default upgrade or full launch from its tune success.

Latest explicit user authorization supersedes the prior stop-until-Source-group
gate for this next fixed evaluation: choose relatively best originalGlobal6
(temp1,sixlayer mean,old hull,original MBR) and run TA-STVG-only dual-domain paper
package. Preserve previous failed Source gates. New queue outputs/tastvg_paper_v1
has its own code/plan fingerprints, complete-member lists1902/10303 and same8
expert observations. Reuse prior exact Source/Full, collect all auxiliary arms
once/query. Do not restore old paused v3/formal/CG queues or run other backbones.

Latest explicit user instruction (2026-10-22): pause the TA paper full and execute a bounded
HC experiment on old event-token misranking and learned real temporal-query correction.
Paper state is paused_by_user at Vid1373; HC1902 sealed/scored retained. No automatic
full resume. New isolated tools/output may implement that authorized experiment; keep
all paper-bound48 and archived434 sources, spatial method, seeds/evidence unchanged.
