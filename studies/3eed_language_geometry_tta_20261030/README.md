# 3EED round4: candidate ranking and finite test-time adaptation

[Chinese report](REPORT.md), [independent findings](INDEPENDENT_FINDINGS.md),
[qualitative interpretation without images](QUALITATIVE_REVIEW.md),
[decision](research/round4/DECISION.json).

The official released mixed-platform checkpoint remains frozen. No source-only
training was resumed. The primary study is24development/48confirmation Waymo VAL
frames from12/24 new research scenes; Drone/Quad TRAIN auxiliaries are separate
and may have been used in union pretraining. No unseen-platform transfer claim.

The reference256candidate pool and standard PCA geometry are fixed before labels.
A512parameter query-projection adapter and3center latents were actually updated.
Neither primary is adopted: Query3 equals matched augmentation/direct outputs;
Geometry3 harms task performance despite a decreasing unlabeled objective.
Candidate coverage/rank headroom is a task defect, not proof of usable feedback.

Code retains its original workspace-relative layout and byte-exact sources.
This is an evidence export, not a standalone private-cache installer. Authorized
official data/weights, an upstream3EED checkout and previous compatibility setup
are required. Historical local input mappings/caches are not published; any new
reproduction must rebind paths and create a new registration rather than edit
old frozen locks. JSON mirrors replace local absolute paths and physical scene
names; original and exported digests are mapped in EXPORT_PROVENANCE.json.

Only explicit code/protocols, scalar metrics, signed tails, hash receipts,
synthetic engineering checks and aggregate teacher diagnostics are included.
No original PRIVATE artifact is copied wholesale: scalar projections select
allowed fields and pseudonymize physical scenes. No target boxes/centers/yaws,
per-candidate target-IoU arrays, captions, RGB, points, embeddings, projected pixel
arrays, teacher distributions, local parameter states, model weights, compiled
operators, full vendor trees, paper PDFs or credentials are provided.

The camera correction follows the official Waymo extrinsic/axis convention,
not a transform selected using target labels. Its algebra and FOV checks do not
prove semantic object correspondence. Parser and face-to-box assumptions remain
limited. All sample and scene negative tails are retained in scalar results;
private raw cases remain local. No background training or continuation is queued.
