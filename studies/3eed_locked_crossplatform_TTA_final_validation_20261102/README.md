# R46 — locked cross-platform grounding TTA evaluation

The predeclared primary R30 and secondary R24 do not improve overall frozen
source performance on this locked384-observation pool. Balanced mIoU source
14.973844%, R3014.802516%(-.171328pp), R2414.917954%(-.055891pp).
R30 is+.163085pp above its all-A/zero-U no-update memory; R24+currentG is
+.158719pp above direct currentG, but each system remains below bare source.
These descriptive positive contrasts are retained; no primary reselection.
Direct currentG itself is14.670101%, below source. SAR's+.000487pp changes
neither query choices nor Acc25/50; it is not a substantive task gain.

REPORT.md contains all6 frozen-transfer systems,6 matched currentG counterparts,
and12 episodic controls in separate tables. ENGINEERING_REPORT.md explains
the exact source/ports and fresh-frame qualification. OVERNIGHT_REPORT.md
summarizes all completed mechanism iterations with public evidence links.

All9216IoUs and37622ordered negative comparisons are exported, with new
anonymous F0001..F0384 row aliases and anonymous scenes; no private linking map.
The negative CSV uses lossless A01..A24 method IDs: see ARM_DICTIONARY.json.
Those IDs label methods, not object/query IDs. All384source-pool Top1/5/10 and
256oracle diagnostic rows,1536method update/skip rows,384contract flags,
768head arithmetic and1152memory audit scalar rows are included.
TopK/oracle are GT-assisted offline diagnostics, not deployed results; source
queries can duplicate instances, and this bound excludes external new boxes.

Methods and hyperparameters were locked before reading this pool. All24outputs
were sealed, independently audited, then scored once under separate label
authorization. A second reviewer recomputed saved scalar aggregates only.
Historical full-validation exposure and1Drone/3Quad existing scenes remain
disclosed; this is locked new-content evaluation, not independent new-scene
gold confirmation. No cross-scene CI or postscore method change is claimed.

Final execution/audit/score succeeded in one pass. Joint24output collection
cost17minutes includes model load,CPU preparation,persistence and closure;
it is not per-method deployment latency. Separate oldA2 engineering I/O
failures and bounded raw reuse are preserved as anonymous history.

Only own source,anonymous protocol/scalars/flags and provenance are distributed.
No rawUIDs/oldaliases/captions/images/pointclouds/boxes/masks/token tensors,
query indices/features/weights/raw answers, licensed cohort registration or
thirdparty source snapshots are included. Overlay own code on the prior
published chain and provide licensed private assets to reproduce the run.
