# Reproduction and scope

R131 is one fixed-config expansion of the R129/R130 original-radius unguarded
partial_F recipe. It does not rewrite their historical gates or adopt a new loss.
Own-code dependency: immutable R130 c51169c019131afab7a4b7983884387b3244df98,
studies/3eed_partial_label_fixedS_shuffle_radius_validation_20261117, and the
R129/R127/R123/R97 dependencies documented there. Upstream native 3EED:
https://github.com/worldbench/3EED. Private licensed data, source/prepared weights,
source-A frozen context caches and prior geometry are required, not republished.

Order: content_inventory131 -> select_content131 -> audit_groups131 -> lock
inputs/design -> register131 -> lock model -> run131 --engineering -> lock
execution -> run131 formal -> seal every arm -> audit131 -> score131 ->
audit_scalars131 -> analyze131 -> report131 -> export131 -> audit_public131.
Models and CPU scoring use experiment_guard_round131.py, one exclusive GPU.
select131 is retained only as an allocation/hash utility and failed-attempt
implementation; it is not the selected final input design.

Registration attempts are fixed before gates; UNKNOWN/no-B/no-scope/no-update
cases remain. Complete F_AB matching uses frozen initial-A late/residual context.
No C is generated and no unfitted-view transfer improvement is claimed. Direct
partial readout uses precisely the same frozen candidate sets; wrong labels
preserve S, group-size-bin label counts and budget, not semantic mass/q0. Identity
controls are marked and effective-only results separately reported.

Content grouping uses exact raw point-file hashes, excludes old development A/B/C
and protected-content groups, and includes all aliases/captions/both sources.
Primary inverse-group inclusion weights are fixed by recording population counts.
All target VAL had historical exposure; this is ordinary fixed-config expansion,
not blind independent testing or new-scene generalization. One Drone recording
cannot support a scene-population CI. FINAL has no new inference or scoring.

Task labels enter only after all predictions and states are sealed. Good/bad
coverage and internal ordering are post-seal explanations, never deployment
labels or per-sample/source parameter choices. Every row remains in the results.
Original point-content and per-directory frame dependence is exposed through
anonymous CONTENT/FRAME/CLUSTER table-position labels across source directions.

Export contains only own code/config/anonymous scalars; no raw descriptions,
original identifiers, query IDs, media, coordinates, support arrays, features,
weights, credentials or third-party snapshots. The cancelled path-only attempt
remains documented and retained privately, not silently overwritten.
