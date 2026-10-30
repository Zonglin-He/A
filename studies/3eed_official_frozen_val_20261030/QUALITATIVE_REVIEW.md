# Root visual review of twelve fixed strata

The root actually opened all twelve rendered private figures, not only the rendering receipts. Every platform has success, annotated-sameclass-other, localization-boundary and candidate-coverage-failure examples, selected by the pre-score salted-hash rule. None was replaced for appearance. Images/point clouds/captions/GT boxes remain local; this public review contains qualitative observations and scalar results only. Full official is class-assisted; noun is text-only, so those two visual traces alone are not an equal-information ablation. The quantitative full-text/noun comparison uses text-only readouts on matched populations.

| Platform | Stratum | Case | Observation grounded in viewed panels |
|---|---|---|---|
| Waymo | success |case_01|Full selects the nearby target vehicle at IoU.8076; noun selects another vehicle at0, although its own pool contains a.7228 alternative. RGB/BEV support different selected locations.|
| Waymo | annotated sameclass other |case_02|Full selects a different pedestrian, targetIoU0/otherIoU.5794, while noun improves to.3648. The full pool contains a.6184 target alternative. This retained negative counters universal full-caption superiority.|
| Waymo | localization boundary |case_03|Full and target center region overlap, but a yaw0 rectangle crosses the oriented vehicle footprint; source.4899, own-pool oracle.5279, GT-yaw injection.7778. A threshold failure is not automatically an identity failure.|
| Waymo | coverage failure |case_04|Source.4314 and best native candidate.4465 still depict the target vehicle region; selectedGTyaw.7913. The BEV footprint is visibly orientation-limited.|
| Drone | success |case_05|Full source.7434 with pool.8174; noun selects another side of the parking region at0 while own pool.8253. Target at image edge is only partly visible.|
| Drone | annotated sameclass other |case_06|Full targetIoU0/otherIoU.5401, but full native poolbest only.2625. Wrong-instance and geometric coverage tags overlap; cannot assume reranking alone passes.5 here.|
| Drone | localization boundary |case_07|Full.4662 versus pool.5097; selectedGTyaw.7248. Noun selects elsewhere at0, ownpool.4738. Both ranking and geometry can contribute.|
| Drone | coverage failure |case_08|Full.2725/pool.2725; selectedGTyaw.4626 still fails.5. One orientation intervention is insufficient in this retained case.|
| Quad | success |case_09|Full pedestrian.6684 and noun0 despite nounpool.7128, with distinct selected image/BEV locations.|
| Quad | annotated sameclass other |case_10|Both inputs select the same wrong pedestrian region, targetIoU0/otherIoU.6493; fullpool.7616. Caption references include appearance/context not directly observed by XYZ-only input. GT-center injection.8522 is a wrong-instance teleport, not evidence that ordinary center regression solves it.|
| Quad | localization boundary |case_11|Source.4856/pool.6206; selectedGTcenter.7154, but GTyaw lowers IoU to.4730. GT-yaw replacement is not a certified maximum.|
| Quad | coverage failure |case_12|Full picks another spatial region at0, pool.2638; noun.2384/pool.2721. GT-center-only.2720 does not rescue.5. Not every failure is yaw-only.|

RGB overlays, BEV and 3D use declared common coordinates. The source sampled points and indices were regenerated against their sealed hashes. The viewed overlays are spatially plausible; finite matrices/positive-depth checks and algebraic projection agreement do not independently certify sensor synchronization or every annotation. Other-object annotations have no guaranteed exhaustive stable instance inventory: lowtargetIoU plus highsameclassotherIoU is specific geometric evidence, not exact-ID certification for all objects.

Global full/noun Hungarian pool mean matching IoUs are only about.213/.247/.303 (Waymo/Drone/Quad). Its numerical replay adds exactly, but weak proposal correspondence prevents a causal interpretation as pure selection versus geometry. Full-caption gains, wrong-instance errors, unobservable appearance attributes and candidate-oracle headroom are research diagnostics, not proof of an available unsupervised adaptation signal.
