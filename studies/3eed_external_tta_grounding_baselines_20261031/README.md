# Real 3EED external grounding TTA baseline ports

[Full report](REPORT.md). Waymo-only100epoch source→Drone/Quad. TENT, SAR and MEMO grounding ports completed real96input episodic inference/update; Reg-TTA3D-inspired preserved NPG/CRR/CBU but uses stated detector-interface substitutes. It is not an original Reg reproduction. Source checkpoint immutable, current predictions sealed before CPU task scoring, all negative cases retained.

Primary A64/B32 are current-input adaptation then clean prediction, separately reported; not prior A→B transfer. Separate32input Reg online diagnostic and32new-input official optimizer/LN-scope diagnostic completed. No continual-main protocol chosen, CoTTA/MOS not executed. Official MEMO RVT AdamW recipe is a post-primary fixed diagnostic, not a cherry-picked replacement. Quad sample-mean gains do not survive equal-scene weighting and do not beat same-information augmentation-only. No new method adopted.

1472 anonymous task-arm rows preserve1184primary+96online+128Cdiagnostic+64omitted-control results; duplicate online-source and C-source-control rows omitted from CSV only. Per-case gradients, parameter changes, skipped updates, matching coverage, costs and anonymous scene grouping included. All128current observations retained. All targetVAL has historical exposure, only1Drone/2Quad scenes; not fullVAL or blind independent-scene evidence.

C controls were omitted from initial storage and captured after C scores were exposed, using fixed identical operations. Two failed bitexact replay attempts retained; source32bitexact, adapted outputs repeated within9.536743164e-7m (declared1e-4m bound). Original C predictions unchanged. Controls are descriptive, not independent confirmation.

Own cached RGB/VLM G is also tested onA64: Drone episodic projection updates harm mIoU, Quad unchanged, same-information direct G improves Quad. These negatives are included; external point/text methods and own extra-RGB systems have different information resources.

No sensor data, captions, candidate coordinates, instance IDs, features, target annotations, weights, credentials or full paper text are published. Public plan files are explicitly redacted templates with original hashes. Dependencies and execution differences are in DEPENDENCIES.json and results/METHOD_PORT_AUDIT.json.
