# R29 — instance preference gates and frozen-box matching TTA

Read REPORT.md. One fixed full-scene/fulltext pair G,308numeric Qwen forwards;
R22cached154physical pairs gate decomposition;24input-selected blind assistant
reviews plus all6accepted corrections reviewed after fit (29uniqueobservations).
Not expert identity gold. No manual/GT filtering of runtime pseudo feedback.

Waymo-only epoch100 source. Fit on208exposed target observations,22accepted
unique-instance-group pseudo preferences from19observations, separately perplatform.
Full matchingqueryMLP184960params,AdamW5e-4x16,boxes/backbone/decoder frozen.
The probability loss failed to implement6corrective preferences. BeforeBscoring,
one exact monotonic logodds loss-readout repair implemented22/22pairs; all failed
arms retained. Local pair satisfaction does not ensure global Top1 or transfer.

New64B:32/platform, actualdecodedRGB/XYZ disjoint prior528anddeclared history,
but2Drone/3Quad scenes overlap prior data and fullVAL has historical label exposure.
Bnot used for Gconstruction/fitting; predictions sealed before current tasklabels.
Same-A nonparametric cache is the matched transfer control; directsameGno-update
is measured onA. No claim these distinct protocols are interchangeable.

Blogodds Drone-6.452706pp,Quad+4.795222pp,mean-.828742pp; not adopted.
Quad3improved0harmed,Drone5improved11harmed. All negative scalar records included.
128fullmodel adapted-state closure forwards confirm frozenboxes and unchanged
nonmatching weights. CPUaudit redoes1712taskscalars,154raw preference policies,
208group partitions and contentdedup. Sceneclusters, not frames, used for intervals.

Requires private licensed3EED assets and dependencies from parent22e7ef1. No weights,
featurecaches, rawmedia/captions, calibration coordinates or opaque originalIDs
redistributed. Programs preserve all localseals and should use freshattemptdirs.
Workflow: audit_pairs29 -> prepare_review29 -> seal_review29 -> fullpair29 + groups29
-> evaluate_G29 -> select_new29 -> capture29 -> fit29(probability) -> registered
readoutrepair -> fit29 --logodds -> closure29 -> score29 -> mechanism29 -> postfit
correctionreview -> audit29 -> finalize29 -> publish29. Scalar output includes
allarms; repeated arms/views/pairs are not independent observations.
