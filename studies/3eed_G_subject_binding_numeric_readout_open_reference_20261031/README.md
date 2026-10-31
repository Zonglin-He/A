# G repair: actual subject fields, numerical visual readout, open references

[Full report](REPORT.md). Real Waymo-only→Drone/Quad direct-reranking experiments completed. 208unique observations, frozen source and candidate boxes, zero TTA parameter updates. Old18missing subject-field cases are actually scored, not merely codepatched. FixedYes/No logits replace freegenerations; numericaloutputsfinite, no formattinggate. IndependentGroundingDINO provides open-vocabulary reference regions.

FirststrictDINOgate144 preserved; W32 exposeddevelopment revealed automatic detector-veto misrejection. RevisedG2 keeps mask/pointcorrespondence,2crop appearance/attribute evidence and knownrelationcontradiction; detectorabsence staysunknown. NewZ64fixed before revisedGoutputs: Quad23.649301→30.852559mIoU(+7.203258pp),Drone13.983825→14.051852(+.068027pp). 5interventions/5IoUimprovements, no observedharms, but clustered95%lowerbounds remain0 and prespecifiedTTA gatefails. No TTA effect claimed.

208G2cases:17interventions,16IoUgains,0harms,1tie. Limitedscenes, lowcoverage, historicaltargetVALexposure and repeatedphysicalsubjects mean this is a researchcandidate, not validateddeployment or broadreliability. W32 is development forG2, not confirmation.

Negativecontrols remain: D112unconditionalvisualrerank has4IoUharms; newZ explicit-field legacybinding control exceeds fullG by1.458905pp onQuad (missedrescue Z_quad_026). No-reference ablation identical toG2, so reference localization has no establishedtaskcontribution. Stable regions are not semanticidentity; genericperson/isolatedgraffiti proposals can be wrong or ambiguous.

All864strictprotocol and1456G2task-armrows, perpair evidence, scenealiases, failed/retainedcontrols, independentCPUaudit, sourceimmutable and hash-boundprotocols are included. These2320evaluationrows overlap inD/W and do not represent2320independent observations. No rawmedia/captions/coordinates/instanceIDs/GT/features/weights/credentials or fullpapertext are exported.
