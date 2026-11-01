# R28 — locate R27 failures without new model execution
Offline cached diagnosis over all528R27 observations, plus24input-hash selected
multi-instance blind visual panels (12/platform). Assistant visual review hid
Qwen/source scores, grounding selected boxes,3DGT andIoU; independent detector
proposals were visible. Labels sealed before unblinding. One assistant annotator,
NOT expert-adjudicated identity gold.3uncertain cases retained unchanged.

Fourwrong strict visual rankings,15correct,2ties. Two correct-image/mapped-cue
cases switch to other visual-point support after source fusion; one correct
image escapes to an unmapped zero-cue candidate. Highest cues broadcast to6-23
queries (median14). No independently confirmed complete wrong-instance mapping
in this subset; sharedcamera/masks cannot certify external physical identity.

All387pooled cases have mapped source winners. Originaljoint selects an unmapped
query in34/528cases; Q64has1such query change but bothIoUs0. Anchor/mapped-only
counterfactual adds0.317953pp onOLD400,0.168197pp onALL528,0onQ64. Gains12,harms7
globally; mean/median centering worsensALL528. Fixedcommonoffset stress retained;
no bestoffset selected. These are posthoc diagnostics, NOT new heldout gains.
Sourcefirstnoun BBFtoken readout verified all528; fulltext still encoded. Raw
noun-token probability andQwenbinarylogodds are not same probability space.

Dependencies through19dfcc13c27daab1b823724bdf85dd7cdb360812 and licensed3EED assets.
No newQwen/groundingforward,training,GPUexperiment orTTA. OldproductionG preserved.
Raw RGB panels/captions/pointclouds/calibrationcoords/opaque taskIDs/modelweights
not exported.24case visual-support labels assume same camera/masks;3Dgeometry
checks explicitly proxies. No error partition generalized to all528 or unseen
scenes. All scalar counterfactuals and negatives retained, repeated arms not
independent observations. Read REPORT.md and audit/limitation files.
Workflow: prepare28 -> blind assistant review/seal -> fusion28,identity28,
support28,summary28,prior28,visual_compare28,join28,audit28,audit_support28,
finalize28,publish28. Existing sealed outputs must not be overwritten.
