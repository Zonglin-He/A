# R30 — preserve unlabelled outputs during matching preference adaptation

Read REPORT.md. Same22R29external pseudo preferences, sameWaymo-onlysource,
matchingqueryMLP184960params and logodds RankNet. Single mainchange: relative
sourceq0 logodds output-MSE on189Aobservations withoutacceptedfeedback.
Finitebeta[0,.1,1,10],AdamW5e-4x16/platform; beta0 exactR29state replay.
One shared beta1 chosen on exposedoldQ64 withprespecified correctivefeasibility.
R29R64never used for choosingbeta. NewS64 contentdisjointprior592+declaredhistory,
but2Drone/3Quad historicalscenes andpriorfullVALlabel exposure; Snotfit/Ginput.

Keep reducesDroneU harm22->12 and retains6/6requestedreplacement pseudopreferences
(includingoneknownbadfeedback), not merely frozenrecovery. S64Drone-1.653647pp,
Quad+2.839300pp,balanced+.592827pp vs source; sceneCI crosseszero. Not adopted as
robustcrossplatform method. ActiveA-memory andallAzeroU-memory no-update controls
retained. No platformwiseposthoc hybrid or Sdriven retuning.

Pairgraph explains one apparentthirdcandidate as anotheracceptedpositive; one
uncoveredthird remains. No automatic negatives/newQwen comparisons. Available
feedbackevidence19mixed,1appearance-only,2neither(sameobservation), no certified
puregeometry group. Not proof ofQwenreasoning orXYZobservability, not an appearance
versusgeometry causaltransfer comparison. Badfeedback unchanged, no manualfilter.

CPUaudit reconstructs8keepobjectives, selection,1344taskscalars,44pairgraphstates,
246negative comparisons;128fullmodelclosureforwards exactargmax/frozenboxes and
unchangednonmatchingweights. NewQwencalls0, newsourcefulltraining0. SourceWaymo
forgetting was not evaluated. EATA/EcoTTA/newRGBmodule not introduced.

Privateassets andprior studies through0e7503a required. No rawRGB/pointclouds,
captions, opaqueIDs, coordinates, features orweightsredistributed. Usefreshoutput
attemptdirs; originalseals never overwrite. Workflow:fit30 -> select30 ->
select_new30/capture30 -> predict30 andmemory_all30 -> closure30 -> seal_evidence30
-> score30 -> mechanism30/graph30 -> audit30/audit_graph30 -> finalize30/publish30.
Allscalararms/negative comparisons preserved; repeatedpairs/views are not
independent observations. Seeprotocol files for selection andsourcecontracts.
