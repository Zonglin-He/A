# Shared full-caption native temporal query in positive clause reads

Fixed exposed HC64 study. Same52 plain-main-AND descriptions and two positive clause
texts, full frame context, original visual features/expert8/spatial method. No new
expert, visual encoder, RGB or optimizer. Original paper full remains paused.

Previous positive clause composition improves means but harms short events. Clause
Source-box drift is only an identity/conditioning proxy, not correctness. This test
keeps each clause's actual encoded text+motion memory/query_content/time position,
but reuses the ORIGINAL Full-caption actual second-pass query_tgt at the same offset.
Only TimeDecoder calls1,3 of the original four-call sequence are replaced; original
expanded stride is restored. Full replacement must be a bitwise no-op. No quality
veto, learned query delta, fitting, attention ROI edit or extra query step.

The Full query encodes more than identity, including original event/global context;
successful reuse would not by itself prove identity binding. Older object-conditioned
queries and query_tgt rewrites already exist, and full-caption TubeReplay changes
attention. Here the new mechanism under test is Full query reuse across actual
positive subevent memories, not novelty of the query interface.

Primary same fixed half Full/half minimal-envelope posterior; original physical MBR
and complete-box DP. Native unshared ClauseFusion is the key single-change control,
plus direct shared envelope, self-hull/identity-hull, reversed placement and no-op.
All12 unsupported descriptions preserve native outputs exactly by declared syntax.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|SharedQueryClauseFusion|31.4014|58.5433|30/21/13|9|
|DirectSharedClauses|30.9340|57.8829|28/24/12|17|
|ClauseFusion|31.3453|58.4307|30/21/13|11|
|SelfHullFusion|30.5252|57.4780|33/18/13|7|
|IdentityHullFusion|29.9362|57.0747|27/25/12|17|
|ReverseSharedClauses|22.7518|43.1178|15/35/14|28|
|NoComposition|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−NativeG6: v+1.7062pp CI[0.23896075111450715, 3.240807718248421], t+2.5820pp CI[-0.048671405892056596, 5.0639850190651075]. Primary−Source: v+4.5076pp, t+3.7203pp. Primary−unshared ClauseFusion: t+0.1126pp CI[-0.20789187697973455, 0.43206043388899856]; versusSelfHullFusion t+1.0653pp. Shortt-6.1812pp, longt+4.8157pp versusNativeG6.

Matched feedback accuracy57.3260% including unsupported ties, native52.1362%; supported-only0.5901660120083199. No online selection uses these labels.

Primary does not pass broad-improvement development qualification. No promotion or shared-query strength/step scan.

All64 true Full text/hidden/posterior/Source boxes and original parent outputs restore bitwise; no-composition exact. Official/dense checks1152, maxerror6.66e-16. Original48+434 sources intact. Worker50.42s inclmodel load and research capture, calls{'text_forwards': 168, 'fusion_offsets': 336, 'native_replay_queries': 168, 'shared_second_pass_queries': 336, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}; not end-to-end deployment time. All new outputs sealed before scoring.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).
