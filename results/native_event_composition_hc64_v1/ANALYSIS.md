# Positive native subevent composition: HC64

Fixed, exposed development study; space/expert8 unchanged, paper full paused.
Remaining failures include compound action descriptions: approach+sit,raise+pour+turn.
This branch tests a semantic alternative to context/position/absence-ratio feedback:
read actual positive subevent clauses through the same trained native temporal model,
then form a distribution spanning the compound event.

Existing Source Stanza dependencies identify only main-root VERB conjunctions with
one inherited subject. Keep the original nominal identity subtree and arguments.
Only plain AND is supported; explicit new subjects and before/after/while/until/when/
unless/or/but relations are not guessed. First/last main conjuncts supply at most2
new texts. Other captions retain the original posterior exactly, for every control.
This is semantic scope, not a confidence/GT rejection gate; coverage is reported.

Original RoBERTa+resizer, FP16 fusion and exact native FP32 two-pass decoder read each
clause at both original offsets/six layers. Original projected visual features and
full target identity are retained. The independent first/last interval distributions
produce the exact minimal-envelope posterior. Primary is fixed half original Full,
half clause envelope, original physical MBR and complete-box DP; no ratio or fitting.

DirectClauses uses identical information without Full fusion. SelfHullFusion uses
two copies of the original posterior to control mere expansion. IdentityHullFusion
and MaskedEventFusion reuse old same-native interventions; reversed clause placement
and exact no-composition are retained. No best control selected posthoc.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|ClauseFusion|31.3453|58.4307|30/21/13|11|
|DirectClauses|30.8003|57.8335|27/25/12|19|
|SelfHullFusion|30.5252|57.4780|33/18/13|7|
|IdentityHullFusion|29.9362|57.0747|27/25/12|17|
|MaskedEventFusion|29.4263|55.0469|28/20/16|14|
|ReverseClauseFusion|22.5854|43.2702|16/34/14|28|
|NoComposition|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Syntactic support52/64;12 captions preserve NativeG6 exactly by declared scope, not by observed quality. Primary−NativeG6: v+1.6502pp CI[0.19355546077736843, 3.182805078161059], t+2.4693pp CI[-0.13039227713088047, 4.967468278166768]. Primary−Source: v+4.4516pp, t+3.6077pp. Primary−SelfHullFusion: t+0.9527pp CI[-0.7223215934531754, 2.632243139590504]. Shortt-5.7702pp, longt+4.5696pp versusNativeG6.

Matched candidate feedback accuracy57.6404% including unsupported ties, native52.1362%; supported-only accuracy0.594035344724579. All new outputs sealed before scoring.

Primary does not pass basic broad-improvement development screen. No promotion or fusion-weight scan.

All64 original reencoded text tuple/hidden/posterior/Source boxes restore bitwise; no-composition exact. Official/dense checks1152, maxerror6.66e-16. Original48+434 sources intact. Worker49.03s inclmodel loading, calls{'text_forwards': 168, 'fusion_offsets': 336, 'native_replay_queries': 168, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. No RGB/visual encoder/expert calls or new observations. Timings are cached research execution, not deployment end-to-end speed.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).

[TFVTG](https://github.com/minghangz/TFVTG) motivates subevent decomposition but uses extra LLM/VLM; [DeCo](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_DeCo_Decomposition_and_Reconstruction_for_Compositional_Temporal_Grounding_via_Coarse-To-Fine_CVPR_2023_paper.html) learns compositional representations. Neither is reproduced here; this bounded deterministic parser/native test has less semantic coverage and no trained compositional head.
