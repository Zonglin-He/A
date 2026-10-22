# Learned semantic projection and cross-fit prototype loop

Original frozen SpatialActivation word head is supervised by pooled verb-label BCE. This study observes its pre-mean per-frame logits and 256-dimensional input; it does not use actionness/foreground as new feedback. Dense frame word responses are an extrapolation from pooled training, not calibrated event probabilities. Original full text and all original visual features/time positions remain unchanged. Motion vocabulary34 and appearance vocabulary20; no class-name/annotation vocabulary reads. Learned query-conditioned semantic coordinates are distinct from raw arbitrary 256-dimensional cosine matching.

NativeG6 soft inclusive interval membership seeds opposite-parity positive/negative class centroids. Expected soft-Bernoulli log-likelihood ratios SUM classes and SUM inclusive interval frames. Exactly TWO rounds, every round anchored to original q0, no cumulative posterior prior, optimizer, learned weights, gain, threshold or abstention. Cross-offset features overlap and q0 uses both offsets: cross-fitting is not independent validation. Class/frame pseudolikelihood independence is not guaranteed.

Controls: one round; direct same SECOND-round evidence under uniform legal prior; reverse65 semantic rows; fixed Gaussian projection of SAME head input with rank mapping to EXACT per-class real sigmoid histogram; appearance word space; original q0 squared; constant-projection exact NativeG6. All64 active. All64 seal before independent CPU scoring. Full64 hidden/logits/q/Source boxes bitwise; spatial complete boxes/eight observations/DP unchanged. Five CPU tests membership/inclusive endpoints/constant-label invariance/parity isolation/synthetic positive control.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|SemanticPrototypeLoop2|30.9658|58.8883|31/17/16|6|
|OneStepPrototype|30.5419|57.9067|29/18/17|5|
|DirectSemanticEvidence|28.2426|54.3489|29/35/0|26|
|ReversedSemanticLoop2|29.5245|54.6838|26/26/12|14|
|RandomProjectionLoop2|30.0988|56.6894|22/23/19|11|
|AppearancePrototypeLoop2|29.5349|55.5638|29/29/6|15|
|SharpenOriginal|28.7258|53.7518|16/38/10|20|
|NoProjection|29.6952|55.9613|0/0/64|0|
|ConstantProjectionNoOp|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|—|—|

Primary−Global6 t+2.9270pp CI[1.112207638645931, 4.876849853192072]; v+1.2706pp CI[0.15956212022804012, 2.480192442798465].
Primary−Source t+4.0653pp; v+4.0720pp. Source short+0.4666/long+4.9826tpp.
Primary−RandomProjectionLoop2 t+2.1989pp CI[0.7184080244262623, 3.7368193614647307].
Primary−ReversedSemanticLoop2 t+4.2045pp CI[1.4595271666384402, 7.235284658036222].
Primary−SharpenOriginal t+5.1365pp CI[2.5848958486358673, 7.778622868877509].
Primary−OneStepPrototype t+0.9816pp CI[0.2390837045946768, 1.8983988568214027].
Primary−DirectSemanticEvidence t+4.5394pp CI[0.22006274953658575, 9.279901993216148].
Primary−AppearancePrototypeLoop2 t+3.3245pp CI[0.47209614948809825, 6.565495809961717].

Basic exposed-development screen only; fixed dual-domain recheck required.
Study worker28.49s includes load/no-op verification/recording; actual calls{'fusion_offsets': 128, 'native_replay_queries': 64, 'extra_dense_wordhead_forwards': 256, 'new_text_encoder': 0, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. Deployment would reuse existing Full hidden and add dense word heads/prototype math; this study is not deployment timing. No new RGB/visual encoder/text encoder/expert/observations/optimizer.
Official/dense checks1408 maximum error6.66e-16. [All signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [Source duration groups](source_duration_groups.json), [fixed protocol](protocol.json).

Original source [SpatialActivation](https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/grounding_model/classifier.py) produces head(query).mean(frames). Pooled-label supervision does not establish frame-event discrimination. This is TA-specific learned semantic-interface feasibility, not a universal three-backbone interface or OPD/DSE reproduction.
