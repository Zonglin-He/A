# Original global-grid temporal positions in overlapping contexts

Fixed, exposed HC64 development. Paper full remains paused, space unchanged.
Previous ContextInnovation almost restores Global6 but does not improve it, while
short events still suffer. Architectural audit finds native query_decoder.time_embed(t)
starts at0 in every crop. Suffix nodes at original offset indices8..32/31 become
0..24/23; a content-context intervention therefore also changes temporal positions.

One new configuration retains full-grid positional embeddings at actual selected
offset indices, with a scoped hook in both original native decoder passes. Frozen
weights/state_dict/source are unchanged. Full caption and original projected feature
values remain fixed; prefix0..48/suffix16..64 and censored innovation qContext/bSameLogits
feedback are unchanged. Full and Prefix have identical native position values and
must restore previous outputs bitwise64. This is a mechanism feasibility test, not a
trained debiasing method or a claim that temporal shift alone causes errors.

All percent metrics, improved/harmed/tied time versusNativeG6:

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|GlobalPositionInnovation|29.8733|55.5108|22/22/20|10|
|ContextInnovation|29.4482|55.8017|26/25/13|9|
|CensoredContext|26.2446|49.1264|14/44/6|33|
|Sharpen|28.7258|53.7518|16/38/10|20|
|ReverseGlobalPosition|28.1659|53.1978|12/34/18|16|
|NoInnovation|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−NativeG6: v+0.1781pp CI[-0.6104204972543043, 0.8877483564555255], t-0.4506pp CI[-1.782078722901815, 0.7744938547072999]. Primary−Source: v+2.9795pp, t+0.6878pp. Primary−rebased ContextInnovation: t-0.2909pp CI[-1.4524151716638358, 0.8524123582404423]. Shortt-0.7680pp, longt-0.3696pp versusNativeG6.

Matched same-duration/near-native-prior feedback accuracy49.0925% CI[0.4614406504957495, 0.5200093348623898], native52.1362%;721,976pairs. This post-seal diagnostic does not choose inference.

Primary does not pass broad-improvement development screen. Do not promote or scan position offsets/feedback weights.

All64 Full hidden/posterior and Prefix posterior bitwise; reused Source/Native/parent controls exact. Official/dense checks1024, maxerror8.88e-16. Original48+434 sources intact. Worker45.29s including source loading, calls{'fusion_offsets': 384, 'native_replay_queries': 192, 'new_expert': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_RGB': 0, 'new_optimizer': 0}; no end-to-end deployment speed claim.0additional expert/observations.

[Signed outcomes](signed.tsv), [severe tails](severe_declines.tsv), [information audit](information_audit.json), [protocol](protocol.json).

References about possible temporal/location bias, not validation of this method: [Towards Debiasing TSG](https://arxiv.org/abs/2111.04321), [Absolute Position Embeddings](https://aclanthology.org/2022.findings-emnlp.326/). The latter concerns language models, not STVG.
