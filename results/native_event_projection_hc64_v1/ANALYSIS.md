# Positive clause feedback projected through the frozen native time model

Fixed exposed HC64. Same sealed shared-clause teacher distribution; unchanged old query optimizer: shared-offset256 temporary state, Adam lr.01,10steps, normalized radius.1, last state. No new encoder, text, expert or observations. Spatial method is unchanged.

Earlier RawQuery used unreliable ReferDINO-derived event targets. Here the feedback is the actual positive native clause posterior that improved direct means, but harmed short events. This tests whether the Full-caption native temporal family filters incompatible clause outputs. It does not increase update steps or claim a new query interface.

All52 syntactically supported descriptions execute all3 optimizer arms. Twelve unsupported descriptions inherit exact native output; no confidence/GT rejection or best-step selection. Same-information direct parent, same-budget self-hull learning and reversed-time learning/direct controls are retained. A lower pseudo-loss is not task correctness.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|ClauseQuery|29.9367|56.2718|13/10/41|1|
|SharedQueryClauseFusion|31.4014|58.5433|30/21/13|9|
|SelfHullQuery|29.9128|56.2846|14/10/40|1|
|SelfHullFusion|30.5252|57.4780|33/18/13|7|
|ReversedClauseQuery|29.9729|56.4490|14/9/41|1|
|ReversedClauseDirect|22.7518|43.1178|15/35/14|28|
|NoUpdate|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−NativeG6: v+0.2415pp CI[-0.04264869297625219, 0.5276226688256362], t+0.3105pp CI[-0.14285980844920973, 0.757559677219362]. Primary−Source: v+3.0429pp, t+1.4488pp.
Learned−identical direct feedback: v-1.4647pp CI[-2.8579869943132237, -0.15339550585538345], t-2.2715pp CI[-4.555597393626646, 0.10620404223647342].
Shortt-0.2232pp, longt+0.4465pp versusNativeG6.

Not qualified for promotion. Preserve all controls and negative tails; no step/lr/radius scan.
All64 native hidden/final logits/parent aliases restore bitwise. Differentiable native posterior maxerror1.97e-16. Official/dense checks1152, maxerror6.66e-16. Actual worker45.34s includes model loading/research recording, not deployment end-to-end. Calls{'native_decoder_calls': 3560, 'optimizer_steps': 1560, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_text_encoder': 0, 'new_expert': 0, 'new_observations': 0}.

[Signed](signed.tsv), [severe tails](severe_declines.tsv), [optimization](optimization_audit.json), [protocol](protocol.json).
