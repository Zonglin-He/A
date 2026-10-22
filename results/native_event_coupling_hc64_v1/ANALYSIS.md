# Coupling genuine native event occurrences before envelope construction

Exposed HC64. Native Full/First/Last posteriors are reused exactly, no new model or expert call. Independent atom draws can pair distinct occurrences. The only new factor is common-quantile coupling of interval atoms ordered by physical midpoint, with deterministic start/end ties; all marginal probabilities retained within1e-12. This assumes comparable occurrence ranks and does not prove common semantic identity. No rare-order conditioning, window, extra weight, atom threshold or GT rejection. Original .5native mixture, MBR and complete-box DP fixed.

Same-information unblended DirectCoupled; independent clause envelope, antimonotone rank pairing, reflected last, self coupling and self-independent-hull controls. All52 supported members,12unsupported native;64seal before independent CPU labels. Native/Source/archived clause/selfhull aliases verified.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|MonotoneClauseFusion|31.0885|58.2454|29/20/15|9|
|IndependentClauseFusion|31.3453|58.4307|30/21/13|11|
|AntimonotoneFusion|31.0056|57.9546|29/23/12|13|
|ReflectedLastFusion|25.2874|48.1823|18/34/12|25|
|SelfCoupledFusion|29.6952|55.9613|0/0/64|0|
|SelfIndependentFusion|30.5252|57.4780|33/18/13|7|
|DirectCoupled|30.5072|57.3750|25/26/13|19|
|NoComposition|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−G6 t+2.2841pp CI[-0.13161162163976034, 4.631800211621231], v+1.3933pp CI[0.05489783992859091, 2.787962934086656]; Source t+3.4224, v+4.1947pp.
Source shortt-5.4940pp; longt+5.6952pp.
Primary−IndependentClauseFusion t-0.1852pp CI[-0.8226325911633476, 0.407327114481039].
Primary−AntimonotoneFusion t+0.2908pp CI[-0.6621346320783107, 1.3785292311830184].
Primary−ReflectedLastFusion t+10.0631pp CI[5.0052779545450035, 15.204036262410316].
Primary−SelfCoupledFusion t+2.2841pp CI[-0.13161162163976034, 4.631800211621231].
Primary−SelfIndependentFusion t+0.7675pp CI[-0.8289798362390279, 2.3826378801571293].
Primary−DirectCoupled t+0.8705pp CI[-1.238515828916512, 3.407359805391563].

Not qualified; no coupling cost/bin/weight scan or posthoc promotion of controls.
Worker 3.84s inclloading/research recording; calls{'new_decoder': 0, 'new_fusion': 0, 'new_text_encoder': 0, 'new_visual_encoder': 0, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_optimizer': 0}. This is not deployment end-to-end cost. Official/dense checks1280 maxerror6.66e-16.

[Signed outcomes](signed.tsv), [severe declines](severe_declines.tsv), [Source groups](source_duration_groups.json), [protocol](protocol.json).
