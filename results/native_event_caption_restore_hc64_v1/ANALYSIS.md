# Native complete-caption-supported feature context restoration

Exposed HC64 prospective branch. Stage17 FillOnly is a control, not promoted. Concrete auxiliary drift and high-target-coverage failures motivate restoration of context supported by original complete-caption attention. Observe the original final spatial encoder self-attention weights without changing args, flags, kernel or outputs. Real caption tokens exclude source-added subject prefix and specials; tokenizer IDs must match cached source. Mean caption rows and sum corresponding original 2D/3D grid-key attention. Attention is not an event probability or causal explanation.

Fixed fulltime BoxFixed auxiliary trajectory and 1.5 context. DenseFill replaces outside-valid 2D/3D features with inside mean. Restore original features by attention/max-valid-attention per frame, W=1 inside ROI. Zero visual attention restores full valid scene and is recorded. FP32 blend, original FP16 cast, exact inside/padding values, original dense keymask and safetycorner. Original motion contains fullclipcontext. No RGB, visualencoder, textencoder, expert, observations, querylearning or optimization. Full caption, physical IDs/time embeddings, native two passes, six heads, two-offset hull, physical MBR and actual final complete-box DP unchanged.

Primary fixed .5Global6+.5CaptionRestored. Controls restore EXACT same per-frame outside weight mass: uniform placement; reverse outside-cell weight multiset; same current weight multiset assigned by reversed-time attention spatial rank. These are matched-strength comparisons, not known false events. DenseFill posterior, interval and final boxes must exactly restore old FillOnly; source/native/self/noop aliases bitwise. All64 predictions seal before independent CPU labels. No quality gate, layer/context/temperature/gain/fusion scan or posthoc control promotion.

|Arm|complete vIoU|tIoU|time +/−/=|time harm>5pp|
|---|---:|---:|---:|---:|
|NativeG6|29.6952|55.9613|0/0/64|0|
|CaptionRestorationFusion|29.7988|55.8500|32/24/8|13|
|DirectCaptionView|28.3643|53.0104|32/28/4|24|
|DenseFillFusion|30.1406|56.5246|31/25/8|14|
|UniformRestorationFusion|30.0153|56.0320|32/26/6|12|
|SpatialPermutationFusion|30.0050|56.0610|31/25/8|13|
|TemporalRankFusion|29.9026|55.7358|33/25/6|14|
|NativeSelfMixture|29.6952|55.9613|0/0/64|0|
|NoView|29.6952|55.9613|0/0/64|0|
|Source|26.8938|54.8230|0/0/64|0|

Primary−Global6 t-0.1113pp CI[-3.2806197669535546, 2.6821095954582788]; v+0.1036pp CI[-1.4318839256140854, 1.4453734645260614].
Primary−Source t+1.0270pp; v+2.9050pp. Source short-1.3523/long+1.6335tpp.
Primary−DenseFillFusion t-0.6746pp CI[-2.2406605810197386, 0.8383373169548164].
Primary−UniformRestorationFusion t-0.1820pp CI[-0.8120359338166102, 0.4491815773875722].
Primary−SpatialPermutationFusion t-0.2109pp CI[-0.8524848836914679, 0.40128418481114886].
Primary−TemporalRankFusion t+0.1142pp CI[-0.5497290921066496, 0.8669913441865897].

Not qualified. Do not scan attention/restoration weights or promote a control.
Worker82.00s including loading/research recording; {'fusion_offsets': 768, 'native_replay_queries': 384, 'new_text_encoder': 0, 'new_expert': 0, 'new_observations': 0, 'new_RGB': 0, 'new_visual_encoder': 0, 'new_optimizer': 0}. Study six reads/query is not deployment timing. A deployment candidate adds actual source-fusion/time readout, not a new encoder, but total deployment latency has not been measured.
Official/dense checks1280, maximumerror9.99e-16. FourCPU tests. [All signed results](signed.tsv), [severe losses](severe_declines.tsv), [Source duration groups](source_duration_groups.json), [protocol](protocol.json).

Separately bound postseal restoration audit: outside original-feature restoration mean0.201346, zeroattentionfallback0; primary posteriorTV vsDenseFill0.06427,Uniform0.04710,SpatialPermutation0.04978,TemporalRank0.04250. Actual predictions did change; mean-only uncertainty is not failure to execute. Matched-duration/prior candidate ranking52.91% CI[49.33,56.42] includesrandom50; native52.14%.

Concrete failures do not establish semantic restoration. RC041 gained26.62tpp relativeDenseFill, but uniform/spatial/time controls restore the SAME final interval: cannot credit semantic placement. RC042 remains−63.16tpp vsG6 under all restorations; RC011 primary−16.87 vsDenseFill−14.04 despite restoring outsidecontext. RC064 loses the priorFillOnly+33.23t gain (primary0vsG6), showing extra context can also undo a beneficial correction. RC023 primary+44.72t, uniform+39.65/spatial+42.19/time+29.76 is one positive placement case; it does not override failed aggregate/short-group/control comparisons. Targetcoverage from prior audit is offline target-only proxy, not online gate or interaction-object truth. [Separately bound audit](restoration_audit.json).
