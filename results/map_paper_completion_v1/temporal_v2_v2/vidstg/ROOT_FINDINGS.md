# Vid complete-target findings: shrinkage and temporal evidence

**Complete10,303queries/732parents, all9arms officially scored and array-replay audited.** The reported NoShrink control sets kappa0; current FullMAP retains kappa64. Same observations/features/prior/beta/context/one round, each posterior’s own joint-MAP interval and a new exact final spatialDP. No full-result parameter selection is made.

| Configuration | tIoU % | vIoU % | Severe t / v / union vs native Source |
|---|---:|---:|---|
| Current FullMAP, kappa64 |34.664067|14.433693|1139 /1154 /1934|
| NoShrink, kappa0 |34.927982|14.548371|1487 /1221 /2238|

NoShrink−FullMAP: **t+0.263915pp [0.044013,0.485967], v+0.114677pp [0.003874,0.223256]**. These are pointwise parent-bootstrap intervals; the v lower bound is close to zero and is not a familywise statement over all controls. Against current FullMAP itself, NoShrink has1,178temporal and436video severe losses (>5pp); all signed unions/rescues are in all_paired_tail_summary.json. The larger native-Source severe union (1934→2238) must not be hidden by the mean gain.

Short6,060queries: NoShrink v+0.122483pp CI[-0.001145,0.252638]. Long4,243queries: v+0.103529pp CI[-0.089427,0.287221]. Both duration-specific v intervals cross zero. Short observed fixed-support sIoU declines0.169078pp CI[-0.303364,-0.040640]. Short boundary MAE increases0.057231s CI[0.010740,0.107891], while long boundary MAE decreases0.110046s CI[-0.176004,-0.043706]. Positive MAE changes indicate worse boundaries, unlike positive IoU changes.

Both HC and Vid show a mean/tail tradeoff: removing shrinkage improves mean metrics under these exposed full targets while increasing native-Source severe union. This does not support calling kappa64 mean-optimal; it also does not establish kappa0 as a replacement setting from these results. The current method remains fixed.

The uncentered EventOnly control loses8.271428vIoUpp versus FullMAP; EventCentered recovers to14.518460%, v+0.084766pp CI[-0.013476,0.184133]. HardMembership loses0.101773vIoUpp CI[-0.160447,-0.046844]; NoPosteriorUpdate loses0.562294pp CI[-0.707613,-0.426163]. UniformPrior v−0.156580pp has an interval crossing zero and native-Source severe union3307. All of these gains/losses and all9means are retained in [complete analysis](ANALYSIS.md), report.json and the signed TSVs.

Audit:92,727officialdensechecks, maximum disagreement3.552713678800501e-15;51,515parent/newprediction/receiptSHAchecks, all arrays cold-rederived exactly, allmeans recomputed. Query cached-work wall sum715.248848s on worker1083189/start245060214 is not complete deployment latency or total study wall; original acquisition cost is inherited. This read-only root export adds no model, expert or GT scoring.
