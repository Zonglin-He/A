# Full shared-time spatial comparison: independent recalculation

Both domains use the complete target lists, raw identical eight expert observations and one immutable Current Full interval per query. All three tIoUs and scored annotated-frame supports are exactly equal per query. PureExpertLinear is raw pixel-corner interpolation on every input video frame with nearest-expert tails. SharedCorrection is freshly reconstructed from Source/raw8 residual consensus. Full reruns the original final DP on unchanged cached candidate/unary/link sufficient state; no new transport fit or unavailable raw grid is claimed.

|Dataset|Path|tIoU|vIoU|R.3|R.5|Event-unobserved IoU|
|---|---|---:|---:|---:|---:|---:|
|hc2|PureExpertLinear|49.764548|25.592718|38.433228|14.511041|50.451763|
|hc2|SharedCorrection|49.764548|26.489749|40.063091|15.772871|52.761593|
|hc2|Full|49.764548|27.349044|41.219769|16.876972|54.354908|
|vidstg|PureExpertLinear|34.918192|14.804402|19.858294|8.686790|39.762174|
|vidstg|SharedCorrection|34.918192|13.948734|18.722702|8.230612|37.861669|
|vidstg|Full|34.918192|14.558151|19.411822|8.463554|39.504891|

HC Full exceeds pure interpolation by +1.756325 pp vIoU; Vid Full is lower by −0.246251 pp. Full exceeds direct shared-corrected E by +0.859295/+0.609417 pp. Different dataset outcomes do not establish sampling imbalance: every within-domain comparison has the exact same complete query list and interval/support. The corrected numbers do not turn Vid into a positive spatial result.

Event-unobserved IoU uses the same dense annotated frames inside common predicted∩GT event, excluding exact eight expert times. HC 1732/1902 and Vid 8241/10303 queries have nonempty support; empty-support conditional values are null, while all queries remain in primary vIoU. This replaces the previous incompatible dense-pure versus sampled-native unobserved comparison; it must not be compared numerically with all-video/native-grid means.

The old baseline wrappers had two distinct issues already preserved: incomplete dense input tails and a historical Source_interval alias corresponding to MBR rather than native Source. This controlled table explicitly binds actual Full_interval and needs neither Source time nor the proposed temporal branch for its interpolation path. The raw-corner interpolation kernel contains no E/transport/DP.

Full still allows E instead of raw expert boxes at observed frames. That unresolved hard-anchor design/implementation contract is not repaired here; output equality at anchors is not a property of this comparison. Numeric reproduction does not validate an intended hard-anchor method.

All signed changes, severe tails, short/long groups and pointwise 10000 parent-cluster CIs are in ROOT_report.json and the scalar TSVs. Cohorts were exposed before this audit; there is no parameter selection or equivalence claim. 36615 official arm/query checks, 97640 independent array checks and all six actual child wait receipts passed. Outer tool37292 became unavailable after context handover, so its external wait/exit is explicitly unknown rather than fabricated zero. All Source/expert/GPU/model/optimizer calls in this recalculation are zero; inherited acquisition costs are nonzero.
