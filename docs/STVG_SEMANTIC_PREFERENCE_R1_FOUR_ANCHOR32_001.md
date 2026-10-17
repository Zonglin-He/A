# R1 semantic preference versus geometric weights: matched32 Student effect

Complete exposed HC16/Vid16, all four prelocked anchors per Query, exact original Direct32 observation positions/raw source pixels/context/query; preference images additionally carry the pre-existing numbered candidate overlays, while Direct uses the raw anchor image. All32 previous slot0 outputs are exactly reused;96 new anchor inputs/192 new native forwards,256 total real forwards. Direct used128 coordinate generations; these are different computation types and counts. Remaining Direct differences are continuous boxes vs fixed candidate support and on-policy32 sampling vs fixed-support distillation. The causal mechanism control is same-support GEOM, not Direct. No GT enters input/candidate/Teacher/Student/acceptance. Both Student arms and parameter-free D2 are complete and rooted before evaluation.

The first four-anchor GEOM worker completed all32 prediction seals before a pre-existing speed-profile receipt name collision at final I/O (FileExistsError). Original profile, completed predictions, failed-worker log/runtime and its full cost are preserved. Recovery uses a new unique profile label for PREF; GEOM predictions are only rooted, never refit. An initial recovery CPU preflight looked for the later post-root barrier instead of the Student output seal, so runtime creation failed and the launcher exited before any model/priority hold/GT; journal and original source are preserved before the corrected prelaunch. No model, support, parameter, numeric precision, reward, acceptance, or update changed. GEOM and PREF use the SAME four frozen WF candidate atoms, SAME hard reference (argmax of two-order averaged R1 preference, with0 abstention), SAME total weight1-p0, original1792 episodic Student/Adam/domain sigma-lr-tau-steps/full-grid response/fullstep reference-IoU acceptance, rollback/reject-stop and full-frame decoder readout. GEOM uses softmax candidate IoU to that hard reference; PREF uses the actual candidate preference mass. This is fixed-support Gaussian distillation, NOT on-policy32 sampled-action OPD. No clamping: an invalid open-logit fixed atom or incoming Source anchor rejects that anchor; abstention rejects that anchor too. All remaining admissible anchors are retained equally in GEOM/PREF/D2; a Query with no admitted anchor is Source/no-fit in all three. This per-anchor handling is the multi-anchor extension of the old single-anchor rule and is disclosed separately. All32 stay in the denominators.

Root reconstructs support/weights/likelihood/mean gradient independently and checks saved Adam/response/acceptance/state chain. The prior one-anchor root-only absolute-moment audit failure and recovery are preserved in that separate study. This four-anchor root directly uses the independently validated audit with pointwise gamma8 float32 operation-roundoff bounds for the moments, retains strict5e-6 parameter-delta error, and rejects a deliberate incorrect-variance control. No model, update, precision, acceptance or prediction changed; failed root cost remains recorded. Source and temporal logits are preserved; final cached and actual complete decoder trajectories must be exact. D2 uses same hard reference/admission and Source-relative center/log-size physical-time residual propagation inside T2, Source outside, with zero parameter update.

| Domain | Arm | vIoU % | dense common-time sIoU % | tIoU % | nonanchor sIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|50.445363|31.250000|25.000000|
|hc2|T2|27.487135|50.497117|53.666875|50.445363|37.500000|18.750000|
|hc2|GEOM|27.363192|50.127756|53.666875|50.081028|37.500000|18.750000|
|hc2|PREF|27.357912|50.107393|53.666875|50.060226|37.500000|18.750000|
|hc2|D2|17.395617|32.810040|53.666875|32.776607|18.750000|12.500000|
|vidstg|Frozen|12.185607|32.870500|35.923576|32.818166|12.500000|0.000000|
|vidstg|T2|12.239461|32.870500|35.680949|32.818166|12.500000|0.000000|
|vidstg|GEOM|12.274769|33.808781|35.680949|33.752911|12.500000|0.000000|
|vidstg|PREF|11.974958|32.782728|35.680949|32.727234|12.500000|0.000000|
|vidstg|D2|8.542736|23.337288|35.680949|23.299604|6.250000|0.000000|

| Domain | Contrast | vIoU delta pp | paired source t95 pp | + / - / neutral / unknown |
|---|---|---:|---|---|
|hc2|GEOM_minus_T2|-0.123943|[-0.4984746847861167, 0.25058786674302513]|1/1/14/0|
|hc2|PREF_minus_T2|-0.129223|[-0.49860576551320945, 0.24016034943513342]|1/1/14/0|
|hc2|PREF_minus_GEOM|-0.005279|[-0.0181427775047896, 0.007584179469805125]|1/1/14/0|
|hc2|D2_minus_T2|-10.091518|[-17.866152998952035, -2.3168829877205868]|1/8/7/0|
|hc2|D2_minus_PREF|-9.962295|[-17.663224325657602, -2.261366244936945]|1/8/7/0|
|vidstg|GEOM_minus_T2|0.035308|[-1.4014598897666388, 1.472075310287843]|1/1/14/0|
|vidstg|PREF_minus_T2|-0.264503|[-1.2628597659943193, 0.7338546738597631]|1/1/14/0|
|vidstg|PREF_minus_GEOM|-0.299810|[-0.9747560972328028, 0.3751355845770422]|1/1/14/0|
|vidstg|D2_minus_T2|-3.696725|[-8.382249479410413, 0.9887997293872832]|1/6/9/0|
|vidstg|D2_minus_PREF|-3.432222|[-7.737828025842223, 0.8733833679536489]|1/6/9/0|

All accepted/rejected/no-fit reasons and per-source signed/unknown deltas are in RESULTS.json. Paired intervals describe exposed-development uncertainty; no posthoc source exclusions, domain splicing, threshold tuning, new independent confirmation or method promotion. Candidate identity/role correctness remains unlabelled; posterior calibration and candidate coverage are not inferred from this effect. Original semantic512 panel remains44 model-assisted/468 pending. The full two-Teacher2x2 is not complete until Qwen matched cells actually run.

## Comparisons with prior one-anchor and Direct four-anchor

End-to-end Direct contrasts retain different action supports, numbered-image/Prompt versus direct-coordinate interfaces, feedback forms and real calls; they do not isolate semantic likelihood. One-anchor outputs remain unchanged. The inherited private scorer schema scope string is superseded by the four-anchor protocol/input alignment; metric computations and sealed rows are unchanged.

| Domain | Contrast | vIoU delta pp | paired source t95 pp | + / - / neutral |
|---|---|---:|---|---|
|hc2|PREF4_minus_Direct4|-1.222345|[-1.9573172962684477, -0.4873724628046069]|2/11/3|
|hc2|PREF4_minus_PREF1|0.115417|[-0.4448621960938668, 0.6756963410395445]|3/2/11|
|hc2|GEOM4_minus_Direct4|-1.217066|[-1.9559758182713392, -0.47815534276673055]|2/11/3|
|vidstg|PREF4_minus_Direct4|-2.597622|[-5.9293662432086105, 0.7341221010645165]|4/7/5|
|vidstg|PREF4_minus_PREF1|-0.532209|[-2.7523086960457515, 1.687891618218932]|2/3/11|
|vidstg|GEOM4_minus_Direct4|-2.297812|[-5.847765009044733, 1.2521413795564005]|4/7/5|

The comparison file contains all source-paired signs and intervals; ANCHOR_ADMISSION.json reports all128 decisions and per-Query reference counts. No selection on GT, no new independent confirmation, no parameter or Prompt tuning.

## Measured decision

Four-anchor PREF vIoU HC27.357912%/Vid11.974958% remains below revised Direct HC28.580257%/Vid14.572580% (−1.222345/−2.597622pp). Versus the previous single anchor it changes +.115417/−.532209pp. The same-support GEOM control is27.363192%/12.274769%; PREF−GEOM −.005279/−.299810pp, with both paired source t95 intervals spanning0. Therefore this experiment does not establish a semantic-weight benefit, and adding anchors did not close the end-to-end gap. These32 are exposed development and are not new confirmation.

Dense common-time sIoU PREF is50.107393%/32.782728%; revised Direct52.108541%/31.878854%. Direct is not better on every metric: Vid dense geometry favors PREF, despite its lower final vIoU. The result cannot be reduced to a universal semantic-versus-geometry verdict because the remaining interfaces/action distributions differ. All signs/tails and two no-fit sources remain; no tuning, method promotion or new search is triggered. Same-reference parameter-free D2 is17.395617%/8.542736% vIoU in this sparse candidate-reference setting.
