# R1 semantic preference versus geometric weights: matched32 Student effect

Complete exposed HC16/Vid16, fixed slot0 one anchor per Query. Original32/4-anchor Direct is a different observation setting and is not used as this matched control. Existing first8 preference outputs are exactly reused; only24 more parents/48 additional native forwards,64 total real R1 forwards. No GT enters input/candidate/Teacher/Student/acceptance. Both Student arms and parameter-free D2 are complete and rooted before evaluation.

GEOM and PREF use the SAME four frozen WF candidate atoms, SAME hard reference (argmax of two-order averaged R1 preference, with0 abstention), SAME total weight1-p0, original1792 episodic Student/Adam/domain sigma-lr-tau-steps/full-grid response/fullstep reference-IoU acceptance, rollback/reject-stop and full-frame decoder readout. GEOM uses softmax candidate IoU to that hard reference; PREF uses the actual candidate preference mass. This is fixed-support Gaussian distillation, NOT on-policy32 sampled-action OPD. No clamping: an invalid open-logit fixed atom or incoming Source anchor makes the entire Query a common Source no-op; abstentions remain Source in all three methods. All32 stay in the denominators.

Root reconstructs support/weights/likelihood/mean gradient independently and checks saved Adam/response/acceptance/state chain. The original root-only fixed5e-6 absolute Adam moment comparison failed on large fixed-support gradients; all32 GEOM predictions and Teacher outputs were preserved. The corrected independent audit uses pointwise gamma8 float32 operation-roundoff bounds for the moments, retains strict5e-6 parameter-delta error, and rejects a deliberate incorrect-variance control. No model, update, precision, acceptance or prediction changed; failed root cost remains recorded. Source and temporal logits are preserved; final cached and actual complete decoder trajectories must be exact. D2 uses same hard reference/admission and Source-relative center/log-size physical-time residual propagation inside T2, Source outside, with zero parameter update.

| Domain | Arm | vIoU % | dense common-time sIoU % | tIoU % | nonanchor sIoU % | R@.3 % | R@.5 % |
|---|---|---:|---:|---:|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|50.475404|31.250000|25.000000|
|hc2|T2|27.487135|50.497117|53.666875|50.475404|37.500000|18.750000|
|hc2|GEOM|27.182523|49.785271|53.666875|49.762593|31.250000|18.750000|
|hc2|PREF|27.242495|49.849448|53.666875|49.827683|31.250000|18.750000|
|hc2|D2|21.557015|39.438650|53.666875|39.436310|25.000000|12.500000|
|vidstg|Frozen|12.185607|32.870500|35.923576|32.845236|12.500000|0.000000|
|vidstg|T2|12.239461|32.870500|35.680949|32.845236|12.500000|0.000000|
|vidstg|GEOM|12.636913|33.644014|35.680949|33.623741|12.500000|0.000000|
|vidstg|PREF|12.507167|33.373820|35.680949|33.353395|6.250000|0.000000|
|vidstg|D2|11.274466|31.559233|35.680949|31.542239|6.250000|0.000000|

| Domain | Contrast | vIoU delta pp | paired source t95 pp | + / - / neutral / unknown |
|---|---|---:|---|---|
|hc2|GEOM_minus_T2|-0.304612|[-0.7395444896951449, 0.1303206633063957]|1/3/12/0|
|hc2|PREF_minus_T2|-0.244640|[-0.6425337782907039, 0.15325421726695038]|1/2/13/0|
|hc2|PREF_minus_GEOM|0.059972|[-0.25233067549843585, 0.37227494086343144]|1/3/12/0|
|hc2|D2_minus_T2|-5.930120|[-12.392257459087675, 0.5320176369903287]|0/5/11/0|
|hc2|D2_minus_PREF|-5.685480|[-11.96146388966806, 0.5905036285944671]|0/5/11/0|
|vidstg|GEOM_minus_T2|0.397452|[-1.5286565162890489, 2.3235611840125134]|1/2/13/0|
|vidstg|PREF_minus_T2|0.267706|[-1.6855681881834321, 2.220980173875695]|2/2/12/0|
|vidstg|PREF_minus_GEOM|-0.129746|[-0.3543593403112851, 0.09486665828008323]|1/3/12/0|
|vidstg|D2_minus_T2|-0.964995|[-2.5564847424020867, 0.6264942800535329]|1/3/12/0|
|vidstg|D2_minus_PREF|-1.232701|[-2.7190711728386505, 0.253668724797834]|0/4/12/0|

All accepted/rejected/no-fit reasons and per-source signed/unknown deltas are in RESULTS.json. Paired intervals describe exposed-development uncertainty; no posthoc source exclusions, domain splicing, threshold tuning, new independent confirmation or method promotion. Candidate identity/role correctness remains unlabelled; posterior calibration and candidate coverage are not inferred from this effect. Original semantic512 panel remains44 model-assisted/468 pending. The full two-Teacher2x2 is not complete until Qwen matched cells actually run.
