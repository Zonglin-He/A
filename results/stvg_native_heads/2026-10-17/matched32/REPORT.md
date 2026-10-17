# Fixed-evidence native head interface32

Same exposed HC16/Vid16, cross-domain TA-STVG checkpoints; frozen R1 Direct references and B3 I1/I2/rewards unchanged. No new Teacher calls or candidate generation. No independent confirmation.

Temporal adapter at native temp_embed input is exactly the existing256x2 residual on this backbone. This axis is an explicitly verified identity, not a distinct model experiment; all fitted weights and full-grid outputs reused. Spatial adds a zero-initialized shared bias-free256x4 residual at the actual final bbox Linear input in the last layer of the final native decoder pass. Original Query/LN parameters frozen. Original Gaussian32, domain settings, Adam/response/fullstep acceptance/rollback/early stop unchanged.

Four cells are controlled fixed-evidence readouts: original interfaces, represented Temporal only, native Spatial only, both native heads. Spatial observations retain the sealed T2 clock for every cell; combinations use the same fixed B3 time. They are not newly generated B3-clock end-to-end Full. Separate fixed-T2-clock spatial rows isolate box-interface effects. Source marginal no-update is kept. No decoder-only increment is counted as adaptation.

|Domain|Arm|vIoU %|dense sIoU %|tIoU %|
|---|---|---:|---:|---:|
|hc2|Frozen|28.116723|50.497117|52.754115|
|hc2|MARGINAL_NO_UPDATE|24.576445|50.497117|48.654998|
|hc2|B3_TEMPORAL|24.713955|50.497117|49.022568|
|hc2|ORIGINAL|25.654075|52.108541|49.022568|
|hc2|TEMPORAL_HEAD|25.654075|52.108541|49.022568|
|hc2|SPATIAL_HEAD|23.492160|46.677584|49.022568|
|hc2|BOTH_HEADS|23.492160|46.677584|49.022568|
|hc2|ORIGINAL_S_FIXED_CLOCK|28.580257|52.108541|53.666875|
|hc2|NEW_S_FIXED_CLOCK|25.658712|46.677584|53.666875|
|vidstg|Frozen|12.185607|32.870500|35.923576|
|vidstg|MARGINAL_NO_UPDATE|11.914150|32.870500|35.236262|
|vidstg|B3_TEMPORAL|11.925210|32.870500|35.215982|
|vidstg|ORIGINAL|14.170323|31.878854|35.215982|
|vidstg|TEMPORAL_HEAD|14.170323|31.878854|35.215982|
|vidstg|SPATIAL_HEAD|13.488011|30.993251|35.215982|
|vidstg|BOTH_HEADS|13.488011|30.993251|35.215982|
|vidstg|ORIGINAL_S_FIXED_CLOCK|14.572580|31.878854|35.680949|
|vidstg|NEW_S_FIXED_CLOCK|13.862596|30.993251|35.680949|

All source metrics, anchor/nonanchor unknowns, all signed paired deltas and nominal t95 intervals remain in RESULTS.json. No mandatory significance gate. Actual gradient/finite-difference, zero recovery, complete Source/updated native outputs and reversible nonanchor influence checked on the fixed first parent/domain; these are engineering checks, not semantic labels. The semantic512 panel is still44 model-assisted/468pending. CURRENT/PAPER unchanged.


Engineering recovery: native head outputs are cxcywh. The first scorer invocation incorrectly received that native format as xyxy and stopped; no metric table from that failure is used. Raw outputs and failure retained. A derived CPU readout converts saved cxcywh to normalized xyxy exactly, without clamping or changing any model/fitted state; all outputs resealed before successful offline scoring. New model/fitting repeats0.

At fixed T2 clock, native box head vIoU is25.658712/13.862596 versus old Query+LN28.580257/14.572580 (HC/Vid). This fixed no-retuning configuration does not support replacing the old Spatial interface on quality alone; nonanchor influence and simpler1024 parameters do not prove identity correction or comparable accuracy. Temporal interface is unchanged by construction. No deployment promotion and no automatic adapter tuning.
