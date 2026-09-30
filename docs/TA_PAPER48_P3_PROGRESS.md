# Paper48 P3: all three expert budgets completed

The prespecified64-source panel has completed all three expert budgets (0%,25%,100%), with768 arrivals per arm and2304 total. The original frozen configuration remains unchanged; these results do not trigger tuning.

| Expert availability | Corruption whole-stream Δdense vIoU pp [95% source-bootstrap CI] | Corruption nonexpert Δdense vIoU pp [95% CI] |
|---|---|---|
|0%|0.0000 [0.0000,0.0000]|0.0000 [0.0000,0.0000]|
|25%|−0.0477 [−0.5050,+0.3118]|+0.0356 [−0.0006,+0.0746]|
|100%|+0.2113 [−1.7555,+2.0614]|No nonexpert arrivals by definition|

At0%, Ours equals Frozen exactly in the runtime prediction checks and all768 scored arrivals, with0 specialist calls and0 updates. This verifies the no-expert behavior, not an effectiveness gain.

At25%, the overall corruption mean is slightly negative and its interval crosses zero. The two order-specific whole-stream gains are−0.2355/+0.1402pp. The nonexpert readout has480 cells and61 sources, with order-specific gains+0.0214/+0.0565pp; its source-weighted overall CI still crosses zero. Clean whole-stream gain is−0.2466pp [−0.9099,+0.1663], and clean nonexpert gain is+0.0363pp [−0.0020,+0.0767]. Thus this64-source25% arm does not establish a positive overall or future-transfer effect.

The positive larger P1/P2 results and this smaller-panel result are all retained. P3 is a nested64-source subset with independent64-query state resets and restricted order/schedule; changing the subset also changes inherited-state histories. Cross-panel differences do not isolate stream length, and the completed budget response does not establish a monotonic benefit from increasing expert availability.

At25% on corrupted arrivals,1/64 source means loses>5pp and none loses>20pp;8/640 cells lose>5pp and3 lose>20pp. No nonexpert cell loses>5pp. The full negative tails and both orders are public, along with clean and paired excess readouts.

All three completed arms use64 hash-selected sources/one query per source, two fixed orders, clean plus five5% corruption families,768 arrivals per arm. Vid-trained TA checkpoint5ab12c86, J0.1 spatial1792 parameters, SGD.005/one step,9 current probes, pre-update current output and persistent state within a stream remain fixed. Each arm and condition/order resets independently. At25%, both specialists are scheduled every fourth arrival; at100%, both are scheduled on every arrival. The sources overlap P1/P2 and have historical project exposure.

Each arm sealed all predictions before GT scoring. Runtime audits, root verification of all768 receipt/payload hashes and24 spatial reinsertion endpoints, and independent public scalar/source/bootstrap checks pass. Full precision results are in their respective directories. No new GPU inference or GT scoring is performed during publication auditing. All raw predictions, GT coordinates, media, weights and learned states remain private. P4 and required P5 follow the budget arms; no total deadline is imposed.

## 100% budget: measured result and uncertainty

Corruption whole-stream dense m_vIoU is19.3138% Frozen versus19.5251% Ours. The mean gain is+0.2113pp, with95% source-bootstrap CI[−1.7555,+2.0614]; both order means are positive (+0.2206/+0.2019pp), but the source-level interval is broad and crosses zero. There are no nonexpert arrivals, so this arm cannot measure expert-free future transfer. Dense spatial IoU improves+0.3772pp [+.1115,+.6295], while temporal IoU changes+0.5470pp [−3.3796,+4.2235]. These component readouts are not substituted for the joint endpoint.

Clean whole-stream ΔvIoU is−0.4260pp [−2.6170,+1.4488]. The paired corruption-minus-clean gain is+0.6373pp [−0.2810,+1.7737], also inconclusive. On corrupted inputs,6/64 source means lose more than5pp and3/64 lose more than20pp;58/640 cells lose more than5pp and24/640 more than20pp. The negative tails remain part of the result.

All768 records and state links,742 actual SGD updates,768 teacher checks,1536 independent dense metric checks (maximum discrepancy0),24 full spatial reinsertion endpoints and17251 public scalar/bootstrap checks pass. All predictions were sealed before GT scoring. Online process time was1152.286908s; it excludes cached specialist generation and is not end-to-end deployment latency.

## P4 engineering handoff

P4's first Frozen cold-load process failed before producing predictions or timing rows: a direct model_load import preceded installation of the existing no-GT loader wrapper. The process-wide annotation guard correctly blocked the constructor's annotation open. The repair moves the binding after wrapper installation and asserts annotation_files_opened=False. It changes no model, data, schedule or timing definition. Original logs and source are preserved under recovery/P4_loader_import_001, and implementation revision006 pins the repair. P4 must complete and be audited separately; P5 remains required.

A subsequent root check compared the first cold Frozen output against the saved same-input Paper48 source-native output, without GT. The raw AMP path differed by up to0.0003591 in box coordinates and decoded indices[11,66] versus[13,66]. Its12 timing rows (101.284181s) and the controlled interruption were preserved under recovery/P4_precision_002, excluded from final efficiency aggregation. Revision007 applies the already established inserted_state(model,{}) precision hooks: source parameters remain unchanged and the post-encoder calculation is FP32, matching the fixed main-experiment interface. Frozen timings are regenerated from zero. This is an interface-consistency repair, not an adaptation change.

The repaired first uncached Frozen output is bitwise identical in all boxes and identical in decoded indices to the saved same-input P1 source-native prediction. This readback uses no GT and adds no inference; broader P4 audit remains due on completion.
