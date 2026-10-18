# Fixed32 spatial candidate ranking and six-layer readout

Same exposed16 sources/domain as the previous matched Source-episodic32, original domain OPD budgets and fixed Native time. Two changes tested separately; not combined, no new model training. Original OPD states/final outputs replay exactly. Raw DINO outputs recaptured on original four frames, same tensors feed both ranking pipelines; original probes and input hashes reproduce exactly.

| Domain | Arm | vIoU % | tIoU % | sIoU % | Unobserved actual decoder sIoU % |
|---|---|---:|---:|---:|---:|
| hc2 | Source | 18.831315 | 45.792579 | 37.714391 | 37.712875 |
| hc2 | original_OPD | 18.471518 | 45.792579 | 35.073558 | 35.193196 |
| hc2 | target_rank_OPD | 18.490575 | 45.792579 | 34.933026 | 35.069864 |
| hc2 | layer_select_original_state | 18.471518 | 45.792579 | 35.073558 | 35.193196 |
| vidstg | Source | 7.147781 | 31.818184 | 36.753536 | 37.113932 |
| vidstg | original_OPD | 11.496033 | 31.818184 | 47.183985 | 47.390089 |
| vidstg | target_rank_OPD | 11.496033 | 31.818184 | 47.183985 | 47.390089 |
| vidstg | layer_select_original_state | 11.496033 | 31.818184 | 47.183985 | 47.390089 |

All signed per-source paired differences, damage tails, R03/R05 and intervals are in RESULTS;10000 paired parent bootstrap, unadjusted. Same initial Source1792 and original LR/steps/noise/tau/seed; no inherited LN. Ranking arm actually re-fits original OPD; layer arm fixes original adapted state, selecting exactly one whole real layer trajectory with original accepted DINO references. Six layers are not independent/calibrated experts. No reference defaults to layer6. Both complete decoder passes and cached six-layer outputs agree.

Candidate diagnosis uses exact annotated physical anchor frames only; no GT box interpolation and no filtering/re-ranking by GT online. Raw best geometric overlap is an oracle availability diagnostic, not evidence a useful semantic selector can find it. NMS and Top3 attrition are separated. Geometry IoU cannot certify correct person; all identity labels unknown. Event-outside frames keep unknown GT and remain part of deployed observations.

Whole-six-layer GT oracle is postseal diagnosis only; it never supplies the selected layer. WBF/coordinate fusion and stacking the two changes are not run. These small development results cannot promote a full method or prove universal conclusions.

## Results and decision

Target-score-consistent ranking changes the admitted reference set in only3/32
queries (all HC). HC vIoU +0.019057pp, paired95% interval
[-0.125330,+0.158102];2 improve,1 deteriorates,13 unchanged. Vid all16
complete outputs are unchanged.29 unchanged-reference queries replay the
ENTIRE fit/final boxes exactly, a strong no-change control.

At exact GT positions in the contextual branch, raw/NMS/Top3 coverage with
at least one IoU>=.5 candidate is HC48/48/33 out of49 slots under the old
ranking, and48/48/36 under target ranking. Vid is25/25/23 out of25 and
25/25/24 under target ranking. Thus Top3 loses geometric availability in
some cases; NMS removes many individual raw-best IDs but does not eliminate
ALL >=.5 boxes in these slots. Counting eliminated raw-best IDs alone would
overstate NMS as the bottleneck. Recovered pool quality does not guarantee
selection/admission: actual selected-reference successes go20->21 for HC
(over49 GT slots),11->11 for Vid(over30 GT slots including5 noncontext/
no-call slots). No identity conclusions are inferred from these IoUs.

Whole-layer selection picks layer6 in15/16 HC and16/16 Vid. The one HC
layer1 choice changes no final metric; all32 per-source final metrics remain
unchanged. The postseal six-layer GT oracle has only HC+1.156328pp and
Vid+0.600988pp mean vIoU headroom over layer6; it is diagnostic, not an
online rule. This does not establish a temporal-MBR-sized hidden spatial
readout gain or justify expanding either variant now.

All arms use fixed Native time. HC unobserved actual-frame sIoU decreases
35.193196->35.069864% under target ranking; Vid stays47.390089%.
The complete32 cohort is historically exposed, not independent confirmation.

124 actual DINO recaptures for128 planned positions; one query lacks a legal
detector input and its4 slots/no-op remain in denominators. Cumulative model
lease accounting268.255869s includes the failed worker's181.932306s upper
bound plus resumed work; not pure GPU kernel or deployment latency. Original
OPD fits are reused exactly; changed-reference OPD actually reruns. A cleanup
error after20 sealed outputs was repaired without replaying those outputs,
changing scientific logic, deadline or resetting budget.

Actual output pixels were viewed in3 unique private panels/96 cells;6 rendered
selection panels include duplicates. Query-specific identity labels remain
unknown. Original params/registries retained; no combined variant, WBF or
automatic parameter search follows this finite pilot.
