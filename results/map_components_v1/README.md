# Current MAP paper evidence — cache-only components, budget and final DP

This independent study implements the latest direct request to recover the current MAP paper evidence from compatible local caches while the original CG-Vid full keeps running. It does not retrain or call a source model, visual prefix or expert. Frozen production and all old studies/raw keys/results remain unchanged.

## Available evidence and actual scope

| Requested evidence | Local cache finding | This run |
|---|---|---|
| TA Table3 progression, both full directions | 12,205 bound Source/E/native-eight/geometric-candidate/q0/pooled-feature captures and current exact MAP outputs | Actual CPU prediction on HC1902 and Vid10303, whole-domain seal then official score/exact audit |
| Final global DP vs independent choice | Same full per-query geometry and final q1 MAP interval | Included as separate7th arm; original auxiliary path/features/posterior fixed |
| K2/4/8/16 budgets | Old fixed128parent/domain confirmation, shared384 native pretext/postfusion captures and genuine separate joint per-K expert masks/boxes | Hybrid and Uniform,8configs x256queries=2,048 new MAP configurations; rebuild each own geometry/auxiliary/pool/q1 on CPU; old per-K MBR exact replay required |
| Blur/JPEG/Dim | Existing final source/full boxes and intervals, original changed-RGB expert evidence. The shift predictor did not persist source grids, final posterior or geometry | Complete current MAP path is NOT recoverable from these final-box files alone. No new acquisition in this run |
| TubeDETR complete MAP method | No compatible method posterior/geometry/capture set found in the maintained checkout or inspected baseline workspace | Not claimed complete; no new backbone/model call in this CPU run |
| End-to-end deployment timing | Old expert/source/module timing provenance exists, current cache replay measured separately | No new deployment latency claim from cached CPU timings |

## Table3 and readout definitions

| Configuration | Boxes | Time interval |
|---|---|---|
| Original native Source | Original Source | Original per-offset MAP then physical hull |
| Eight-frame replacement | Source except the8observed positions replaced by the exact same native expert boxes | q0 joint physical MAP |
| + Shared correction | Existing source-calibrated global E | Same q0 joint MAP |
| + Spatial inference | Original both-anchor geometry and exact spatial DP | Same q0 joint MAP |
| Complete method | Same fixed auxiliary path/pooled features, revised q1, exact final spatial DP | q1 joint MAP |

All methods use the same8observations and original fixed TA kappa64/beta1/context1.5/spatialweights. A **SourceJointMAP** control explicitly separates the q0 joint-MAP readout from untouched Source. The five main rows report vIoU/R.3/R.5 and vIoU delta vs originalSource, but the change of readout between originalSource and the first adaptive row must remain disclosed. Shared correction is the existing E, not a new calibration. No result-based hyperparameter or query selection.

The final-DP control uses the same candidates, validity, final q1 interval and unary weights, but chooses each frame's valid unary maximum independently, dropping link/path optimization. It does not change auxiliary trajectory, pooled features or q1. It is a targeted final solver contrast, not a strict deletion of every spatial operation or proof of the first DP's effect. FullMAP must reproduce the previously completed MAP arrays/intervals bitwise for every query.

## Budget fidelity and costs

Each Hybrid/Uniform K uses its own original jointly acquired teacher context, boxes and masks; no K8 subset impersonates another K. Strict receipt/input-context/payload-SHA bindings are checked, then CPU reconstruction repeats exact original consensus, geometric candidates, full-video auxiliary DP, actual native-grid repooling and q1. The old MBR boxes/interval must reproduce the corresponding original per-K prediction bitwise before generating MAP+new-intervalDP. Hybrid8 additionally reproduces the completed main MAP output exactly. The original complete confirmation cohort is retained, with no new eligibility ranking. K32/dense failures and original ineligibility remain preserved in the historical study; they are not retuned or resurrected here.

Original measured expert forward costs are kept as acquisition provenance/curve x-axis, not relabeled as new calls or cache deployment acceleration. Current CPU wall is conditional research reconstruction; it excludes uncached source/expert acquisition. Record actual K and T, not only nominal K. Source/load/preprocess/postprocess costs and nested timing limitations stay explicit.

## Isolation and scheduling

Entry `studies/map_components_v1/run.py queue`; outputs/results are separate `map_components_v1`. Two computation processes maximum, one thread each, CPU14-15, nice19 and idle-class3I/O, CUDA hidden. AvailableRAM<8GiB or a precautionary originalVid rolling slowdown pauses only own CPU groups; resume after60s and RAM>10GiB. No CG/controller/teacher/unrelated-service signal or priority change. Low priority is not a mathematical guarantee of zero shared-host interference.

The old automatic MAP dispatcher was safely stopped **without stopping its independent CG-MAP domain child**. That child still consumes immutable originalCG committed caches and performs its original frozen score/report when complete. The new controller takes over resource guarding. Old source/plans/fingerprints/predictions are preserved. The zero-work MBR `strict_deletion_v1` automatic launch is superseded by this latest current-MAP paper scope; its frozen protocol remains an archived, unexecuted MBR study, never mislabeled as MAP. This CPU study launches no future GPU reconstruction.

Actual4case repeated cache preflight covers both tasks and both domains: currentFullMAP or all8own-K originalMBR outputs reproduce exactly; no model/expert/GPU/GT access. A passed bridge is not a whole-domain effect result. Then both full component domains run before the two budget confirmation domains. Separate score subprocesses only read official labels after each full task/domain seal. Failures are preserved without dropping queries, filling source fallbacks, retrying paid acquisitions or editing frozen hashes.

Every signed positive/negative/unchanged outcome, severe t/v union once/rescue, event duration group, pointwise parent-cluster CI, scalar recalls, input/prediction/receipt hashes and worker-PID/start conditionalcost are preserved. All datasets historically exposed. No parameter tuning, unseen-test claim, corruption-robustness claim or universal backbone-module benefit is inferred.

After each score AND full replay audit, root runs `studies/map_components_v1/export.py --task components|budget --dataset hc2|vidstg` to export Table3/final-DP scalars or actual budget curves; this postprocessor reads already-scored metrics only,0newGTscoring/model. Inspect generated privatePNG before publishing the budgetSVG. Publish complete positive and negative results as each domain finishes. Current protocol/startup is not completed experimental evidence.

## Completed HC component domain

TA-HC1902 is fully sealed, scored and replay-audited; [all effects and limitations](components/hc2/ANALYSIS.md), [five-row Table3](components/hc2/table3.tsv), and [matched final-DP contrast](components/hc2/final_DP_control.json) are available. Vid components remain active and budget reconstruction follows both completed component reports. This domain completion does not mean the whole study is finished.
