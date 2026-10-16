# Caption availability decomposition: core256 and confirmation128

Saved sealed predictions only, no model calls or new GT-reader calls. Core256 and confirmation128 are reported separately. The primary split is the Caption status captured before Spatial Teacher selection; successful update status is descriptive only. All sources and denominators remain. Differences are vIoU percentage points.

| Cohort | Domain | Contrast | Subset | N | Subset mean pp | Signed overall contribution pp |
|---|---|---|---|---:|---:|---:|
|core256|hc2|B_minus_A|available C|120|+0.181139|+0.169818|
|core256|hc2|B_minus_A|missing M|8|-0.140706|-0.008794|
|core256|hc2|B_minus_Temporal|available C|120|+0.379088|+0.355395|
|core256|hc2|B_minus_Temporal|missing M|8|+0.000000|+0.000000|
|core256|vidstg|B_minus_A|available C|101|+0.544736|+0.429831|
|core256|vidstg|B_minus_A|missing M|27|+0.219081|+0.046212|
|core256|vidstg|B_minus_Temporal|available C|101|+1.024265|+0.808209|
|core256|vidstg|B_minus_Temporal|missing M|27|+0.000000|+0.000000|
|confirmation128|hc2|B_minus_A|available C|61|-0.123254|-0.117477|
|confirmation128|hc2|B_minus_A|missing M|3|-1.123250|-0.052652|
|confirmation128|hc2|B_minus_Temporal|available C|61|+0.479801|+0.457311|
|confirmation128|hc2|B_minus_Temporal|missing M|3|+0.000000|+0.000000|
|confirmation128|vidstg|B_minus_A|available C|48|+0.091881|+0.068911|
|confirmation128|vidstg|B_minus_A|missing M|16|-1.293340|-0.323335|
|confirmation128|vidstg|B_minus_Temporal|available C|48|+0.565053|+0.423790|
|confirmation128|vidstg|B_minus_Temporal|missing M|16|+0.000000|+0.000000|

Confirmation HC: both Caption-present and Caption-missing contributions are negative, so lack of opportunities is not the sole explanation. Confirmation Vid: present-Caption B minus A is +0.091881pp (overall contribution +0.068911), but missing-Caption contribution -0.323335pp reverses the total. This is descriptive attribution of saved outcomes, not causal identification of an optimal selection policy.

B confirmation no-fit20 =19 missing-Caption parents (3 HC/16 Vid,76 missing-window receipts) plus1 Vid source with no legal/parsed non-Source reference. The other14 fitted but finally unchanged cases are different:3 HC no-information/no-proposal and11 Vid all proposed steps rejected. They must not be added to the no-fit20 or equated with76 windows.

| Cohort | Domain | Caption | Arm | N | Stage outcomes |
|---|---|---|---|---:|---|
|core256|hc2|True|A|120|{"accepted_nonzero_final": 111, "attempted_all_proposals_rejected": 3, "attempted_no_proposal_no_information_or_weight": 5, "no_legal_or_parsed_nonSource_reference": 1}|
|core256|hc2|True|B|120|{"accepted_nonzero_final": 112, "attempted_all_proposals_rejected": 1, "attempted_no_proposal_no_information_or_weight": 6, "no_legal_or_parsed_nonSource_reference": 1}|
|core256|hc2|False|A|8|{"accepted_nonzero_final": 7, "attempted_no_proposal_no_information_or_weight": 1}|
|core256|hc2|False|B|8|{"caption_unavailable_no_WHO_no_reference": 8}|
|core256|vidstg|True|A|101|{"accepted_nonzero_final": 80, "attempted_all_proposals_rejected": 17, "no_legal_or_parsed_nonSource_reference": 1, "reference_not_admitted": 3}|
|core256|vidstg|True|B|101|{"accepted_nonzero_final": 79, "attempted_all_proposals_rejected": 16, "attempted_no_proposal_no_information_or_weight": 1, "reference_not_admitted": 5}|
|core256|vidstg|False|A|27|{"accepted_nonzero_final": 19, "attempted_all_proposals_rejected": 7, "reference_not_admitted": 1}|
|core256|vidstg|False|B|27|{"caption_unavailable_no_WHO_no_reference": 27}|
|confirmation128|hc2|True|A|61|{"accepted_nonzero_final": 61}|
|confirmation128|hc2|True|B|61|{"accepted_nonzero_final": 58, "attempted_no_proposal_no_information_or_weight": 3}|
|confirmation128|hc2|False|A|3|{"accepted_nonzero_final": 3}|
|confirmation128|hc2|False|B|3|{"caption_unavailable_no_WHO_no_reference": 3}|
|confirmation128|vidstg|True|A|48|{"accepted_nonzero_final": 38, "attempted_all_proposals_rejected": 10}|
|confirmation128|vidstg|True|B|48|{"accepted_nonzero_final": 36, "attempted_all_proposals_rejected": 11, "no_legal_or_parsed_nonSource_reference": 1}|
|confirmation128|vidstg|False|A|16|{"accepted_nonzero_final": 16}|
|confirmation128|vidstg|False|B|16|{"caption_unavailable_no_WHO_no_reference": 16}|

The JSON retains every window reference admission reason, nonfinite/reference-rule proposal rejection, no-information skipped step and per-source final state. A selected Source candidate is still forbidden by the original contract; this rule is retained. Missing Caption must not be called a model parse error merely because the generic anchor converter emits that reason.

Whole-query missing-Caption fallback-A is an offline scalar diagnostic selecting one complete independent query prediction. It does not splice boxes, states or windows and is not a new independent confirmation. Diagnostic strategy minus A: core HC+0.169818/Vid+0.429831pp; confirmation HC-0.117477/Vid+0.068911pp. It remains separate from noCoverage.

Next model work is only the original noCoverage deletion on the complementary224 core development parents. Primary contrast is against matched A; secondary against B. Original32, additional224 and full256 are separate; all old confirmation conclusions remain. No Global32/C/BT/Prompt/LR/budget search, CURRENT change or official fullset. Semantic512 arm-blind review is separate and only actual viewed pixels may support labels.

Public exports exclude private media/annotations/GT coordinates, weights/raw tensors/optimizer states, actual query-caption text/Teacher raw responses, credentials and personal conversation. Whole-project FINAL is not claimed.
