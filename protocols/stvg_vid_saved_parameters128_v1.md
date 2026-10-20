# Vid128 saved-step and output-writeback parameter comparison

User scope: evaluate saved steps20/30/40 on the complete existing Vid128; the user then explicitly allowed parameters beyond steps and dataset-specific configurations. Cache decoder replay is permitted, but new RGB/Source encoding and expert forwards are forbidden. No network weights may be trained.

The original Vid runner saved trajectories, temporary states, gradients and expert outputs, but not spatial decoder inputs. Legacy matched-input files for all128 also contain only predictions/observations. No matching complete decoder-input cache was located. Therefore new learning-rate/temperature/policy-scale replay is not claimed or executed. Do not replace missing features with a new Source forward.

This finite matrix uses the existing seed41 paths at steps20/30/40 and one additional executable parameter: output-residual writeback alpha in {0.25,0.5,1}. Nine global configurations plus E+Strong. Alpha is a post-adaptation output mapping parameter, not an optimizer learning rate, a shorter learned state, or a new on-policy rollout. The saved optimization path is identical across alphas.

For saved box b and initialization e, in normalized cxcywh coordinates:

    center(alpha) = e.center + alpha * (b.center - e.center)
    size(alpha) = e.size * exp(alpha * log(b.size / e.size))

Implement alpha1 by an exact copy of b and alpha0 by an exact copy of e. Positive widths/heights, finiteness, frame support and time interval are verified. No clipping or per-video threshold is added. The original E-preserving mapping is retained at alpha1.

All128 original parents and one query per parent, seed41, Source HC2 checkpoint and teacher/evidence/candidates/optimizer/hyperparameters otherwise unchanged. Every configuration uses that parent's original sealed Strong interval. No RGB, model/decoder, expert call, new optimization or network training. All extracted/transformed predictions and source hashes are sealed before new GT evaluation.

This is explicitly exposed development data: all128 final40 outcomes were evaluated previously, and BV041 motivated the requested budget check. The same panel now participates in global configuration selection; it is not independent confirmation. No per-video stop/alpha selection, deletion of harmful samples, or GT intervention in prediction generation. HC remains unchanged and does not have to share Vid's future configuration.

Metrics: complete official vIoU, tIoU, dense-GT sIoU, fixed actual-node sIoU and unobserved actual-node sIoU, identical denominators across configurations. Parent paired bootstrap10000 with seed20261020, signed gains/losses, >3/>5pp harm, all per-parent results, relevant success/failure cases. Report all nine, not just the maximum. Within-panel ranking: maximum complete-vIoU, exact ties prefer alpha1 then fewer steps; ranking is a development candidate, not automatic method promotion. Comparisons with E+Strong, original40/alpha1, and adjacent alpha1 steps are predeclared. Other pairwise comparisons are descriptive. Bootstrap intervals are not adjusted for configuration search.

Independent checks: alpha0/E and alpha1/original identities; saved path0/E and path40/published Live identities; original E and40 metrics reproduce the prior result; official and DenseTube metrics agree for every output; final parent-level aggregation and bootstrap independently recomputed; original predictions/traces/results/configuration and registries unchanged.

Cache extraction/scoring and any original inference costs are separate. Zero new inference does not mean deployment experts or Source are free. No new shorter-budget deployment timing is claimed from these saved paths. Code, fixed protocol, all signed results, limitations and negative findings go to Zonglin-He/A under the existing allowlist, followed by remote bytes/hash/tree and dual-ref CAS receipts. No automatic CURRENT/MAIN/PAPER promotion or further grid expansion.
