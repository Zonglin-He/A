# Cached diagnosis: why24 states did not beat the center

Scope: only existing new-parent HC16/Vid16 from8089d3dc. Same checkpoints/E/final-MBR/four anchors/aligned content-mask/.35/24 denominator. No model, media, weights, annotations, new GT scoring, new predictions, hypothetical output arms, parameter search or new sample. Do not start dynamic OPD. Preserve all32 and unknowns. CPU budget600s.

Let e=z(E), c=z(Tcenter), t=mean accepted sampled z(T), p=n/24 and M=accepted center. Actual displacements are Uc=M(c-e), Up=pUc and Us=p(t-e), with Us=0 if n=0. Independently check saved output arrays and identities:

Us-Uc=(Up-Uc)+(Us-Up).

On common-admitted nodes, Up-Uc=(p-1)(c-e) is admission scaling, Us-Up=p(t-c) is endpoint difference. Outside common support the latter also contains sampled-only admission; do not call it a pure endpoint-quality effect. Center-only nodes have p0 and are turned off in Center_p. Same p is not matched displacement norm. Nonlinear box decoding/IoU/interpolation prevent additive causal attribution of task gains to vector norms.

Use ONLY already saved per-query four-arm metrics to compute exact arithmetic path:
(Samples24-Center_full)=(Center_p-Center_full)+(Samples24-Center_p).
Report paired parent/query bootstrap10000 seed20261018, full16/domain, common-admitted saved sIoU, no-support/unknown and all signed outcomes. Split existing center-benefit/harm/no-change query groups descriptively; labels do not enter any deployment rule.

Read saved native and conditional traces to quantify accepted teacher endpoint similarity to the center, source-scaled dispersion relative to admitted input references, endpoint-direction cosine and actual displacement norms. These are geometry, not GT accuracy or identity judgments. No cached GT score exists for each new per-state endpoint or its unshrunk mean: mark those qualities unmeasured, do not read annotations to fill them.

Analyze p=(legal_states/24)*(accepted/ legal_states), the same whole-state legality across four anchors. Report both legal fraction and acceptance conditional on legal states, support overlaps and magnitude suppression. p is not a correctness probability. Distinguish active/planned/common denominators.

Use the top three positive and negative Samples24-Center_full queries per domain selected mechanically from saved differences; retain all32 signed cases. Save no raw coordinates/hidden states/query text/private parent identifiers in public export. Independently audit exact vectors, supports, p factorization, saved score identities/aggregates, raw-trace mappings, and unchanged inputs. Archive and verify publication. No further experiment is automatically authorized by this diagnosis.
