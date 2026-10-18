# Measured interpretation: local residual update

The three requested outputs are complete. This round does not start dynamic OPD.

VidSTG: Independent4_observed improves complete vIoU over E by +0.097184 pp
(95% paired parent bootstrap [+0.026135, +0.181096]), and over the same
shared4 restricted to the four nodes by +0.092743 pp [+0.020446, +0.181436].
This is evidence that keeping per-anchor residual information helps in this
exposed development cohort. Restricting the old shared4 to four nodes recovers
+0.051691 pp over global application, while adding only +0.004441 pp over E.
Eight queries improve, one worsens and seven are neutral. No complete-vIoU
loss exceeds 5 pp.

HC: Independent4_observed remains -0.030264 pp below E
[-0.063122, -0.001052], with 3 positive / 7 negative / 6 neutral queries.
Its +0.003659 pp advantage over local shared4 has an interval crossing zero.
Restricting application reduces the old global harm, but retaining local
residuals does not establish useful new feedback beyond E here. This result
cannot be presented as a successful bidirectional method.

For both local arms, every unobserved actual decoder box is bit-exact E, so
unobserved actual-node sIoU change is exactly zero. Vid Independent4's complete
+0.097184 pp comprises +0.016448 from observed actual nodes and +0.080736
from official benchmark xyxy interpolation. HC's -0.030264 comprises -0.003265
and -0.026999 respectively. These are exact additive metric contributions,
not evidence of a newly learned visual propagation mechanism.

Observed-node sIoU for Independent4 vs E is HC -0.291001 pp
[-0.598312, -0.020389] and Vid +0.945308 pp [+0.097641, +1.778138].
HC observed-node support is 16/16 queries. Vid support is 14/16, consisting of
10 active queries and 4 original unsupported/no-update queries with known GT.
V025 and V042 have no GT at planned nodes and remain unknown there; both
are retained in all16 complete-tube results. The separate active-known endpoint
and its denominator are saved in ROOT_DISPLACEMENT_SUPPORT_AUDIT.json. It is
not interchangeable with prior sample-level teacher-quality accounting.

Update strengths differ. Mean observed center displacement (normalized image
units) for global/shared-local vs independent-local is HC 0.004444 / 0.002837,
Vid 0.003037 / 0.004007. Corresponding log-size norms are HC 0.025438 /
0.014676 and Vid 0.015427 / 0.020298. No matched-amplitude intervention was
run, so the observed contrast is between the requested aggregation rules,
not proof of an amplitude-independent effect.

Retain aligned content/mask and E as the research basis. The cached experiment
supports local residual retention on Vid only; HC does not improve. Do not
resume global shared4 multi-round updates or promote a universal local method
from these results. No threshold/step-size search, GT selection or new teacher
calls occurred. Current production registration remains unchanged.
