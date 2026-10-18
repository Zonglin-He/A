# Teacher-endpoint versus transported-residual control, original32

Question: remove the admitted-input displacement term while preserving teacher
outputs, admission, sample count and output support. This is a changed estimation
target, not a presumed correction and not evidence of OPD superiority.

Freeze the original exposed HC16/Vid16 roster, Source/E, four planned decoder
nodes, aligned content/mask teacher feedback from505103c1, all24 states and
admission masks,0.35 threshold, legality rules, source coordinate representation
z=(cx,cy,log w,log h), and final-distribution MBR interval. Reuse E and the exact
d9f0c700 Independent4_observed output as Residual_local. Generate only:

    Endpoint_local_i = decode(z(E_i)+sum_k m_ik*(z(T_ik)-z(E_i))/24).

No new teacher/expert/model/feature calls, network updates, GT-based admission,
step-size search, extra states or new samples. All other decoder nodes remain
bit-exact E. Unsupported queries and no-feedback nodes are exact no-ops and
remain in every denominator. Identity teacher T=sample does not generally imply
endpoint no-op; this different property must be checked and reported.

Seal all32 full outputs and input hashes before new GT scoring. Evaluate complete
vIoU/tIoU/sIoU, observed/unobserved actual decoder-node sIoU and benchmark
interpolation separately, with exact additive vIoU contributions. Missing GT is
unknown. All tIoUs must match and unobserved actual nodes must remain exact E.
Report Endpoint-E, Residual-E and Endpoint-Residual paired differences,10000
parent bootstrap intervals(seed20261018), positive/negative/neutral and >5pp
losses, as well as actual center/log-size displacement. Same coefficients do
not guarantee equal actual movement. All old signed cases are retained.

Outcome interpretation: endpoint gain with residual decline supports changing
the target within this cohort; no HC endpoint gain means removing this input
term alone did not solve HC; similar outputs weaken assigning the main loss to
this term. These are development findings, not independent confirmation. This
finite run ends at the cached comparison, independent audit and publication.
It does not run a new cohort, center-reference teacher calls or dynamic OPD;
those later comparisons need a separately locked roster and rule.
