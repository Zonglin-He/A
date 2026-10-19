# Structured temporal conditional OPD, cached HC128, 2026-10-20

Autonomous temporal exploration; spatial outputs/config, all network weights and
experts frozen. Previous Source448 experiment did NOT pass its >=2pp confirmation
gate: primary+.685pp CI crosses0, best point1.556 also crosses0. Its fixed
HiResMix gave+1.232718pp CI[+.322089,+2.199435] and t+3.323969pp on exposed RC128.
This is a supported subtarget observation/readout signal, not OPD/confirmation.
The posthoc same-student-start teacher advantage also crossed0. New information
is sufficient for one cheap actual mechanism test, not a claim of a strong
conditional teacher. No extra visual inference, no parameter search.

Teacher T=.5p0+.5q448, fixed perquery and stop-gradient. This teacher choice is
a new development decision after the prior results, not a retrospective pass
of the earlier gate. Conditional labels T(start|student_end) and
T(end|student_start) are derived from this constructed interval distribution;
they are not native start-conditioned TA-STVG decoder outputs or calibrated
event probabilities. No RGB refresh is required or pretended.

Student p_eta(i,j) proportional p0(i,j)*exp(u_i+v_j) on the original positive
support; each video has fresh 2N=130 temporary logit biases, no new network/head,
no pretrained update, no across-video state. eta0 returns p0 and original MBR
exactly. Original65 physical endpoints/2080 strict-index actions and finalMBR
unchanged; final spatial arrays are read-only old E/Live41/42/43.

For an actual sampled student interval (i,j), feedback is dense in each decision:
0.5[ KL(T(.|j)||p_eta(.|j)) + KL(T(.|i)||p_eta(.|i)) ]. These are bidirectional
coordinate-conditional labels, not autoregressive GKD replication or a scalar
fixed-box IoU reward. Candidates/teacher targets do not receive gradients.

Outputs: Native, TeacherDirect (same T MBR); DirectJoint exact .5KL(T||p_eta);
DirectConditional exact expectation of the above conditional loss under T;
Fixed samples32 fresh intervals from INITIAL p0 every step; Live samples32 from
CURRENT p_eta. Both fit CURRENT probabilities and share paired uniforms/seeds
41/42/43. Fixed never merely reuses its first32 candidates. DirectConditional
isolates teacher-forced states with exactly the same conditional objective;
DirectJoint is an additional strong full-information baseline, not handicapped
to manufacture OPD benefit. Teacher calls are all cached lookups. A sampled pair
asks two full conditional distributions (up to65 entries each), not32 cheap
scalar rewards; report this count honestly.

One configuration: Adam lr.1,40 updates, default(.9,.999)/eps1e-8, no weight
decay/clipping/temperature/trust sweep. Finite logit-state scale permits odds
changes across the global temporal support; this is not inherited box-policy lr.
Center u and v after each update only to fix the additive probability gauge.
2 deterministic +6 stochastic fits perparent=1024 fits/40960 scheduled updates.
Final p_eta/eta/samples/teacher-table IDs+digests/gradients/update chain retained.
Perquery optimizer resets. Seed-paired temporal/spatial outputs form primary
FixedMean/LiveMean; all cross-seed values auxiliary, not independent parents.

CPU only,4threads,900s finite process; host>=6GiB/disk>=30GiB, no GPU lease or
new source/expert forward. First2 engineering/timing, all128 full fits/predictions
seal before new scoring. Required checks: eta0/no-feedback exact; qTeacher=p0
noop; AD/manual/finite-difference gradients; KL chain rule; conditional versus
joint gradients genuinely differ on a synthetic overlap case; every first-step
Fixed/Live sample/gradient/update exact; later Live sample law actual current
student, all candidate probabilities/teacher lookups match saved states.

All RC128 repeatedly exposed development. Whole vIoU/t/actual-node spatial s,
parent pairedCI, seedmean/tails/cost; no oracle or pervideo branch selection.
Key Live-Native, Live-TeacherDirect, Live-DirectJoint, Live-DirectConditional,
Live-Fixed. Only if Live has meaningful >=2pp net gain and independent advantage
over strong same-information controls is this an OPD route for frozen parent-
disjoint confirmation. Any weaker result remains scoped evidence, no automatic
parameter search, no promotion or claim that all conditional OPD is impossible.
Prior Source448 gate and negative findings remain unchanged. Publish code,
actual config/all signed unknown results/negative tails and byte receipts;
exclude media/GT/text/IDs/weights/raw caches and trace tensor contents.
