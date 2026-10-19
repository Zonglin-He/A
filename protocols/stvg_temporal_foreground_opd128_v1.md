# R19 Raw Action temporal conditional OPD, HC128 2026-10-20

Question: does the existing task-trained dense foreground evidence transfer better
through exactly the R7 conditional OPD student? Only teacher input changes from
R7 .5Source224+.5Source448 endpoint posterior. No new expert, checkpoint, RGB,
network training or spatial change. R18 information qualification Action dense
AUROC .809147, same-node-length pair .757093(n127), benefit/harmAUC .681308(n105)
and fixed wrong-parent controls support one experiment, NOT calibrated interval
probabilities or 2–3pp gain. R18 auxiliary rho invalid and excluded; post-main
donor control descriptive, not independent confirmation. RC128 repeatedly
exposed, TC109 not used for selection. Same Vid TA-STVG checkpoint and frozen
ReferDINO spatial config/arrays; original DINO/Refer8 evidence stays unchanged.

Target on original positive p0 support: logT(i,j)=sum_{t=i..j} raw Action[t],
i<j; normalize once in FP64 logsumexp. p0 is ONLY support for T, not multiplied
into T; student remains p_eta ∝p0 exp(u_i+v_j). No EOS shift, temperature,
strength mixture, physical integration weight or head search. This is existing
Bernoulli mask algebra/additive endpoint potentials, not a new semantic source
or independence/calibration guarantee. Float64 target must be finite and
strictly positive on p0 support; otherwise abort and preserve partials, no
posthoc clipping or added epsilon. Target and sampled candidates stop-gradient.

Exactly reuse original R7 helper without edits: 130 reset biases, Adam .1,
40 steps, no weight decay or clipping, gauge centering. 32 fresh intervals per
step, seeds41,42,43. Fixed samples from INITIAL p0 each step with fresh paired
uniforms; Live from CURRENT p_eta. Both update current student probabilities.
For each actual sampled(i,j), teacher returns full T(start|j), T(end|i), not
fixed-box IoU. Teacher is static but evaluated at actual student states; no
fictional visual refresh. First step Fixed/Live samples/gradients/update exact.
Strong DirectJoint=.5KL(T||p_eta), DirectConditional full expectation of same
conditional loss under teacher marginals. Do not handicap Direct. No learning
rate/step/policy sweep or gating OPD by hard teacher-MBR results.

Outputs Native, Raw TeacherDirect MBR, DirectJoint, DirectConditional, Fixed/Live
41/42/43 (1024fits/40960 scheduled updates), plus old HiResAuxMix strong static
reference and WrongTeacherDirect static control from sealed R18 wrong-parent
energy. Wrong only a control, never eligible. Same finalMBR/65physical ids/
2080strict-index actions and E/Live41/42/43 spatial trajectories exact. Strong
reference numeric scores reproduce original R14 values. No new spatial fit.
Primary is same-seed temporal/spatial paired, then mean within parent. Multiple
seeds are not independent parents. Preserve all128, null/zero/negative tails.

All fits and predictions globally sealed BEFORE any new GT scoring. Independently
replay all NumPy gradients/Adam/sample laws/teacher conditional digests beforeGT;
zero/no-feedback synthetic controls and old helper finite differences. Verify
all target energies with independent scalar sums/softmax; original Native
MBR exact, spatial arrays exact, official/Dense metrics. Preserve all state
updates/samples/gradients/target digest/outputs privately. Public code/config,
all signed/unknown perquery metrics/cost/tails but no private texts/IDs/media/
annotations/weights/raw caches. CPU4threads <=900 seconds per process including
fits (audit/score separately bounded), first2 timing, run.lock nonblocking,
host>=6GiB/disk>=30GiB. No GPU lease held for CPU task.

Report expected candidate GT fullv/t quality of RawTeacher vs prior/Wrong/old
strong posterior as offline diagnostic AFTER prediction seal; no oracle online
selection. Hard readout failure is not automatically OPD failure; teacher
expected improvement is not final gain either. Main fullv/t/boundary/unchanged
actual-node s, paired parentCI, seeds, severe tails/LOO/cost. Decisions:
Live >=2pp net with positiveCI and independently beats TeacherDirect, both
Directs, Fixed and the existing HiResAuxMix strong static reference -> freeze for parent-disjoint confirmation; no parameter
selection on confirmation. If only teacher/direct gains, retain component but
no independent OPD claim. If no useful final transfer, close this single
configuration without extra temperature or fitting-strength rescue. Neither
one negative setting nor failure of Direct uniquely refutes temporal OPD.
CURRENT/MAIN/PAPER and spatial config unchanged; goal remains active until
reproducible gain and its attribution are established.
