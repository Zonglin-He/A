# Single-point repair: teacher-relative mean update, original32

Old result:60c617cc used empirical teacher-output mean. Identity targets then moved the policy by the sample-noise mean. HC dynamic15/16 had no admitted feedback, contributing -5.223792pp. The old measurements, negative results, implementation and interpretation remain preserved.

Only the finite-sample mean update changes:

mu_new = mu + mean_k(teacher_action_k - sampled_action_k).

At a fixed current mean, the population update agrees with the old one under zero-mean Gaussian sampling; finite-sample estimators differ. This is NOT the exact minimizer of the old uncentered empirical KL. The logged loss is a centered-correction diagnostic, not an acceptance test. No optimizer, LR search, new temperature, gate or teacher change.

CPU properties before any model run: exact identity at K1/8/24/32;100 identity rounds; partial-feedback zeros with denominator all K; independence from sample mean; old/new equality when empirical noise mean is zero. No GT. All-identity feedback must leave the mean bitwise unchanged.

Original locked HC16/Vid16, identical row order, aliases, Source/E/frame positions/MBR, seed/24 epsilon draws, source scale, .35 target threshold, nearest-native-proposal content and isolated query attention. Frozen network hashes and original legal-reference/no-E-support policies unchanged. E, Source and one-correction outputs and one-correction trace exactly reuse old sealed artifacts. They are unaffected baselines, not new teacher calls.

Fixed24 and dynamic8x3 are actually rerun. Re-encode original4 image/text once/query; reuse those encoded features across rounds. Fixed24 and dynamic round1 actions and conditional targets must equal v1 exactly. From dynamic round2 onward, newly reached states obtain fresh DINO decoder feedback; no old-path output splicing. Actual qualifications use same first query/domain and tolerances. A fresh raw-teacher response is saved per visited state.

All32 predictions sealed before new GT read. Old target GT exposure is acknowledged; no claim of untouched confirmation. Exact same official metrics, actual unobserved-frame sIoU excluding original4, paired query bootstrap/negative tails. Report feedback per candidate/anchor/query/round, invalid/no-support states in denominator, all identity round no-op checks and output displacements. Include old fixed/dynamic results as historical matched comparators, not current deployment candidates.

Budget:900s GPU lease/2400s created-to-seal/score wall upper bounds,16/20GiB memory high/max,Swap1GiB,host6GiB/GPU4GiB/disk30GiB. First original two examples timed; no model/prediction retry or deadline reset. No QR/LN/Source fit, new DINO candidate rule, new backbone, critic, Qwen/SAM, target tuning or dense independent visual-evidence claim.

Decision: dynamic beating E and fixed with both-domain complete v and actual unobserved s positive provides evidence to continue, not automatic expansion. Merely returning near E means drift fixed without added utility. Feedback remains extremely sparse or admitted feedback remains harmful => stop current teacher interface, no data expansion or parameter rescue. Full signed and uncertain results retained. All files and negative findings published through established privacy-preserving export, then archive check/snapshot/check.
