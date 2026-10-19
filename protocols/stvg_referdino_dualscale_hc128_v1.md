# Target-normalized dual-scale ReferDINO OPD, one structure-selection round

User scope: change only the output-box policy and its mathematically consistent
likelihood. Same frozen TA-STVG/checkpoint, E + finalMBR, native cached ReferDINO
selected eight-frame pseudo-trajectory, eight independent temporary256D content
states before temporal attention, decoder and physical frame/offset support.
No new expert model/head/training or observation positions. Same40 Adam .01,
betas .9/.999 eps1e-8 wd0,32 candidates,tau.05,seeds41/42/43. No seed winner.

Predeclare three (near,far) pairs: (.05,.25),(.05,.5),(.1,.5). In ACTUAL corrected
student coordinates m=(cx,cy,logw,logh), D=diag(w,h,1,1). Equal mixture density
half N(m,near²D²) + half N(m,far²D²). Each component has8 antithetic pairs,
16 candidates; reuse the previous seed/alias/round stream and pair arms/configs.
Only m has gradient. Candidate actions, D, teacher weights/boxes stop gradient.
Positive widths/heights via exp; crossing image borders is allowed and counted,
with no clipping/rejection that would silently require a truncated density.
Candidates use the SAME5L1+2(1−GIoU) geometric feedback and softmax over32.
Weights are constructed feedback, NOT native ReferDINO probabilities.

Loss is weighted negative log of the TRUE two-component mixture density,
computed by logsumexp including component normalization. Do NOT retain the old
single-Gaussian analytic KL or control-variate. This changes policy coordinates,
scales and the corresponding likelihood objective as one package; a difference
from old Live is not proof that exploration coverage alone caused it. Candidate
boxes are sampled outputs at8 observed frames, NOT separately decoder-generated
complete candidate trajectories. Each actual update reruns the full decoder.

Live proposes around current actual mean/current D, detached for that round.
Fresh Fixed proposes around INITIAL actual mean/INITIAL D each round, new noise
and re-scoring every time; its optimized current mean is not frozen. Its density
uses that initial D as well. Thus fixed versus live compares refreshing the whole
size-dependent proposal, not an extra moving-covariance Fixed variant. First
round actions/weights/update exact. No importance-corrected off-policy claim.

HC128 DEVELOPMENT is the exact exposed roster of the previous native experiment.
Reuse its E/Direct/old Live/old fresh Fixed only with pinned identical inputs,
teacher, coordinate mapping, optimizer and evaluation. New2304 fits/92160 state
updates;8 chunks of16 parents,900s per chunk,4800s total GPU including model/load/
guards. First2 actual experimental parents measure cost, not extra pilot runs.
Shared gpu.lock,20GiB memory,1GiB swap,CPU quota4,free GPU>=4GiB and existing
host/disk guards; no signals to other services. One finite controller, no schedule.

ALL development predictions/configurations sealed before new GT evaluation.
Metric means average three seed METRICS within each parent, then bootstrap128
parents (10000 resamples,seed20261018); no384-parent pseudoreplication or box
ensemble. Report every seed, all signed/unknown/zero-time/no-original-reference
parents, full vIoU/outside-eight known actual sIoU, negative tails, candidate
geometry/feedback/gradient/state changes and cost. Lower NLL is not task evidence.

Selection rule locked before results: rank old Live and the three new Live
policies by mean full vIoU across all128 parents/three seeds, with ties<=1e-9
resolved by outside8 sIoU then lower measured per-fit seconds then lexical name.
Report uncertainty, spatial/tail/cost tradeoffs even for selected winner; ranking
does not establish superiority or independent replication. Direct stays a control,
not a fourth new structure. Freeze one old/new policy, finish structure iteration.

After selection lock only, confirm selected policy on the32 RH parents previously
reserved before HC128; no GT-driven replacement/tuning or outcome-based branches.
They exclude these rounds' design/selection but have historical whole-project
Source/old-method exposure. Same8 positions. Necessary fresh Source capture and
native ReferDINO evidence acquisition for this previously uncached panel are
separate counted costs, not new experts/extra positions; no development teacher
calls. Confirmation compares E, new Direct, selected freshFixed/Live and oldLive
if a new policy was selected. All3 seeds retained. No later confirmation-based
parameter switch, rollback, interface change, or CURRENT/MAIN/PAPER promotion.

Engineering: CPU mixture logdensity vs torch distributions, analytic gradients
and finite differences, antithetic component/noise pairing, exact zero E and
frozen pretrained weights, mixture-noise/proposal/density/feedback/Adam/path
replay, old-input/target hashes, independent saved-metric/bootstrap checks.
Publish exact code/config/full results/negative findings using public allowlist,
verify dual-ref freshCAS and remote bytes/hash/tree; archive check→snapshot→check.
