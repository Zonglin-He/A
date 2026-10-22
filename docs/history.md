# Method history

2026-10-22 — User selected the original BoxFixed + Global6 + original physical MBR
as the main method. 320 was rejected; 448 remains an accuracy/cost reference, with
no offset1 upgrade. Resume preserves the completed HC1902 and saved Vid6088 outputs
under the same frozen full plan. This cleanup does not change that plan or method.

Bounded HC64/Vid64 tuning and fixed32-parent development rechecks did not justify
changing tau1/observed.5/guard.1/link.25. Final and Untuned are aliases. The original
Source checkpoint, ERef strength/calibration and single ReferDINO/eight observations
remain unchanged. Existing full results are not used to select parameters.

2026-10-22 — User requested local and GitHub cleanup while full evaluation runs.
An isolated maintained tree was prepared. It exposes only the selected method;
Global6 moved from research scripts into the package. Legacy OPD, reward studies,
conditional time replay, alternate backbones and resolution trials are archived,
not deleted from history. Required original TA-STVG and ReferDINO sources remain.
CPU tests, saved-capture numerical compatibility and original-code checks precede
publication. No GPU/model/GT scoring is performed for cleanup. Active runtime files
and saved predictions are preserved; its source identity is not migrated or relaxed.

Historical details, negative findings and exact versions:
[pre-cleanup research archive](https://github.com/Zonglin-He/A/tree/archive/research-pre-mainline-20261022). Current verification:
[receipt](../results/verification.json). Publication is recorded separately after
remote tree and file bytes are verified.

Cleanup implementation published at `1448faaa4d88d1abb4c460fd68a66d80410c161a`
to main and research/stvg-opd-paper-hc2-revision-v2. Remote tree and all 250 files
(1,585,327 bytes) verified. Archive branch retains the original 852-file tree and
commit ancestry. Publication receipt: [docs/publication.json](publication.json).
The active full queue retained all 434 original source hashes and continued advancing.

2026-10-22 — Complete official full evaluation finished: HC2 1902/227 and VidSTG10303/732
sealed before separate CPU scoring. Both jobs scored; controller/worker/teacher/scorer
exited and shared GPU lock released. Full vIoU Source→Final is24.9948→26.7597
(HC,+1.7649pp, parent-cluster CI[1.2804,2.2751]) and11.7777→14.2685
(Vid,+2.4908pp,CI[2.1351,2.8504]). Unobserved sIoU gains1.7881/2.3067pp;
tIoU changes−.0666/+.9617pp. All severe declines266/1149 and every signed row
retained. This is ERef+BoxFixed+Global6, not OPD or spatial-to-temporal feedback.
No full-result selection or parameter changes. Independent publication audit verified
12205 prediction/receipt hashes, all raw Final/Untuned aliases, signed means/thresholds/
counts and 10000-resample parent-cluster intervals with zero aggregation error.
Original434 sources and all seals unchanged;24,410 official/dense scorer checks pass.
Actual costs split original/resumed worker counters; original imported capture cost is
excluded. Corrected generic scorer wording: Final aliases Untuned, so only one posterior/DP
is computed. [Complete report](../results/boxfixed_full_v1/REPORT.md).

Full results published at `5d8b2c09e0d2e0d62f3a01dec03de53684bcf3e6` to main and
research/stvg-opd-paper-hc2-revision-v2; remote tree and all274 files/6031045bytes
verified, retaining clean single-method code and archived ancestry. Exact238 public
frozen sources match c310af58;196 private installed runtime dependency sources remain
hash-bound, not exported. Publication adds no GPU/model/GT scoring.

2026-10-22 — User explicitly authorized mechanism-led HC2 temporal tuning followed
by a new full HC evaluation. Old published v1 remains immutable. New protocol in
outputs/temporal_hc_v2 locks a SHA-ranked official96-parent panel (64tune/32validation),
18 temporal profiles maximum: endpoint temperature, layer slope and duration tilt only.
Spatial weights/teacher/observations/ERef/precision/old hull/physical MBR fixed.
Old HC64 had only31 exact query IDs in HC1902 (62 parent IDs overlap); old temperature
1.5 improved old development but harmed prior32 recheck, so no blind temperature repeat.
New96 captures use unmodified old provider with exact old expert runtime fingerprint
and explicit evidence reuse; Source/ERef/native intervals bitwise checked against v1
before panel labels. No old resume hash exemption. Fresh temporal outputs have new
plan/code/receipt bindings. All official full outcomes were already exposed, so this
is disclosed target development and post-selection full rerun, not independent test.
Root supervises sequential tuning; one locked validation, no validation retuning.

2026-10-22 — HC temporal v2 bounded search completed:96 official captures sealed
(6 complete reuse/90 native Source captures/0 new expert forwards),64 tuning parents,
14 profiles, then one fixed32-parent validation. Selected tau1.5 improved tuning
t+.8584/v+.2973pp but validation failed t−1.5605/v−1.0859pp versus Global6.
Fixed-space control localizes the loss to time; short/already-good events suffer
interval overexpansion. No validation retuning, fallback selection, gate relaxation
or new full launch. Default and old complete published full remain unchanged.
32 CPU engineering checks passed. All signed positive/negative evidence retained.
[Temporal tuning report](../results/temporal_hc_v2/REPORT.md).

2026-10-22 — User explicitly requested continuing temporal tuning after v2 failed.
New v3 protocol excludes all96 v2 parents; new64 tune/32 validation are fixed
before labels. Additional distinct knobs: separate start/end temperatures,
offset hull strength versus mixture, last-layer ensemble depth, MBR IoU utility
curvature. Source forward/eight expert observations/spatial weights fixed; all
old evidence retained. New output-only readout changes are explicit and not
claimed to preserve old hull/MBR. Root supervises; no reuse of failed validation
to pick fallback, no new full until a new fixed validation passes.

2026-10-22 — V3 broader temporal search completed16 profiles. Selected start
temperature1.5/end1, original six layers/hull/MBR, frozen before new32 validation.
Tune t+.5326/v−.0271pp; validation t+.3683/v+.4105pp versus previous Global6.
All relative-previous protections pass but Source t gate fails(−.4775pp); original
protocol/failed eligibility retained, no fallback/retuning. Under the user’s explicit
request to rerun full, a separate pre-full exploratory extension locks only this
profile for HC1902 descriptive assessment. It is not labeled validation-passed or
promoted to main; full results will not select parameters. Old v1/Vid remain unchanged.

2026-10-22 — New exploratory HC1902 full started in mainline private outputs/
hc_temporal_full_v3_exploratory with fingerprintf33e75a4652dfc369277b92e90cb63e3014b0c046f13a9ed02e9975828cb208d.
Original plan and all core source hashes frozen before launch; original Source/E/
Global6 interval/spatial outputs must match per query. New source capture costs
exclude reused original expert generation. Existing heartbeat reused for this
queue and existing Luna reused for read-only monitoring; no new agent/automation.
[All16 parameter profiles, preserved failed Source gate and rationale](../results/temporal_hc_v3/REPORT.md).

V3 full positive/negative development evidence and exact exploratory protocol/source
published at c261d9e5f2ebeb16a3a6070eb37bc6b41d9b7945 to main/research.
Remote334 files/13,582,128bytes and exact tree verified; archive ancestry preserved.
Full job continues with its original frozen sources; no default upgrade or full
accuracy claim. Source gate remains failed and validation intervals cross zero.

2026-10-22 — User explicitly rejected launching full without exceeding Source.
Root acknowledges the exploratory launch was an execution error. Paused at338/1902;
all338 prediction/receipt/capture hashes verified, no orphan files, own controller
and worker exited, heartbeat PAUSED. Original code/plan/results preserved.
No paused-full scoring or automatic resume. New temporal v4 is Source-primary:128
prior tuning parents only; prior validation parents excluded, last35 parents fixed
for one validation. Spatial code/weights and original forward/eight observations
fixed. No full launch unless Source-primary validation passes, no exploratory bypass.

2026-10-22 — Source-primary v4 completed24 cached profiles. Frozen entropy-
conditioned start calibration+IoU3 had tune Source t+.9591/v+1.3243pp, but
remaining35 recheck Source t−.2713pp and Global6 v−.9331pp. Failed gates preserved,
no full resume/launch,338 old outputs remain paused; no spatial modifications.
The remaining35 has now been exposed and must not be relabeled untouched.

2026-10-22 — User explicitly added both short and long event increments must
be positive. Before v5 validation, lock Source as reference, existing<=5s/>5s
GT-duration groups for offline evaluation only, strict positive group gains on
tuning and fixed recheck, no full if either fails. Rank by worst-group gain among
admitted profiles; no GT-conditioned inference or post-hoc duration threshold.

2026-10-22 — Temporal v5 finished12 cached profiles with no new GPU/expert calls.
Internal boundary-peak penalties systematically traded short gains for long losses;
endpoint-radius0/2 and measured MBR3 combinations did not fix that behavior.
Before recheck, transparently amended pairing-only eligibility to freeze the sole
Source-positive-two-group tuning control MBR3, matching the user's latest objective.
Same35 exposed development recheck: short Source t+.8493pp but long−.6587pp,
overall−.1847pp. Strict group criterion and original protections fail. No fallback,
full resume or default/spatial change. All signed results and parent CIs preserved.
44 CPU tests pass; paused40 core files/provider434 remain exact. The338 paused
predictions remain valid saved evidence, not a basis for selection or full claims.
[Source-primary v4 report](../results/temporal_hc_v4/REPORT.md) and
[strict short/long v5 report](../results/temporal_hc_v5/REPORT.md).

Temporal v4/v5 code, strict group criterion, pause correction and all signed outcomes
published at f51883f8683553c47c015a4f29bf5567bad09816 on main/research; exact remote395
files/30,698,716bytes/tree and historical archive ancestry verified. Source qualification
remains failed and full remains paused. This receipt introduces no experiment or selection.

2026-10-22 — User authorized case-based temporal continuation, spatial fixed and
full paused until both Source duration groups positive. In v5 recheck8 harmed long
events,7 had final-layer t at least as good as all shallower individual layers;
P157 is a counterexample. New v6 tests final-anchor refinement continuation with
no fitted scalar: same-sign log-probability deltas bounded by the preceding/final
step minimum. Global6/Source references, unbounded contrast, bounded continuation,
reverse control; only two forward candidates eligible. Five fixed modes, no sweep,
128 tune/same35 exposed development recheck, no full on failure.

2026-10-22 — V6 completed5 fixed modes, no candidate passed strict Source groups;
no35 recheck/full. Bounded layer continuation gained long+.0978pp but lost short
-.3445pp versus Source. No continuation-strength sweep. Further case audit finds
Global6 boundary outside final3 layer-readout range in19/40 harmed long tune queries
and11/19 helped short tune queries. V7 therefore tests one constrained original-MBR
readout, with early-layer constraint and simple last3-average controls. Original
Global6 posterior unchanged; no layer-count/width/weight scan. Full remains paused.

2026-10-22 — V7 primary late-bracket has tune Source short+.1393/long+.2365pp,
but fails original extra tune+.5pp and Global6 preservation gates. A separate
Source-objective mechanism extension was frozen before35 predictions; failed
original eligibility retained, no control fallback. Same35 exposed recheck gives
Source short+1.3590/long−.7984pp, overall−.1203pp; Source qualification fails.
P131 partially repairs+6.107pp versusG6, but P132 loses7.805pp from a correctG6
interval; late consensus cannot certify correctness.50 CPU checks pass;1059
new cached prediction records sealed/scored;0newGPU/experts;40 core and434 provider
files unchanged. Full remains paused338, default/spatial unchanged.
[Case-based continuation](../results/temporal_hc_v6/REPORT.md) and
[deep-supported MBR recheck](../results/temporal_hc_v7/REPORT.md).

Case-based v6/v7 implementation, full signed outcomes and all repaired/harmed cases
published at e64803ec5dc0664dd916b1aecae8144371c0ed58 on main/research. Remote422
files/34,658,271bytes/tree fully verified with archive ancestry preserved. Source
long-event qualification remains failed; no full resume or default promotion.

2026-10-22 — Latest user explicitly authorizes relatively best frozen full despite
failed Source group gate, then TA-STVG-only paper task. Choose originalGlobal6
temperature1/sixlayer mean/original hull+MBR, unchanged BoxFixed. Same35 totalt/v
beats the newer MBR3/late-bracket candidates; no claim Source temporal gate passed.
Reuse already-complete Source/Full after bitwise verification; new dual-domain
collection supplies interpolation/OursS/two space ablations/stage and extent analysis
from one capture/query. Third-party generic TTA not verified; historical projectOPD
HC1902 Source/ERef/positions/native time all exactly match, a separately labeled
reference; incompleteVid8405 is rejected, never resumed. Initial zero-execution
preflight expected192 official cache hits but159 inputs match current official lists;
all159 aliases passed. Archived preflight plan/sources, fix regression to verify all192
math captures plus159 formal aliases, without promoting33 unmatched cache entries.
No model/GPU/expert call occurred in the failed preflight.

2026-10-22 — New TA-only paper queue launched at10:42:49UTC with own frozen
45dd2a86 fingerprint/432e69b5 planSHA. Controller3673523/start223111135, HCworker
3673524/start223111140; actual process identity, lock and matching receipts healthy.
55 CPU checks,192 cached mathematical regressions and159 exact official aliases
passed. Main/ablation/stage/extent share one capture per query. Full package remains
in progress; per-domain seals precede CPU scoring, then matched16-parent timing
and private qualitative review. Old full already complete; v3 paused338 never resumed.
[Locked TA paper package](../results/tastvg_paper_v1/RUNNING.md).

Frozen TA paper implementation/protocol published at5b943d5524b0b69c12830e3575f73ceb8ca3b024
on main/research;436files/34,844,204bytes/whole tree and archive ancestry verified.
Queue remains running, no new paper result completion claimed.

2026-10-22 — TA paper HC1902/227 fully sealed/scored; score55.41s,15216
official/dense checks max1.33e-15; Source/Full reproduce old published outputs.
OursS v26.8147 vsSource24.9948,Full26.7597; Full−NativeTemporal−.0550pp,CI
crosses0. Both local propagation (+.4680pp Full−NoLocal) and path selection
(+.2779pp Full−Independent) core CIs positive; all signed/severe retained.
No result-based parameter changes.48runtime+434provider files unchanged.
Vid worker4124023/start223882623 is the planned next segment; HC scorer receipt
complete, controller retains child record until planned wait. Full package not done;
cost/qualitative/publication await both domains.
[HC stage record](../results/tastvg_paper_v1/hc2/STATUS.md).

2026-10-22 — Read-only historical temporal adaptation audit:17 Git source/evidence
objects and4 preserved reports inspected;1118 saved-scalar identities checked
(max2.22e-16), no new model/head forward, fit, expert, RGB or GT scoring. Separate
no-update decoder damage, feedback calibration, failure to absorb useful direct
feedback, old positive temporal-head/Temporal-Lite results and development-scale
non-replication. DSE/T3AL interface prerequisites are not a local reproduction claim.
Current compact full caches lack head-input hidden; old HC64 replay caches preserve
six-layer hidden. Four sample schemas are receipt-bound (Vid not yet domain-sealed).
No old actionness teacher, external time expert or unaligned cosine loss is promoted.
48 runtime/434 provider hashes and active plan/process identity unchanged; Vid
progressed during the audit. HC stage is complete, full paper package still running.
[Historical audit](temporal_adaptation_audit.md) and
[saved scalar checks](../results/temporal_adaptation_audit_v1/SAVED_SCALAR_AUDIT.json).

Historical temporal audit plus completed HC-stage signed results published at
2a58edea00e6eb2cde37b237863da682faf5f6b9 on main/research;446 public files,
36,826,854 bytes and complete remote tree verified with archive ancestry retained.
Audit performed0 new model/head forwards, fits, expert calls or GT scoring;
active48+434 runtime files unchanged. Dual-domain paper package remains in progress.

2026-10-22 — User prompted recheck corrects incomplete historical audit: ReferDINO
pre-max event-token feedback(HC128), shared text256 adaptation(HC16/Vid16), actual
second-pass object itq reconstruction and target×event query(HC128) were already
implemented and measured.11 source/report blobs inspected; no new inference/GT/fit.
EventE−MBR completev−6.1649pp; shared-text32 intervals all unchanged; object-query
HC−.3452/Vid+.0303pp; event-query ROcc0Mix+.0273pp CI crosses0. Query-onlyOPD64
was spatial, not temporal. Withdraw token extraction/real query connection as new
recommendations; exact event-supervised optimized temporal delta remains unverified,
not established absent across all history. Main full/spatial/code unchanged.
[Audit correction](temporal_adaptation_audit.md#9-补充纠正事件token与时间query并非尚未尝试).

2026-10-22 — Latest user pauses paper full for bounded event-feedback/query experiment.
Verified owned controller/worker identities; stopped controller scheduling then interrupted
worker at next committed query boundary. HC1902 remains sealed/scored; Vid1373 complete
receipt/prediction/capture hashes preserved,0 unbound outputs. Both owned PIDs exited;
GPU lease available. Original plan/48 maintained+434 provider hashes unchanged.
State explicitly paused_by_user; no automatic resume. Proof in private maintenance.
Old event-token/query reports remain baselines; new experiment addresses interval
misranking and actual learned time-query response, not repeated token discovery.

2026-10-22 — Bounded HC64 temporal-query experiment locked after user pause. Old
HC128 saved event readout harms82/improves45 in tIoU;63 harmed still have anchor
AUC>.5. Compare NativeG6, archived LegacyContrast, prior-anchored RawDirect/RawQuery,
and predictive leave-one-anchor-out Direct/Query. One fixed256 shared temporal delta,
10 Adam steps/.01/radius.1; no GT state choice or sweep, same spatial geometry/DP.
Old semantic tensor directory missing; permit same ReferDINO original8-only semantic
capture per query, no new Source encoder. All64 seal before CPU labels.
First pilot semantic check recomputed sigmoid on CPU versus native CUDA output;
0 semantic outputs/1 expert query attempted. Preserve plan/code/log at private
maintenance/event_query_premax_check_v0_20261022, compare native CUDA sigmoid and
exact raw logits instead, without changing inference or relaxing numeric tolerance.
Restart has a newly locked fingerprint; no GT read or scientific negative implied.

2026-10-22 — Event/query HC64 experiment completed and scored, no new method gain.
64 same8 ReferDINO semantic queries restore native boxes bitwise;0 new Source encoders.
Two256-delta fits/query,1280 actual Adam steps. RawQuery−G6 v−.0870/t−.2186pp;
PredictiveQuery−G6−.1924/−.4172pp, all primary CIs span0.63/64 losses decrease/arm,
17 temporal harms each. Radius never clips; only24.1%/13.1% KL absorbed, no universal
query-capacity failure claim. Same evidence-pattern boundary ambiguity remains.
All64 seal before labels;896 official/Dense checks max7.77e-16; four opposite-case
final query reinsertion checks exact. Frozen48+434 sources unchanged. First semantic
CPU/GPU sigmoid check0outputs/1 expert extra preserved;64 successful (65 actual).
Final scorer np.int64 serialization recovered by separate CPU adapter, no inference
or fingerprint change. Full stays pausedHC1902/Vid1373, no automatic resume.
[Experimental result](../results/event_query_hc64_v1/ANALYSIS.md).

Event/query experiment code, fixed protocol, complete signed outcomes and negative
analysis published at50ed82b9c6049d82d3f6140f13b166d945559cec to main/research.
466files/37,503,162bytes/tree fully verified; history ancestor preserved. Heartbeat
paused after experiment completion; paper full remains pausedHC1902/Vid1373.

2026-10-22 — User authorizes a new lightweight HC64 boundary-feedback test, not more
query steps. Implemented isolated dense text-conditioned hidden change contrast,
computed separately on native offsets with2-frame neighborhoods and six layers;
primary BoundaryCell preserves original posterior mass per eight-anchor-membership
class. Compare unrestricted BoundaryGlobal, same-value temporal ShiftCell, old
raw-event RawCell with identical preservation, bitwise NativeG6 and frozen Source.
All64 five-arm predictions sealed before new CPU labels;0 model/expert/decoder
forwards,0 optimizer steps. BoundaryCell−G6 completev+.0779/t+.0817pp, both CIs span0;
short t+1.1898pp/long−.2008pp. RawCell overallt+.3114pp is higher, still hurts short
events. New feedback differentiates2282/2283 nontrivial old evidence classes and
posterior-expectedt improves+.1956pp, but does not establish reliable final gains.
RC025/029 recover;RC042/016 move toward stronger changes but wrong inner boundaries.
17 temporal harms,2>5pp;0 complete-v losses>5pp. No method/default/full promotion.
768 official/dense checks max6.66e-16 and9 independent denseMBR checks max1.55e-15;
48 maintained+434 provider hashes exact, full remains pausedHC1902/Vid1373.
Scoring first final JSON write failed on NumPy integer; original code/log preserved,
separate CPU serializer adapter completed same metrics with sealed outputs unchanged.
[Complete boundary-feedback result](../results/boundary_feedback_hc64_v1/ANALYSIS.md).

Boundary-feedback implementation and complete signed development results published
at9d9ceb1a396c275fd8461800456105c4409db77c on main/research;480 files/37,954,286
bytes/full tree verified. Archive ancestry retained. No candidate promoted and no
full resume. Private publication/engineering/score receipts retained.

2026-10-22 — User requests real HC64 joint event-boundary feedback, improve more
queries with fewer harms; space fixed and full still paused. Three bounded,
case-driven studies complete on the same64 exposed parents. V1 builds target-identity-
relative predicate token margins from saved ReferDINO8, disposable ridge1 projections
to saved six-layer hidden, whole-interval soft semantic IoU and18-view/LOO guard.
No new visual/decoder/expert forward, backbone update or query optimizer step.
V1 guarded v+.1904/t+.4586pp but all22 accepted intervals expand and score/log-length
correlation.9353. V2 subtracts a time-uniform mask with identical semantic mass;
constant masks now reward no interval. Same projections, no weight scan/refit.
V2 guarded v+.0996/t+.0129pp,4v improvements/2 harms/58 ties;7 changed intervals,
0 severe v/t harms. Low coverage and near-zero temporal gain, not a new main method.
V3 applies exactly this feedback to at most9 one-node paired-boundary neighbors;
17v gains/16 harms, t−.2927pp and6 temporal harms>5pp. More corrections do not
establish reliable selection; shifted-semantic controls not clearly beaten.
All192 records sealed before their corresponding CPU scores,64 unique parents,
2304 official/dense checks max6.66e-16,18 independent box-path replays exact.
Post-seal role/margin audit64 exact and12 text-resource hashes match the frozen
paper assets. Lower-is-better MAE count orientation corrected in summaries only;
original reports/receipts preserved, all scalar scores/means/CIs/predictions unchanged.
48 maintained/434 provider hashes unchanged; HC1902 scored/Vid1373 saved/full paused.
No profile promotion, fallback selection, GT-duration mixture or automatic resume.
[All three studies and complete signed outcomes](../results/joint_event_hc64_v3/ANALYSIS.md).

Joint-event stages1/2/3 implementation, fixed protocols, all signed positives and
negatives published at51073d070f7cac0dc15613de1ac2087a2b4be885 on main/research;
513 files/39,685,424 bytes/complete remote tree verified with archive ancestry.
Full remains paused and no research candidate is promoted.

2026-10-22 — User requests teacher event-vs-object information audit and realHC64
feedback correction, without reducing harms merely by abstention. The frozen teacher
classification is token-max referred-object focal supervision, not event-boundary
supervision. Predicate/noun correlation median.866, noun in-sample R2mean.689;
predicate minus noun event-anchor AUC+.0206 CI[-.0450,.0881]. Conditional high-noun
predicate AUC.5278 CIcross.5; presence is a response proxy, not visibility GT.
Two isolated fixed tests complete: V1 other-slot nuisance OLS has49/64 extreme
calibration extrapolations and t−.1396pp; V2 same-target leave-one-time nuisance OLS
t+.0964pp CI[-.4647,.6708], v+.1444pp CIcross0, time13win/16harm, shortt−.8801pp,
long+.3454pp. Neither establishes reliable event feedback or low-harm broad improvement.
All arms use same unguarded full-posterior reweight+physical MBR; no membership
protection/CV-veto/new optimizer steps. Matched raw-verb/ownership/noun/shift controls
retained; best control not promoted posthoc. 128decisions sealed before new scores,
1792 official/dense checks max6.66e-16,320 control pairs bitwise,18 spatial and18 denseMBR
replays pass. Frozen48+434 unchanged, no model/expert/encoder/decoder forward or RGB read.
Premature V1 scorer invocation rejected absent seal before labels; log retained,
same code rerun after seal without prediction/fingerprint changes. Paper full stays
pausedHC1902/Vid1373; no default/full promotion. [Evidence and signed results](../results/event_presence_hc64_v2/ANALYSIS.md).

Teacher event-information audit and both fixed conditional-feedback HC64 tests
published at76e6f5de4fa277bbbd760f5dedc150820e8bd1b6 on main/research;538 files/
41,031,483 bytes/complete tree verified, historical ancestry retained. No candidate
promoted; paper full remains pausedHC1902/Vid1373.

2026-10-23 — User authorizes active, case-driven native temporal/event-information
research until07:00 Asia/Shanghai today, not only ReferDINO response normalization.
No additional expert model/observations; space fixed. Real text interventions are
one branch, not a mandated exclusive route; old25daec full/static negative must be
audited before reuse. At deadline, if no direction validates, user explicitly permits
verified same-plan resume of paused paper queueHC1902/Vid1373; if a direction validates,
finish finite tuning, freeze distinct compatible plan and run full. Never mix new
method outputs/fingerprints into old receipts, hide failures or force-stop unrelated jobs.
Private continuation ledger:outputs/overnight_event_loop_20261023/state.json.

Overnight branch1 fixed: native_event_text_hc64_v1 fp1dfe044804731753fdf7b0ba52d048ad260463000ea506b652c0cc5eb77241b9.
Old25daec did real static-caption forwards and lost9.3220pp tIoU, so not represented
as a first-ever intervention. New branch masks only main-clause predicate BPE IDs
before actual RoBERTa encoding; protects queried nominal subtree, matches control
mask BPE count and original token/attention positions; native two-pass/two-offset
sixlayer recomputation with exact text-independent visual memoization. Primary
q0*qControl/qEvent, old q0²/qId/sharpen/individual masks/reversed effect controls.
First actualHC query restores original hidden/posterior/Source boxes bitwise;
4 visual misses/12hits across4 text forwards,0 expert calls. Full remains paused.

Overnight branch1 actualHC64 complete: full native hidden/posterior/boxes restore
bitwise64; all64 seal before scoring. Selective text feedback−G6 completev−5.3244pp,
t−8.7718pp,17timewins/42harms/5ties and32time losses>5pp. Same-length/near-prior
721,976 pair macro accuracy46.74% vs native52.14%; no useful event-ranking evidence.
Protected text does not ensure identity invariance: event-mask Source geometryIoU
mean.7794,7/64 below.5; identity-only11/64 below.5. All failures retained; stop this
ratio branch rather than tune weights. Scorer NestedTensor import failure preserved;
independent CPU path-only adapter runs original pinned scorer,1024 checks max9.99e-16.
No prediction/source/hash change or inference rerun. Native256queries/512textforwards,
visual256miss/768hit,0experts. [Full signed result](../results/native_event_text_hc64_v1/ANALYSIS.md).

Overnight branch2 launched: native_event_motion_hc64_v1
fp5b7e969d8e21e44000d426ceebc36ad75611e8967ac323606a648e0c6b82e742.
Full text fixed, compare native pre-fusion motion temporal-mean replacement against
appearance replacement, individual contrasts/directNull/sharpen/reverse controls.
OriginalFP16 fusion+FP32 two-pass native replay must recover hidden/posterior bitwise.
First7 queries pass no-op;0newRGB/visual/text/expert forward, cached fusion only.
PaperHC1902/Vid1373 stays paused; no early full resume or method promotion.

Overnight branch2 actualHC64 complete:64 native pre-fusion hidden/posterior recover
bitwise; all64 sealed before1152 official/dense checks(max9.99e-16). SelectiveMotion
v−2.4442pp/t−5.2222pp,29timewins/34harms,30time losses>5pp; no Source/G6 gain.
Both directNull arms also lose~3.9pp tIoU. Matched-pair score accuracy52.81%CIcross.5
vs native52.14%; low-native-support quartile posterior mass.000912→.042626 andTV.5786.
No calibrated event-absence likelihood established; stop this ratio, not weight scan.
Actual384fusion offsets/192native replays in46.85s,0newRGB/visual/text/expert forwards.
[Complete signed result](../results/native_event_motion_hc64_v1/ANALYSIS.md).

Overnight text/motion branches and complete signed positives/negatives published
atcdf99674c63dc9ce544998349c728d2e98c62d7f on main/research;565files/42,728,638bytes
and entire tree verified, archive ancestry retained. Neither promoted. Next separate
branch card tests actual overlapping native temporal contexts with explicit boundary
censoring and structural-hull/sharpen/reverse controls; not launched at this receipt.
Original paper remains pausedHC1902/Vid1373 until07:00 user-authorized decision.

Overnight branch3 real overlapping contexts completed, fp4b9f91a28171b76554b5835684944407c545e30a0d3af316e22f2725b0debe92.
Full text/raw projected feature values preserved; prefix0..48/suffix16..64 use
original two-offset native initialization/decoder and crop-censored endpoints.
All64 full hidden/posterior bitwise, structural-null no-op64;1152 official/dense
checks max8.88e-16. Primaryv−3.4506/t−6.8349pp,14timewins44harms;shortt−2.4362,
long−7.9562. Matched candidate accuracy53.96%CI[51.74,56.14] does not translate
to final gains. Original mean length8.648s falls7.293s; same-information q0²/uFull
control6.680s,t−10.4859pp. Concrete RC059 native t82.19→13.29%,end moves−320frames;
RC05682.89→36.84%,both endpoints move inward. New context is not automatically
useful/calibrated feedback. Next isolate context-induced innovation from original
logit confidence with same-window full-logit control; no context/weight scan.
Actual384fusion offsets/192replays46.09s,0RGB/visual/text/expert calls.
[All signed outputs](../results/native_event_context_hc64_v1/ANALYSIS.md).

Overnight branch4 ContextInnovation cache-only test completed,
fp9d2ac4cc4279135b638f336bf89e449a9ab0a7d4d360ee0c3bf39431bdac6e3a.
Compare real context qW with exactly same cropped-grid posterior from unchanged
full-context sixlayer endpoint logits bW, not an uninformative null. This separates
new context response from repeatedly using existing confidence. Primaryv−.2470pp/
t−.1596pp vsG6,CIcross0;26timewins25harms13ties,9time losses>5pp. Source comparison
v+2.5544/t+.9787pp does not erase Global6 and short-event failures:shortt−1.1780pp,
long+.1000pp. SameInformationContext t−9.2426pp; removing repeated confidence
explains recovery but no positive net mechanism established. Matched candidate
accuracy53.40%,increment vsNative+1.27ppCI[−1.72,+4.15]pp.1024official/dense checks
max8.88e-16; all64 no-innovation/parent arms exact,CPU4.90s,0model/expert/optimizer
forwards. No promotion. [Signed result](../results/native_event_innovation_hc64_v1/ANALYSIS.md).
Next isolate context content from cropped temporal-position rebasing; no stronger
guard, exponent scan or early paper resume.

Overnight context/innovation implementations, protocols and every signed outcome
published atdd5d4780ca0c912a379bd0fd926e5cde12db7906 on main/research.592files/
44,407,707bytes and entire tree verified; historical archive ancestry retained.
Neither primary passes predeclared qualification; no method promotion. Next scoped
mechanism card keeps original full-grid temporal positions in real context replay.
Original paperHC1902/Vid1373 remains paused until07:00 authorized decision.

Overnight branch5 global-grid position preservation completed,
fpf405056accc58b028610910f24a353be56ab1876677734346a289b6ff48e643c.
Scoped frozen time_embed hook supplies original temporal slots during prefix/suffix
replay;64Full hidden/posterior and64Prefix posterior restore bitwise,6CPU checks.
Primaryv+.1781pp/t−.4506pp vsG6,22timewins22harms20ties,10losses>5pp;
shortt−.7680,long−.3696pp. Versus rebased innovationt−.2909ppCIcross0.
Matched candidate accuracy49.09%CIcross.5. Position-rebasing hypothesis not supported
as the main cause or a useful correction; stop, no offset/weight sweep.1024official/
dense checks max8.88e-16,384fusion offsets/192replays45.29s,0RGB/encoder/expert calls.
[All signed outputs](../results/native_event_position_hc64_v1/ANALYSIS.md).
Failure-caption audit finds compound action descriptions (approach+sit;raise+pour+look),
so next separate test asks whether actual native predictions of original positive
subevent clauses can jointly recover full-event boundaries. No claim yet of cause.

Overnight branch6 positive native subevent composition completed,
fp664ad2efebb61a9478b921322de333752df476ec1e82020413e6bba330a0de98.
Existing query-only Source Stanza selects52/64 same-subject main-AND descriptions,
reads first/last actual clauses with same frozen native model, combines interval
envelope then fixed half Full fusion. No LLM/VLM/expert added;12unsupported exactNative.
All64 reencoded Full text/hidden/posterior/Source boxes bitwise. Primaryv+1.6502pp
CI[.1936,3.1828],t+2.4693ppCI[−.1304,4.9675] vsG6,30timewins21harms13ties,11losses>5pp.
VersusSourcev+4.4516/t+3.6077pp. Self-hull alreadyt+1.5166pp; clause-minus-self+.9527pp
CIcross0. Shortt−5.7702pp,long+4.5696pp, so not adopted. Matched score accuracy57.64%,
supported59.40%; all signed controls retained.1152official/dense checks max6.66e-16.
Actual168text/native queries336fusion offsets49.03s,0RGB/visual/expert/optimizer calls.
[Evidence](../results/native_event_composition_hc64_v1/ANALYSIS.md).
Offline failure auditRC014/054/026 shows final-action query can select an EARLIER
occurrence, so unordered hull spans incompatible phases. Next fixed cache-only test
jointly enforces nondecreasing first/last starts AND ends (overlap allowed), keeping
identical evidence/half fusion and role-only/swapped/self-order controls. No guarantee
that AND implies chronology; no GT/order-probability acceptance gate or weight scan.

Overnight branch7 cached joint semantic order completed,
fp2c053719d4877aa6bd59ae4f0f4098516f258a8bb19db16b98f154c7a9a673b7.
Primaryv+1.3136/t+1.9392pp vsG6,28timewins23harms13ties;shortt−6.4844,long+4.0863.
Swapping first/last order is nearly identical(57.8953 vs57.9005tIoU), while same-info
self-order reaches57.5333. New chronological relation does not explain stable gains
or repair short harms. Keep positive means but no method qualification/promotion;
no order/weight scan.1280official/dense checks max6.66e-16,3.81sCPU,0model/expert calls.
[All signed arms](../results/native_event_order_hc64_v1/ANALYSIS.md).
Positive clause source-box proxy drifts from Full:mean firstIoU.8038,last.7479,
with8last clauses below.5 (proxy only, not identity truth). RC014 lastIoU.287 and
25.21pp time harm;RC056 lastIoU.066 but time improves, so drift is not a correctness
veto. Next test full-caption query reuse in actual clause decoder, not rejection:
keep original subject/whole-frame context in native time query while clause memory
changes. Existing old full-caption object-query/attention negatives remain controls
and preconditions; do not call the interface itself new.

Position-preserving native contexts, positive subevent composition and joint-order
follow-up published at7027d8f096c914cb3e679fd6fef5856f1aabfdd8 on main/research.
637files/47,077,326bytes and complete remote tree/files/hash verified; archive
ancestry preserved. Positive clause averages and all short/severe harms public;
no qualified new method or main/default change. Next shared Full native query
test isolates conditioning drift across positive subevents; original paper remains
pausedHC1902/Vid1373 until07:00 authorized decision.
