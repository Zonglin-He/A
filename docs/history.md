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
