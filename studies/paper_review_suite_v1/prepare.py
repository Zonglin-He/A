"""Metadata-only two-level selection and exhaustive review work ledger."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import read,sha,atomic,RUN,PROVIDER,check
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
assert not(D/'plan.json').exists();check(True);selection={};jobplan=read(RUN/'plan.json')
for ds,j in jobplan['jobs'].items():
 groups={}
 for r in map(json.loads,(PROVIDER/j['manifest']).read_text().splitlines()):groups.setdefault(str(r['metadata']['source']),[]).append(r)
 h=lambda v:hashlib.sha256(f'20261024:review-v1:{ds}:{v}'.encode()).hexdigest()
 ordered=sorted(groups,key=h);assert len(ordered)>=192
 selection[ds]={stage:[min(groups[v],key=lambda r:h(r['id']))for v in pp]for stage,pp in [('screen',ordered[:64]),('confirm',ordered[64:192])]}
 assert not({str(r['metadata']['source'])for r in selection[ds]['screen']}&{str(r['metadata']['source'])for r in selection[ds]['confirm']})
atomic(D/'selection.json',selection)
items=[
 ('E1','Fig1','excluded_by_user','Existing paired entropy diagnostic frozen; no rerun or plot edits.'),
 ('E2a','Sparse expert interpolation','reuse_full','Exact eight native boxes; center and log-scale interpolation, E extrapolation fallback, native MBR time. Existing12205 sealed outputs.'),
 ('E2b','Dense expert native time','pending_gpu','Same singleton expert joint sampled frames and native interval; actual cost/coverage. OOM preserves failed run; do not silently substitute chunking.'),
 ('E2c','Temporal entropy baseline','reuse_screen','Existing fixed128parents/domain five-step temporal LN adapter; not canonical BN-Tent. Existing entire results already read.'),
 ('E2d','Consistency TTA','pending_gpu','Fixed5steps Adam.001 episodic baseline, same native readout. Define exact augmentation and inverse mapping before start; no labels.'),
 ('E2e','Task-aware TTA','pending_gpu','Fixed box pseudo-label or geometric consistency objective, Source no-op and reset; genuinely update native parameters. Do not label a newly designed baseline VLOD reproduction.'),
 ('E3a','Temporal decisions and prototypes','running_next','Full12205 cache-only temporal controls on fixed spatial support; seal before CPU labels.'),
 ('E3b','Source box vs fullframe vs proposed ROI','pending_shared_features','Acquire actual pretext grids; pool each support independently. Same posterior/readout/settings; screening64 then fixedconfirm128.'),
 ('E4','K budgets and costs','pending_gpu','K2/4/8/16/32/dense; genuine per-K joint ReferDINO evidence, actual GPU wall, K/T. Uniform/hybrid; no use K8 subsets as native smaller-K inference. General-K consensus requires audited scalable implementation, never enumerate2^32.'),
 ('E5','Coupled DP2x2','pending_shared_features','Greedy/DP fulltimeaux x Greedy/DP finalinterval. Repool alteredaux, recomputeq1/interval; final contrast fixes time. Screening then fixedconfirm.'),
 ('S1','Mask and propagation','pending_support','Current main is completebox; existing paired mask candidates report jointavailable AND missing/allslot. One-way affinity distinct from past-anchor-only; no mislabeled deletion.'),
 ('S2','Expert distance','reuse_full','Existing official GT-supported unobserved distance curves12205; no repeated GPU.'),
 ('S3','Schedules','pending_gpu','Uniform/Random/Source-only/Hybrid, sameK8, stable perqueryseed. Joint-context cache keys distinct.'),
 ('S4','Prototype variants','running_next','Soft/Hard, EventOnly/EventCentered, NoShrink, UniformFinalPrior; fullcache cheap.'),
 ('S5','Expert reliability','pending_mixed','GT anchor-IoU bins OFFLINE; fixed25percent corrupted anchors by input hash, fixed reflection corruption; recompute geometry/path/pooling, no quality gates.'),
 ('S6','Native coverage','running_next','Offline recall and complete-miss rates after all predictions seal. Raw native hull and native MBR distinguished.'),
 ('S7','Sensitivity','pending_mixed','Cache-only kappa16/64/256 beta.5/1/2 without selection; context1/1.5/2 and DP weights0/.25/.5 require coupledrepools. Actual gamma initialized.5HC/1Vid source-calibrated; eta/theta only if actual current components. Never resurrect deleted cycle threshold.'),
 ('S8','Controlled shifts','pending_gpu','Fixed blur, JPEGcompression, illumination screening then fixedconfirm; preregister exact transformations before inference; same frozen main no adaptation to results.'),
 ('S9','Third benchmark','pending_availability','Optional resource condition in review. Audit local complete video/query/time/box labels and compatible checkpoint; report unavailable rather than fabricate.'),
 ('E6','CGSTVG second backbone','queued_last','Explicit user substitution for TubeDETR. Local two checkpoints found. Official-code audit + native interface bitwisebridge; Source/SharedCorrection/Space/Full bothdomains; no target tuning; only AFTER TA suite.'),
 ('P','Exposure and selection','pending_audit','Full officialsets previously scored and contain developmentparents. New screen/confirm are parent-disjoint, prospectively frozen for these new contrasts; not unseen datasets or remediation of historical exposure. Export development-excluded statistic only with traceable historicalparentregistry.')]
plan=dict(schema='paper_review_suite_v1',authorization='Latest direct user request: all attachment experiments EXCEPTFig1, screening64-128parents/domain, cheapcached full12205, CG-STVG LOCAL LAST. This authorizes NEW isolated budget/dense expert and TTA optimizer studies; does not change production or old roots.',seed=20261024,selection_sha256=sha(D/'selection.json'),selection=dict(screen_parents_per_domain=64,confirm_parents_per_domain=128,one_query_per_parent=True,parent_disjoint=True,selection_uses='Manifestmetadata ONLY; no labels or quality',confirm_exposure='Previously scored fullsets; fixed new contrast evaluation, NOT unseen holdout.'),no_production_retuning=True,production_plan_sha256=sha(RUN/'plan.json'),source_fingerprint=jobplan['fingerprint'],review_attachment_sha256=sha('/home/wwww/.codex/attachments/01e75a03-e78a-4a3a-adc6-afd466f14913/已粘贴的文本.txt'),tasks=[dict(id=a,name=b,status=c,protocol=d)for a,b,c,d in items],staging='Cached full CPU and shared featurecollection first; new GPU/teacher/TTA candidates screen then fixedconfirm, no target-GT hyperparameter reselection; exact interfaces freeze before running; CGlast.',max_new_expert_models=0,expert_model='Same existing frozen ReferDINO; variable frame budgets are independent studies; production eight unchanged.',private_assets='No videos/captions/GTcoordinates/features/masks/checkpoints/credentials in public exports.')
plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',plan);atomic(P/'protocol.json',plan);atomic(D/'state.json',dict(status='active',fingerprint=plan['fingerprint'],tasks=plan['tasks'],production_unchanged=True));print(plan['fingerprint'])
