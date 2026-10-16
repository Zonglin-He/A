"""Finite overnight state, strict reservations, and post-seal global decisions."""
from pathlib import Path
import hashlib,json,math,os,time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_overnight_consolidation_20261016'
CORE=ROOT/'artifacts/stvg_spatial_controls256_v1'
OLDGLOBAL=ROOT/'artifacts/stvg_global32_context_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not replace:assert not p.exists(),str(p)
    tmp=p.with_name(p.name+'.partial.'+str(os.getpid()));tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def log(phase,status,**fields):
    with (BASE/'DECISION_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),phase=phase,status=status,**fields),allow_nan=False)+'\n')
def status(stage,**fields):write(BASE/'STATUS.json',dict(status='running',stage=stage,GT_read=False,**fields),True)
def deadline():return read(BASE/'PLAN_LOCK.json')['deadline_epoch']
def verify():
    p=BASE/'EXECUTION_RUNTIME001.json'
    if not p.exists():p=BASE/'P0_DISPATCH_RUNTIME.json'
    d=read(p)
    for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==read(BASE/'ENTRY_STATE.json')['CURRENT_sha256']
def reserve(label,WHO=0,Temporal=0,fits=0):
    p=BASE/'BUDGET_LEDGER.json';d=read(p) if p.exists() else dict(status='locked',reservations=[],caps=dict(WHO=2560,Temporal=512,fits=1280),GT_read=False)
    if any(r['label']==label for r in d['reservations']):return
    r=dict(label=label,WHO=WHO,Temporal=Temporal,fits=fits,scope='upper-bound charged until final actual accounting',time=time.time())
    for k in d['caps']:assert sum(x[k] for x in d['reservations'])+r[k]<=d['caps'][k],('budget',k,label)
    d['reservations'].append(r);write(p,d,True)
def score(arm):
    from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
    validate_barrier(arm);b=CORE/'trials'/arm;p=b/'POSTSEAL_RESULTS002.json';c=read(b/'CPU_COMPLETION002.json')
    assert c['status']=='complete' and c['count']==256 and c['GT_read'] and c['result_sha256']==sha(p)
    return read(p)
def choose_core():
    p=BASE/'CORE_DECISION.json'
    if p.exists():return read(p)
    verify();a=score('B');b=score('B_BT');ia={(r['case_id'],r['arm']):r for r in a['rows']};ib={(r['case_id'],r['arm']):r for r in b['rows']};rows=[]
    refs=[read(CORE/'trials'/arm/'REFERENCE_LOCK.json') for arm in ('B','B_BT')]
    assert refs[0]['records']==refs[1]['records']
    designs=[read(CORE/'trials'/arm/'DESIGN_LOCK.json') for arm in ('B','B_BT')]
    for k in ('configs','response_settings'):assert designs[0][k]==designs[1][k]
    for ds in ('hc2','vidstg'):
        ids=[r['probe_alias'] for r in read(CORE/'ROSTER_LOCK.json')['records'] if r['dataset']==ds];assert len(ids)==128
        for cid in ids:
            for control in ('Frozen','Temporal','Full_E1','Full_E1_Spatial'):assert ia[cid,control]['metrics']==ib[cid,control]['metrics']
        values=[100*(ib[cid,'B_BT']['metrics']['m_vIoU']-ia[cid,'B']['metrics']['m_vIoU']) for cid in ids]
        rows.append(dict(dataset=ds,delta_pp=math.fsum(values)/128,positive=sum(x>1e-6 for x in values),negative=sum(x< -1e-6 for x in values),neutral=sum(abs(x)<=1e-6 for x in values)))
    times={arm:read(CORE/'execution/native256_core6'/arm/'student_worker/WORKER_EXIT.json')['wall_seconds'] for arm in ('B','B_BT')}
    profile=read(ROOT/'artifacts/stvg_pipeline_speed_v1/student_profiles/B.json')
    extra=sum(max(0,r['capture_wall_seconds']-r['model_load_seconds']) for r in profile['rows'] if not r['cache_hit'])
    lower=times['B']-extra;assert lower>0;ratio=times['B_BT']/lower;M=math.fsum(r['delta_pp'] for r in rows)/2
    passed=M>.05 and min(r['delta_pp'] for r in rows)>=-.20 and ratio<=1.25
    d=dict(status='locked',count=256,selected_arm='B_BT' if passed else 'B',recipe='lite_backtrack' if passed else 'lite_fullstep',mean_domain_delta_pp=M,domain_deltas=rows,worker_wall_seconds=times,B_new_capture_seconds_removed_for_conservative_cost_bound=extra,conservative_matched_worker_time_ratio=ratio,criteria=dict(M_strictly_greater_than=.05,domain_loss_at_most=.20,time_ratio_at_most=1.25),rule_sha256=sha(BASE/'PLAN_LOCK.json'),cost_rule_sha256=sha(BASE/'RULE_AND_CLOCK_BINDING002.json'),registered_before_BBT_score=True,shared_controls_references_configs_exact=True,GT_read=True,scope='development-only selection, not confirmation or production')
    write(p,d);reserve('night_A_BT_and_B_BT',fits=sum(x['actual_fit'] for x in score('A_BT')['usage'])+sum(x['actual_fit'] for x in b['usage']));log('P0','core_selected',selected_arm=d['selected_arm'],decision_sha256=sha(p));return d
