"""HC2-only finite greedy coordinate search; original science locks stay intact."""
import hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
from scripts.decota_opd_tuning_common_v1 import verify_paper,bridge
BASE=ROOT/'artifacts/stvg_opd_hc2_coordinate_v1'
OLD=ROOT/'artifacts/decota_spatial_opd_tuning_v1'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
PUB=ROOT/'results/stvg_opd_hc2_coordinate/2026-10-08'
PYTHON=ROOT/'.conda/tubedetr/bin/python'
COORDINATES=['lr','sigma','tau','steps','writeback']
GRIDS=dict(lr=[.001,.003,.01,.03,.1],sigma=[.025,.05,.1,.25,.5],
    tau=[.05,.1,.25,.5,1.],steps=[1,3,5,10,20,40],writeback=[0.,1/64,1/32,1/16,1/8])
START=dict(lr=.03,sigma=.1,tau=.25,steps=20,writeback=1/16,samples=32)
FIELDS=['Frozen_v','Before_v','After_v','Frozen_t','After_t','Frozen_s','Before_s','After_s',
    'delta_total_v','delta_current_v','delta_inherited_v','delta_total_s',
    'correct_to_wrong_0.3','correct_to_wrong_0.5','wrong_to_correct_0.3','wrong_to_correct_0.5']

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'Free disk below 8 GiB'
def candidate_configs(incumbent,coordinate):return [{**incumbent,coordinate:v} for v in GRIDS[coordinate]]
def trial_id(cfg):return 'cfg_'+digest(cfg)[:16]
def rank(summary,incumbent,candidate_ordinal):
    m=summary['statistics']['metrics']['delta_total_v'];cfg=summary['config']
    return (m['mean'],-m['harm_gt20pp_sources'],int(cfg==incumbent),-cfg['steps'],-candidate_ordinal)
def commit(initial,final,alpha):
    import torch
    return {n:torch.zeros_like(v.cpu()) if n=='spatial.query_residual' else v.cpu()+(final[n].cpu()-v.cpu())*alpha for n,v in initial.items()}

def verify():
    verify_paper();r=read(BASE/'RUNTIME_LOCK.json')
    assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    return r

def prepare():
    if (BASE/'DESIGN_LOCK.json').exists():return read(BASE/'DESIGN_LOCK.json')
    old=read(OLD/'DESIGN_LOCK.json')['datasets']['hc2'];plan=read(PAPER/'hc2/PLAN.json')
    qs=old['development_parents'];sources={plan['rows'][q]['source'] for q in qs}
    assert len(qs)==len(sources)==32
    paper=read(ROOT/'artifacts/stvg_opd_paper_v1/DESIGN_LOCK.json')
    assert not sources & {plan['rows'][q]['source'] for q in paper['datasets']['hc2']['confirmation_query_ordinals']}
    assert read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets']['hc2']['config']==START
    d=dict(version='stvg_opd_hc2_coordinate_v1',dataset='hc2',source=old['source'],parents=qs,
        orders=old['refine_orders'],source_count=32,queries=32,arrivals_per_config=64,
        historically_exposed_development=True,confirmation_or_P1_scores_used_for_selection=False,
        coordinates=COORDINATES,grids=GRIDS,start=START,passes=1,
        candidate_evaluations_maximum=26,unique_configurations_maximum=22,
        adaptation_arrivals_maximum=1408,qualification_fits=4,
        selection='parent-macro delta_v; ties fewer harm20 parents, incumbent, fewer steps, grid ordinal',
        original_development_design_sha256=sha(OLD/'DESIGN_LOCK.json'),
        PLAN_sha256=sha(PAPER/'hc2/PLAN.json'),
        paused_P1_receipt_sha256=sha(ROOT/'artifacts/stvg_opd_paper_v1/user_hc2_coordinate_pause_20261008/PAUSE_RECEIPT.json'),
        EATA_user_paused=True,VidSTG_configuration_unchanged=True,time=time.time())
    write(BASE/'DESIGN_LOCK.json',d);return d

def lock():
    prepare()
    if not (BASE/'RUNTIME_LOCK.json').exists():
        files=['scripts/stvg_opd_hc2_coordinate_common_v1.py','scripts/run_stvg_opd_hc2_coordinate_v1.py',
            'scripts/score_stvg_opd_hc2_coordinate_v1.py','scripts/continue_stvg_opd_hc2_coordinate_v1.py',
            'scripts/finalize_stvg_opd_hc2_coordinate_v1.py','scripts/test_stvg_opd_hc2_coordinate_v1.py',
            'protocols/stvg_opd_hc2_coordinate_v1.md','vg_tta/decota_spatial_opd_tunable_v1.py',
            'vg_tta/decota_spatial_opd_tunable_audit_v1.py','scripts/run_decota_spatial_opd_v1.py',
            'scripts/run_decota_paper_main_v1.py','methods/decota_spatial_opd_v1/predictor.py',
            'methods/decota_spatial_opd_v1/configs.json','methods/CURRENT_METHOD.json']
        write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},
            design_sha256=sha(BASE/'DESIGN_LOCK.json'),scientific_core_unchanged=True,
            original_paper_runtime_sha256=sha(ROOT/'artifacts/stvg_opd_paper_v1/RUNTIME_LOCK.json'),time=time.time()))
    return verify()

def ensure_trial(cfg,qualification=False):
    d=read(BASE/'DESIGN_LOCK.json');uid=('qual_' if qualification else '')+trial_id(cfg)
    f=BASE/'trials'/uid/'CONFIG.json'
    t=dict(dataset='hc2',source=d['source'],trial=uid,config=cfg,phase='qualification' if qualification else 'coordinate',
        parents=d['parents'][:2] if qualification else d['parents'],
        orders={'order1':d['orders']['order1'][:2]} if qualification else d['orders'],
        historical_exposure=True,development_selection=not qualification)
    if f.exists():assert read(f)==t
    else:write(f,t)
    return uid

def exact_prior_trial(t):
    matches=[]
    for p in sorted((OLD/'trials/hc2').glob('*/CONFIG.json')):
        x=read(p)
        if x['config']!=t['config'] or x['parents']!=t['parents'] or x['orders']!=t['orders']:continue
        if not (p.parent/'PREDICTION_BARRIER.json').exists():continue
        matches.append(p.parent)
    return matches[0] if matches else None

def seal_exact_alias(uid):
    dest=BASE/'trials'/uid;t=read(dest/'CONFIG.json');old=exact_prior_trial(t)
    if old is None:return False
    barrier=read(old/'PREDICTION_BARRIER.json');assert barrier['status']=='sealed' and not barrier['GT_read']
    assert barrier['config_sha256']==sha(old/'CONFIG.json') and barrier['cells']==64
    files={};inputs={};count=0
    for order,seq in t['orders'].items():
        for at,q in enumerate(seq):
            p=old/order/f'{at:05}.pt';h=sha(p);rc=read(p.with_suffix('.json'))
            assert barrier['files'][str(p.relative_to(OLD))]==h==rc['sha256'] and not rc['GT_read']
            assert rc['time']<=barrier['time'];key=str(p.relative_to(ROOT));files[key]=h;count+=1
    assert count==64
    write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',trial=uid,cells=64,files=files,
        input_base=str(OLD.relative_to(ROOT)),config_sha256=sha(dest/'CONFIG.json'),
        exact_history_reused=True,origin=str(old.relative_to(ROOT)),
        original_barrier_sha256=sha(old/'PREDICTION_BARRIER.json'),original_seal_time=barrier['time'],
        GT_read=False,time=time.time()))
    status(dest/'STATUS.json',dict(status='sealed_exact_prior_development_history',done=64,total=64,GT_read=False,time=time.time()))
    return True

def records(uid):
    dest=BASE/'trials'/uid;t=read(dest/'CONFIG.json');b=read(dest/'PREDICTION_BARRIER.json')
    origin=ROOT/b['origin'] if b['exact_history_reused'] else dest
    for order,seq in t['orders'].items():
        for at,q in enumerate(seq):
            p=origin/order/f'{at:05}.pt';yield order,at,q,p,ROOT/b['input_base']

def coordinate_barrier(index):
    d=BASE/'coordinates'/f'{index:02}_{COORDINATES[index]}'
    b=read(d/'PREDICTION_BARRIER.json');c=read(d/'CONFIG.json')
    assert b['status']=='all_candidates_sealed_or_unscored_numerical_invalid' and b['config_sha256']==sha(d/'CONFIG.json')
    for v in c['candidates']:
        p=BASE/'trials'/v['trial'];name='PREDICTION_BARRIER.json' if (p/'PREDICTION_BARRIER.json').exists() else 'INVALID_DISPOSITION.json'
        assert b['dispositions'][v['trial']][name]==sha(p/name)
    return c,b
