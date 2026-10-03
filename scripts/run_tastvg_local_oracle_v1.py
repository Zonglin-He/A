"""Bounded CPU analysis of previously published anonymous candidate metrics."""
import json,sys,time,hashlib,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.tastvg_local_oracle_math_v1 import derive,summarize,RADII,EPS
BASE=ROOT/'artifacts/tastvg_temporal_local_oracle_v1'
PUBLIC=ROOT/'results/tastvg_temporal_local_oracle/2026-10-03'
PRIOR=ROOT/'results/tastvg_temporal_boundary_support/2026-10-03'
COMMIT='4ecf366dee49c844f1af7aabfcba5517d74c86e4'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj,mutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert mutable or not p.exists(),p
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def status(value,**kwargs):write(BASE/'STATUS.json',dict(status=value,time=time.time(),**kwargs),True)

def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    receipt=read(ROOT/'artifacts/tastvg_temporal_boundary_support_v1/FINAL_COMPLETION.json')
    assert receipt['status']=='completed_and_verified_publication' and receipt['commit']==COMMIT
    manifest=read(ROOT/'artifacts/tastvg_temporal_boundary_support_v1/PUBLIC_MANIFEST.json')['files']
    for name,entry in manifest.items():assert sha(ROOT/name)==entry['sha256'],name
    paths=[str((PRIOR/sp/ds/f).relative_to(ROOT)) for sp in ['search','confirm']
           for ds in ['vidstg','hc2'] for f in ['ROWS.json','SUMMARY.json']]
    paths.append(str((PRIOR/'CONFIG.json').relative_to(ROOT)))
    pins=['protocols/tastvg_temporal_local_oracle_v1.md',
          'docs/tastvg_temporal_local_oracle_v1/EXECUTION.md',
          'scripts/tastvg_local_oracle_math_v1.py','scripts/run_tastvg_local_oracle_v1.py',
          'scripts/audit_tastvg_local_oracle_public_v1.py','scripts/report_tastvg_local_oracle_v1.py',
          'scripts/test_tastvg_local_oracle_v1.py']
    config=read(PRIOR/'CONFIG.json')
    lock=dict(version='tastvg_temporal_local_oracle_v1',predecessor_commit=COMMIT,
        predecessor_files_verified=len(manifest),inputs={p:sha(ROOT/p) for p in paths},
        pins={p:sha(ROOT/p) for p in pins},radii=RADII,tie_epsilon=EPS,
        source_bootstrap_draws=10000,seed=20261003,
        previous_design=config['design'],previous_sampling=config['sampling'],
        production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        input_use='already_sealed_anonymous_GT_derived_metrics_posthoc_diagnosis',
        annotation_files_opened=0,new_model_calls=0,new_expert_calls=0,new_predictions=0,
        GPU_used=False,new_refinement_started=False,time=time.time())
    assert lock['production_method_sha256']==config['production_method_sha256']
    write(BASE/'RUNTIME_LOCK.json',lock);write(PUBLIC/'CONFIG.json',lock)
    status('locked_pending_cpu_analysis',expert_candidate_cells=0)
    print('LOCAL_ORACLE_INPUT_RULES_LOCKED',flush=True)

def verify():
    lock=read(BASE/'RUNTIME_LOCK.json')
    for f in sorted((BASE/'revisions').glob('revision_*.json')):
        revision=read(f);assert revision['base_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        assert revision['science_changed'] is False and set(revision['pin_overrides'])<=set(lock['pins'])
        lock['pins'].update(revision['pin_overrides'])
    for p,h in {**lock['inputs'],**lock['pins']}.items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==lock['production_method_sha256']
    return lock

def compute():
    started=time.time();lock=verify();status('running_cpu_posthoc_analysis')
    write(PUBLIC/'EFFECTIVE_PINS.json',lock['pins'])
    totals=dict(arrivals=0,experts=0,nonexpert_without_candidates=0,corrupt_experts=0,clean_experts=0)
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            old=read(PRIOR/split/ds/'ROWS.json');rr=[]
            assert len(old)==(384 if split=='search' else 192)
            for r in old:
                totals['arrivals']+=1
                if not r['expert_scheduled']:
                    assert 'candidate_v' not in r
                    assert all(abs(r[a+'_v']-r['A8_v'])<=EPS for a in ['B8','D8','A32','B32','D32'])
                    totals['nonexpert_without_candidates']+=1;continue
                q=derive(r);q['input_row_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert abs(q['A8_v']-r['A8_v'])<1e-10 and abs(q['capacity_gain']-r['capacity_gain'])<1e-10
                rr.append(q);totals['experts']+=1
                totals['clean_experts' if r['condition']=='clean' else 'corrupt_experts']+=1
            sums={}
            for group in ['corruption','clean']:
                part=[r for r in rr if (r['condition']=='clean')==(group=='clean')]
                sums[group]=summarize(part)
                predecessor=read(PRIOR/split/ds/'SUMMARY.json')[group]['expert']
                assert sums[group]['sources']==predecessor['sources']
                assert abs(sums[group]['metrics']['capacity_gain']['mean']-predecessor['metrics']['capacity_gain']['mean'])<1e-10
            write(PUBLIC/split/ds/'ROWS.json',rr);write(PUBLIC/split/ds/'SUMMARY.json',sums)
            examples={}
            for group in ['corruption','clean']:
                part=[r for r in rr if (r['condition']=='clean')==(group=='clean')]
                pos=[r for r in part if r['capacity_gain']>EPS]
                for kind,items in [('largest_added_gain',sorted(pos,key=lambda r:-r['capacity_gain'])[:4]),
                        ('nearest_gain_positive',sorted(pos,key=lambda r:(r['nearest_distance_anchor'],-r['capacity_gain']))[:4]),
                        ('farthest_gain_positive',sorted(pos,key=lambda r:(-r['nearest_distance_anchor'],-r['capacity_gain']))[:4])]:
                    examples[group+'/'+kind]=items
            write(PUBLIC/split/ds/'CASES.json',examples)
            status('running_cpu_posthoc_analysis',last_completed=ds+'/'+split,coverage=totals)
            print('PANEL_COMPLETE',ds,split,len(rr),flush=True)
    assert totals==dict(arrivals=1152,experts=288,nonexpert_without_candidates=864,corrupt_experts=240,clean_experts=48)
    verify()
    write(PUBLIC/'COVERAGE.json',totals)
    write(PUBLIC/'RESOURCES.json',dict(CPU_wall_seconds=time.time()-started,wall_is_not_GPU_kernel_time=True,
        new_model_calls=0,new_expert_calls=0,new_decoder_replays=0,new_backprop=0,new_predictions=0,
        private_GT_annotation_files_opened=0,GPU_used=False,python=platform.python_version()))
    status('completed_pending_independent_audit_report_publication',coverage=totals)
    print('LOCAL_ORACLE_CPU_COMPLETE',json.dumps(totals),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'compute':compute}[sys.argv[1]]()
