"""Bounded sensitivity/coordinate search; then exact baseline-only continuation."""
import sys,os,time,subprocess,fcntl,traceback,collections,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_opd_tuning_common_v1 import *

def run():
    lock();gate=(BASE/'CONTROLLER.lock').open('a');fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=read(BASE/'DESIGN_LOCK.json')
    def step(script,*args,allow_numerical_invalid=False):
        verify();budget();log=BASE/(Path(script).stem+'_'+'_'.join(args)+'.log')
        command=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/script),*args]
        with log.open('ab') as out:
            p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            status(BASE/'STATUS.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,
                   stage=Path(script).stem,args=list(args),log=str(log),time=time.time()))
            code=p.wait()
        if code:
            # An invalid configured trajectory has no score and cannot win.
            # All alignment/math/data errors remain hard engineering failures.
            if allow_numerical_invalid and 'Policy requires an invertible native box chart' in log.read_text(errors='replace'):
                return False
            raise RuntimeError('Worker failed: '+str(log))
        return True
    def config(ds,trial,phase,cfg,orders,parents,**extra):
        p=BASE/'trials'/ds/trial/'CONFIG.json'
        if not p.exists():write(p,dict(dataset=ds,trial=trial,phase=phase,config=cfg,orders=orders,parents=parents,
                                    development_selection=True,historical_exposure=True,**extra))
        else:assert read(p)['config']==cfg
    def trial(ds,name,phase,cfg,**extra):
        d=design['datasets'][ds];orders=d['screen_orders'] if phase=='screen' else d['refine_orders']
        parents=d['screen_parents'] if phase=='screen' else d['development_parents']
        config(ds,name,phase,cfg,orders,parents,**extra)
        dest=BASE/'trials'/ds/name
        if (dest/'NUMERICAL_INVALID.json').exists():return None
        if not (dest/'PREDICTION_BARRIER.json').exists():
            if not step('scripts/run_decota_opd_tuning_v1.py',ds,name,allow_numerical_invalid=True):
                write(dest/'NUMERICAL_INVALID.json',dict(status='invalid_trajectory_preserved',config=cfg,
                      not_scored=True,time=time.time()));return None
        if not (dest/'CPU_COMPLETION.json').exists():step('scripts/score_decota_opd_tuning_v1.py',ds,name)
        return read(PUB/ds/name/'SUMMARY.json')
    def rank(s):
        m=s['statistics']['metrics']['delta_total_v']
        return (m['mean'],-m['harm_gt20pp_sources'],-s['GPU_fit_seconds'])
    if not (BASE/'QUALIFICATION.json').exists():
        step('scripts/test_decota_opd_tuning_v1.py')
        receipts=[]
        for ds in ['hc2','vidstg']:
            d=design['datasets'][ds];parents=d['development_parents'][:2];orders={'order1':parents}
            for name,cfg in [('qual_default',DEFAULT),('qual_varied',{**DEFAULT,'lr':.01,'sigma':.1,'tau':.5,'steps':3,'writeback':0.})]:
                config(ds,name,'qualification',cfg,orders,parents)
                step('scripts/run_decota_opd_tuning_v1.py',ds,name)
                receipts.append(read(BASE/'trials'/ds/name/'QUALIFICATION.json'))
        assert all(r['status']=='pass' for r in receipts)
        assert any(r['informative_updates'] for r in receipts),'Qualification produced no live feedback gradient'
        write(BASE/'QUALIFICATION.json',dict(status='pass',real_queries=8,default_exact_parity_queries=4,
              receipts=receipts,GT_read=False,time=time.time()))
    selections={}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for ds in ['vidstg','hc2']:
        if (BASE/'selections'/f'{ds}.json').exists():selections[ds]=read(BASE/'selections'/f'{ds}.json');continue
        screen=[]
        # Seal all one-factor screen predictions before any screen GT scoring.
        d=design['datasets'][ds]
        for i,spec in enumerate(design['screen_configs']):
            name=f'screen_{i:02}';cfg=spec['config'];config(ds,name,'screen',cfg,d['screen_orders'],d['screen_parents'],factor=spec['factor'])
            dest=BASE/'trials'/ds/name
            if not (dest/'PREDICTION_BARRIER.json').exists() and not (dest/'NUMERICAL_INVALID.json').exists():
                if not step('scripts/run_decota_opd_tuning_v1.py',ds,name,allow_numerical_invalid=True):
                    write(dest/'NUMERICAL_INVALID.json',dict(status='invalid_trajectory_preserved',not_scored=True,time=time.time()))
        for i,spec in enumerate(design['screen_configs']):
            s=trial(ds,f'screen_{i:02}','screen',spec['config'],factor=spec['factor'])
            if s is not None:screen.append(s)
        assert screen
        reference=next(s for s in screen if s['config']==DEFAULT)
        sensitivity={}
        for factor in SPACE:
            vals=[rank(reference)[0]]+[rank(s)[0] for s in screen if read(BASE/'trials'/ds/s['trial']/'CONFIG.json')['factor']==factor]
            sensitivity[factor]=max(vals)-min(vals)
        sensitive=sorted(SPACE,key=lambda k:(-sensitivity[k],k))[:2]
        write(BASE/'selections'/f'{ds}_sensitivity.json',dict(effect_ranges=sensitivity,selected_coordinates=sensitive,
              screen_trials=len(screen),selection_data='16 exposed development sources',time=time.time()))
        incumbent=max(screen,key=rank)['config'];refined=[];seen={};studies={}
        for factor in sensitive:
            studies[factor]=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=20261006,n_startup_trials=3))
        for i in range(12):
            if i==0:cfg=dict(incumbent);proposal=None;factor='initial_incumbent'
            else:
                factor=sensitive[(i-1)%2];proposal=studies[factor].ask()
                val=proposal.suggest_categorical(factor,SPACE[factor]);cfg={**incumbent,factor:val}
            h=digest(cfg)
            if h in seen:s=seen[h]
            else:
                s=trial(ds,f'refine_{i:02}','refine',cfg,coordinate=factor);seen[h]=s
            if proposal is not None:
                if s is None:studies[factor].tell(proposal,state=optuna.trial.TrialState.FAIL)
                else:studies[factor].tell(proposal,rank(s)[0])
            if s is not None:
                refined.append(s);incumbent=max(refined,key=rank)['config']
            status(BASE/'SEARCH_PROGRESS.json',dict(dataset=ds,refine_proposals=i+1,maximum_refine_proposals=12,
                   unique_refine_configs=len(seen),coordinates=sensitive,time=time.time()))
        assert refined
        best=max(refined,key=rank)
        # Independent parent mean arithmetic verifies the selected aggregate.
        with gzip.open(PUB/ds/best['trial']/'ROWS.jsonl.gz','rt') as f:rr=[json.loads(x) for x in f]
        group=collections.defaultdict(list)
        for r in rr:group[r['source_id']].append(r['delta_total_v'])
        import numpy as np
        a=np.array([sum(v)/len(v) for _,v in sorted(group.items())]);assert len(a)==32
        assert abs(a.mean()-rank(best)[0])<1e-12
        rng=np.random.default_rng(20261006);bs=np.array([a[rng.integers(len(a),size=len(a))].mean() for _ in range(10000)])
        assert np.max(np.abs(np.quantile(bs,[.025,.975])-best['statistics']['metrics']['delta_total_v']['ci95']))<1e-12
        selected=dict(dataset=ds,status='selected_on_development',config=best['config'],trial=best['trial'],
                      summary_sha256=sha(PUB/ds/best['trial']/'SUMMARY.json'),statistics=best['statistics'],
                      sensitive_parameters=sensitive,refine_proposals=12,unique_refine_configs=len(seen),
                      independent_selected_mean_and_bootstrap=True,old_confirmation_used=False,time=time.time())
        write(BASE/'selections'/f'{ds}.json',selected);write(PUB/ds/'SELECTED.json',selected);selections[ds]=selected
    cfgfile=ROOT/'methods/decota_spatial_opd_v1/configs.json'
    status(cfgfile,dict(status='selected_on_exposed_development',datasets={ds:dict(status='selected',config=s['config'],
           selection=str((BASE/'selections'/f'{ds}.json').relative_to(ROOT))) for ds,s in selections.items()},time=time.time()))
    write(BASE/'SELECTION_BARRIER.json',dict(status='sealed',datasets=selections,configuration_file_sha256=sha(cfgfile),
          official_confirmation_used=False,other_method_experiments_run=False,time=time.time()))
    write(PUB/'SELECTION_BARRIER.json',read(BASE/'SELECTION_BARRIER.json'))
    archive('两集有限敏感性搜索实际完成并独立复核选定配置，methods/decota_spatial_opd_v1/configs.json已写入各一套开发参数；参数不是独立confirmation结论，按用户最新顺序立即接续原baseline完整状态')
    status(BASE/'TUNING_COMPLETION.json',dict(status='completed_selection_pending_root_publication',datasets=['vidstg','hc2'],
          baseline_resume_next=True,time=time.time()))
    step('scripts/continue_decota_paper_baselines_only_v2.py')
    status(BASE/'STATUS.json',dict(status='tuning_complete_baseline_continuation_handoff',time=time.time()))

if __name__=='__main__':
    try:run()
    except BaseException:
        d=BASE/'controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc());status(BASE/'STATUS.json',dict(status='failed_preserved',failure=str(d),time=time.time()));raise
