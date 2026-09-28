"""F41 larger development pool and source-fold tuning; all old artifacts read-only.

Prepare fixes the input pools, candidate menu and selection rules without GT.
The episode fitter never reads labels. Search/choose use development labels only;
the evaluation panel is scored once after an immutable selection is written.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.dense_support_tuning_v1 import keyfile,config_id,METRICS

OUT=ROOT/'artifacts/decota_dense_support_generalization_v1'
F40=ROOT/'artifacts/decota_dense_support_tuning_v1'
F37=ROOT/'artifacts/decota_spatial4_attribute_v1'
F36=ROOT/'artifacts/decota_simplification_partial_v1'
OWN=['scripts/dense_support_generalization_v1.py','vg_tta/dense_support_prefix_v1.py']
BUDGETS=[1,3,5,10,20,30,50,75,100,150,200,300,500]


def hkey(r,tag):return hashlib.sha256((tag+'|'+r['group']).encode()).hexdigest()


def menu():
    pairs={}
    def add(lr,beta,maximum=100):
        cid=config_id(dict(lr=float(lr),beta=float(beta)))
        pairs[cid]=dict(id=cid,lr=float(lr),beta=float(beta),maximum=max(maximum,pairs.get(cid,{}).get('maximum',0)))
    for lr in [1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,.006189725770618397,.01,.03,.1,1.,10.,100.]:
        for beta in [0.,.01,.1,1.,10.]:add(lr,beta)
    for lr in [.00015,.0002,.0004,.0006,.0008,.002,.004,.006,.008,.012,.02]:
        for beta in [0.,.1,1.]:add(lr,beta)
    for lr in [.003,.006189725770618397,.01]:
        for beta in [.03,.3,.5,.75,.8085669875668131,1.5,3.]:add(lr,beta)
    for lr in [.0001,.0003,.001]:
        for beta in [0.,.1]:add(lr,beta,500)
    add(.006189725770618397,.8085669875668131,500)
    add(.01,1.,500)
    for p in pairs.values():p['budgets']=[n for n in BUDGETS if n<=p['maximum']]
    return list(sorted(pairs.values(),key=lambda x:(x['lr'],x['beta'])))


def prepare():
    old=read(F40/'LOCK.json');p37=read(F37/'LOCK.json');seal=read(F40/'COMPLETION.json')
    protected={**old['protected_pins'],**seal['files'],**seal['code_pins'],str((F40/'COMPLETION.json').relative_to(ROOT)):sha(F40/'COMPLETION.json')}
    rows={}
    for c,rr in old['rows'].items():
        dev=[copy.deepcopy(r) for r in rr if r['f37_role']=='development']
        ev=[copy.deepcopy(r) for r in rr if r['f37_role']=='evaluation']
        extra=sorted(copy.deepcopy(p37['space_extension'][c]),key=lambda r:hkey(r,'F41_additional_split_v1'))
        assert len(extra)==(33 if c=='hcstvg1_test' else 32)
        for r in dev:r.update(f41_role='development',f41_panel='old_development')
        for r in ev:r.update(f41_role='evaluation',f41_panel='F40_repeated_evaluation')
        for i,r in enumerate(extra):
            r.update(f41_role='development' if i<24 else 'evaluation',
                     f41_panel='additional_development' if i<24 else 'additional_locked_evaluation')
        allr=dev+ev+extra
        d=sorted([r for r in allr if r['f41_role']=='development'],key=lambda r:hkey(r,'F41_four_folds_v1'))
        assert len(d)==32
        for i,r in enumerate(d):r['f41_fold']=i%4
        for i,r in enumerate(allr):
            f=F36/'space'/keyfile(r['key']);arm='A4'
            if not f.exists():f=F37/'space'/keyfile(r['key']);arm='S4'
            assert f.is_file();receipt=read(f.with_suffix('.json'));assert sha(f)==receipt['sha256']
            r['A4_reference']=dict(path=str(f),sha256=sha(f),arm=arm)
            protected[str(f.relative_to(ROOT))]=sha(f)
            if (F40/'cache'/keyfile(r['key'])).exists():
                r['cache_reference']=str(F40/'cache'/keyfile(r['key']))
            else:
                donor=next(t for t in d[i%32:]+d[:i%32] if t['group']!=r['group'])
                r['F41_wrong_query']=dict(key=donor['key'],caption=donor['input']['caption'],subject=donor['subject'])
        assert len({r['group'] for r in allr})==len(allr)
        for field in ['group','source']:
            assert not {r[field] for r in allr if r['f41_role']=='development'}&{r[field] for r in allr if r['f41_role']=='evaluation'}
        assert not {r['input']['video_sha256'] for r in allr if r['f41_role']=='development'}&{r['input']['video_sha256'] for r in allr if r['f41_role']=='evaluation'}
        rows[c]=allr
    for f,h in protected.items():assert sha(ROOT/f)==h,f
    specs=menu()
    lock=dict(name='F41_dense_support_larger_dev_source_fold_HPO',created=time.time(),rows=rows,
        protected_pins=protected,code_pins={f:sha(ROOT/f) for f in OWN},menu=specs,
        authority='User explicitly requests further broad per-dataset tuning focused on development-to-evaluation transfer.',
        fixed=old['fixed'],spatial=old['spatial'],historical_exposure=True,untouched=False,
        selection=dict(development_per_direction=32,folds=4,sources_per_fold=8,
            menu='fixed before new fits, informed by prior F40; no adaptive use of held-out fold metrics in menu generation',
            rules=['mean','mean_minus_one_se'],
            rule_choice='For each rule choose on the other24 sources, score held8 in each fold; maximize mean held-fold delta_v. Tie <=1e-12 prefers mean_minus_one_se.',
            final='Select one config separately per direction on all32 development under the chosen rule.',
            mean_minus_one_se='mean(delta_v)-sample_std(delta_v)/sqrt(n); GT used offline only',
            tie='fewer steps then lr closer to original .01 then smaller beta',
            caveat='Fold estimates are development diagnostics, not fresh test; original8 and prior F40 informed menu. Rule selection reuses fold outcomes.',
            evaluation='original16 per direction plus extra9 HC/8 Vid; no parameter choice on any of them; separate panel reporting',
            controls=['T0','T1','Tmean','Tstable','T40','T2','T3','T4'],
            no_eval_retuning=True,no_per_sample_GT_fallback=True),
        search=dict(backend='Optuna GridSampler over locked lr/beta pairs; all trajectory-prefix budgets scored',
            seeds={'hcstvg1_test':202609141,'vidstg_test':202609142},pairs_per_direction=len(specs),
            configs_per_direction=sum(len(s['budgets']) for s in specs),prefix_reuse=True,
            exact_rejected_tail='Only after two bit-identical rejected proposals with optimizer+parameter restoration; actual and virtual counts separated',
            no_pruning=True),
        caps=dict(actual_trajectories=12000,actual_backwards=1600000,wall_seconds=14400,new_DINO=0,new_space_fits=0),
        registries_change=False,new_backbone=False,corruption=False,automation=False)
    write(OUT/'LOCK.json',lock);write(OUT/'STATUS.json',dict(stage='prepared',finished=False))
    write(OUT/'ENGINEERING_NOTES.json',dict(initial_unit_fixture_error='Synthetic endpoint logits omitted required leading batch dimension; four tests stopped at native decoder shape assertion before any fit. Fixture fixed from (8,2) to (1,8,2), production decoder/algorithm unchanged.'))
    print('F41 locked',[(c,len(rr)) for c,rr in rows.items()],len(specs),'pairs',sum(len(s['budgets']) for s in specs),'prefix configs per direction',flush=True)


def plan(parents=False):
    p=read(OUT/'LOCK.json')
    for f,h in p['code_pins'].items():assert sha(ROOT/f)==h,f
    if parents:
        for f,h in p['protected_pins'].items():assert sha(ROOT/f)==h,f
    return p


def record(path,value):
    save(path,value);write(path.with_suffix('.json'),dict(key=value.get('key'),sha256=sha(path),completed=time.time(),code_pins={f:sha(ROOT/f) for f in OWN}))


def recorded(path):
    return path.exists() and sha(path)==read(path.with_suffix('.json'))['sha256']


def cache_for(r):
    path=Path(r.get('cache_reference',OUT/'cache'/keyfile(r['key'])))
    assert recorded(path),path
    x=load(path)
    x['A4_reference']=r['A4_reference']
    return x


def a4_for(r):
    info=r['A4_reference'];assert sha(info['path'])==info['sha256']
    old=load(info['path']);return old,old['fits'][info['arm']]


def capture(role):
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.parametric_observation_v1 import ObservationReplay
    from vg_tta.simplification_partial_v1 import FullInputHeadReplay
    from vg_tta.dense_support_temporal_v1 import teacher_from_logits
    from vg_tta.dense_support_tuning_v1 import move,HeadReplay
    from vg_tta.foreground_runtime import state_digest
    p=plan(True)
    if role=='evaluation':assert (OUT/'SELECTION.json').exists()
    for c,rr in p['rows'].items():
        pending=[r for r in rr if r['f41_role']==role and 'cache_reference' not in r and not recorded(OUT/'cache'/keyfile(r['key']))]
        if not pending:continue
        model=model_for(c);digest=state_digest(model)
        for r in pending:
            frames,_,records,views,qa,cost=capture_timed(model,r);ids=r['input']['frame_ids']
            it=FullInputHeadReplay(ObservationReplay(model,views,len(ids),'head'));z=it.zero
            old,a4=a4_for(r)
            assert qa==old['qa'] and torch.equal(z['boxes'].cpu(),old['native_boxes'])
            assert all(torch.equal(v.cpu(),w) for v,w in zip(z['logits'],old['native_logits']))
            t0=dict(boxes=a4['final']['boxes'],logits=old['native_logits'],indices=old['native_indices'])
            teacher=teacher_from_logits(z['actions'],records)
            dense=[];hook=model.register_forward_hook(lambda m,a,o:dense.append(o['pred_actioness'].detach().cpu().clone()))
            try:audit=full_prediction(model,frames,ids,r['input'],r['subject'],a4['state'],t0)['audit']
            finally:hook.remove()
            assert len(dense)==2 and all(torch.equal(v,w.cpu()) for v,w in zip(dense,z['actions']))
            donor=r['F41_wrong_query'];dr=copy.deepcopy(r)
            dr['input']['caption']=donor['caption'];dr['subject']=donor['subject']
            _,_,drecords,dviews,dqa,_=capture_timed(model,dr)
            dit=FullInputHeadReplay(ObservationReplay(model,dviews,len(ids),'head'))
            wrong=teacher_from_logits(dit.zero['actions'],drecords)
            assert dqa==qa and [x['frame_ids'] for x in records]==[x['frame_ids'] for x in drecords]
            rolled=[{**t,'a':torch.roll(t['a'],len(t['a'])//2).detach().clone(),
                     'raw_logits':torch.roll(t['raw_logits'],len(t['a'])//2).detach().clone(),'roll':len(t['a'])//2} for t in teacher]
            cache=dict(key=r['key'],cohort=c,group=r['group'],source=r['source'],role=role,frame_ids=ids,
                records=records,inputs=move(it.inputs,'cpu'),head_state=move(it.initial,'cpu'),
                zero=dict(boxes=t0['boxes'],logits=t0['logits']),teacher=move(teacher,'cpu'),
                wrong_teacher=move(wrong,'cpu'),rolled_teacher=move(rolled,'cpu'),wrong_query=donor,T0=t0,
                qa=qa,capture_seconds=cost,GT_online=False,A4_reference=r['A4_reference'],
                exact_native_logits=True,final_dense_field_exact=True,native_full_model=audit,new_DINO=0)
            portable=HeadReplay(cache);del portable
            record(OUT/'cache'/keyfile(r['key']),cache)
            assert state_digest(model)==digest
            del it,dit,views,dviews,frames,z
            print('F41 capture',role,r['key'],flush=True)
            status(OUT/'STATUS.json',dict(stage='capture_'+role,last=r['key'],finished=False))
        del model;gc.collect();torch.cuda.empty_cache()


def run_trajectory(cache,spec,arm='T1'):
    from vg_tta.dense_support_tuning_v1 import HeadReplay,move
    from vg_tta.dense_support_temporal_v1 import OutputReplay
    from vg_tta.dense_support_prefix_v1 import fit_prefixes
    it=HeadReplay(cache);teacher=it.teacher
    if arm=='T2':it=OutputReplay(it.zero)
    if arm=='T3':teacher=move(cache['wrong_teacher'],'cuda')
    if arm=='T4':teacher=move(cache['rolled_teacher'],'cuda')
    return fit_prefixes(it,cache['records'],cache['frame_ids'],teacher,
        lr=spec['lr'],beta=spec['beta'],budgets=spec['budgets'],output_control=arm=='T2')


def equivalence():
    import torch
    from vg_tta.dense_support_tuning_v1 import HeadReplay,state_hash
    from vg_tta.dense_support_prefix_v1 import selected_fit
    from vg_tta.dense_support_temporal_v1 import fit
    p=plan(True);result=[]
    for c,rr in p['rows'].items():
        r=next(r for r in rr if r['f41_role']=='development');cache=cache_for(r)
        for lr,beta,nn in [(.01,1.,[0,1,5,100]),(.0003,0.,[0,5,100]),(100.,1e4,[0,1,5])]:
            tr=run_trajectory(cache,dict(lr=lr,beta=beta,budgets=nn));it=HeadReplay(cache)
            for n in nn:
                ref=fit(it,cache['records'],cache['frame_ids'],it.teacher,lr=lr,beta=beta,steps=n)
                z=selected_fit(tr,n,cache['T0']['boxes'])
                assert z['best_step']==ref['best_step'] and z['final']['loss']==ref['final']['loss']
                assert state_hash(z['state'])==state_hash(ref['state'])
                assert all(torch.equal(v,w) for v,w in zip(z['final']['logits'],ref['final']['logits']))
                result.append(dict(key=r['key'],lr=lr,beta=beta,steps=n,exact=True,reference_backwards=ref['backwards']))
            record(OUT/'equivalence'/(r['key'].replace(':','_')+'_'+config_id(dict(lr=lr,beta=beta))+'.pt'),dict(key=r['key'],trajectory=tr))
    write(OUT/'PREFIX_EQUIVALENCE.json',dict(rows=result,reference_fits=len(result),exact=True))
    print('F41 exact original prefix equivalence',len(result),flush=True)


def search():
    import numpy as np
    import optuna
    from scripts.analyze_spatial10_components_v1 import labels_for,checked_score
    from vg_tta.dense_support_prefix_v1 import selected_fit
    p=plan(True);assert not (OUT/'SELECTION.json').exists()
    assert read(OUT/'PREFIX_EQUIVALENCE.json')['exact']
    optuna.logging.set_verbosity(optuna.logging.WARNING);t0=time.time();counts=dict(trajectories=0,backwards=0)
    for c,rr in p['rows'].items():
        rows=[r for r in rr if r['f41_role']=='development'];assert len(rows)==32
        labels=labels_for(rows);caches={r['key']:cache_for(r) for r in rows}
        native={r['key']:checked_score(caches[r['key']]['T0']['boxes'],labels[r['key']],caches[r['key']]['frame_ids'],caches[r['key']]['T0']['indices'])[0] for r in rows}
        pairs={s['id']:s for s in p['menu']};ids=list(pairs)
        study=optuna.create_study(direction='maximize',study_name=c,storage='sqlite:///'+str(OUT/'optuna.sqlite3'),
            sampler=optuna.samplers.GridSampler({'pair_id':ids},seed=p['search']['seeds'][c]),load_if_exists=True)
        def objective(trial):
            pair_id=trial.suggest_categorical('pair_id',ids);spec=pairs[pair_id];dest=OUT/'search'/c/(pair_id+'.json')
            if dest.exists():return read(dest)['best_prefix_mean_delta_v']
            per_budget={str(n):[] for n in spec['budgets']}
            for r in rows:
                key=r['key'];cache=caches[key];path=OUT/'search_fits'/c/pair_id/keyfile(key)
                if recorded(path):tr=load(path)['trajectory']
                else:
                    assert counts['trajectories']<p['caps']['actual_trajectories']
                    assert counts['backwards']+spec['maximum']<=p['caps']['actual_backwards']
                    assert time.time()-t0<p['caps']['wall_seconds']
                    tr=run_trajectory(cache,spec);record(path,dict(key=key,spec=spec,trajectory=tr,GT_online=False))
                counts['trajectories']+=1;counts['backwards']+=tr['backwards']
                for n in spec['budgets']:
                    z=selected_fit(tr,n,cache['T0']['boxes'])
                    m,_=checked_score(z['final']['boxes'],labels[key],cache['frame_ids'],z['final']['indices'])
                    assert m['sIoU']==native[key]['sIoU']
                    per_budget[str(n)].append(dict(key=key,group=r['group'],fold=r['f41_fold'],
                        delta_v=m['vIoU_corrected']-native[key]['vIoU_corrected'],metrics=m,T0=native[key],
                        best_step=z['best_step'],state_delta=z['state_delta'],indices=z['final']['indices'],path=str(path)))
            bybudget={n:dict(config=dict(lr=spec['lr'],beta=spec['beta'],steps=int(n)),rows=vals,
                delta_v=float(np.mean([v['delta_v'] for v in vals]))) for n,vals in per_budget.items()}
            val=max(v['delta_v'] for v in bybudget.values())
            write(dest,dict(spec=spec,budgets=bybudget,best_prefix_mean_delta_v=val,GT_use='development only',evaluation_read=False))
            print('F41 HPO',c,len(list((OUT/'search'/c).glob('*.json'))),'/',len(pairs),
                  'lr',spec['lr'],'beta',spec['beta'],'best_dev_pp',round(100*val,4),'counts',counts,flush=True)
            status(OUT/'STATUS.json',dict(stage='search',cohort=c,pairs_completed=len(list((OUT/'search'/c).glob('*.json'))),pairs_total=len(pairs),counts=counts,finished=False))
            return val
        complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,len(ids)-complete))
        assert len(list((OUT/'search'/c).glob('*.json')))==len(ids)
    write(OUT/'SEARCH_COMPLETED.json',dict(created=time.time(),counts=counts,seconds=time.time()-t0))


def pick(candidates,rule,indices=None):
    import numpy as np
    scores=[]
    for c in candidates:
        a=np.array([r['delta_v'] for r in c['rows'] if indices is None or r['key'] in indices],float)
        assert len(a)>1
        value=float(a.mean())-(float(a.std(ddof=1)/math.sqrt(len(a))) if rule=='mean_minus_one_se' else 0.)
        scores.append((value,c))
    best=max(v for v,c in scores);tied=[c for v,c in scores if best-v<=1e-12]
    return min(tied,key=lambda c:(c['config']['steps'],abs(math.log10(c['config']['lr']/.01)),c['config']['beta']))


def choose():
    import numpy as np
    p=plan(True);assert not (OUT/'SELECTION.json').exists();by={}
    for c,rr in p['rows'].items():
        rows=[r for r in rr if r['f41_role']=='development']
        candidates=[v for f in sorted((OUT/'search'/c).glob('*.json')) for v in read(f)['budgets'].values()]
        assert len(candidates)==p['search']['configs_per_direction']
        cv={};selected={}
        for rule in p['selection']['rules']:
            folds=[];values=[]
            for f in range(4):
                train={r['key'] for r in rows if r['f41_fold']!=f};held={r['key'] for r in rows if r['f41_fold']==f}
                z=pick(candidates,rule,train);hold=[r for r in z['rows'] if r['key'] in held]
                assert len(train)==24 and len(hold)==8
                vv=[r['delta_v'] for r in hold];values.extend(vv)
                folds.append(dict(fold=f,config=z['config'],held_keys=sorted(held),held_delta_v=float(np.mean(vv)),held_rows=hold))
            cv[rule]=dict(folds=folds,mean=float(np.mean(values)))
            z=pick(candidates,rule);selected[rule]=dict(config=z['config'],development_delta_v=z['delta_v'],rows=z['rows'])
        winner=max(cv,key=lambda rule:(round(cv[rule]['mean'],12),rule=='mean_minus_one_se'))
        by[c]=dict(chosen=selected[winner]['config'],rule=winner,development_delta_v=selected[winner]['development_delta_v'],
            cv=cv,candidates_by_rule=selected,candidate_configs=len(candidates),evaluation_used=False)
        print('F41 LOCK SELECT',c,winner,by[c]['chosen'],'dev_pp',100*by[c]['development_delta_v'],
              'fold_scores',{k:100*v['mean'] for k,v in cv.items()},flush=True)
    write(OUT/'SELECTION.json',dict(created=time.time(),by_cohort=by,lock_sha256=sha(OUT/'LOCK.json'),evaluation_used=False,
        selection_scope='32 development sources each; fold diagnostics and meta-rule selection are not independent test evidence'))
    status(OUT/'STATUS.json',dict(stage='selection_locked',finished=False,chosen={c:v['chosen'] for c,v in by.items()}))


def evaluate():
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.dense_support_prefix_v1 import selected_fit
    from vg_tta.foreground_runtime import state_digest
    p=plan(True);sel=read(OUT/'SELECTION.json');oldsel=read(F40/'SELECTION.json')
    for c,rr in p['rows'].items():
        pending=[r for r in rr if r['f41_role']=='evaluation' and not recorded(OUT/'eval'/keyfile(r['key']))]
        if not pending:continue
        model=model_for(c);digest=state_digest(model)
        cfg=sel['by_cohort'][c]['chosen']
        configs={'T1':cfg,'Tmean':sel['by_cohort'][c]['candidates_by_rule']['mean']['config'],
            'Tstable':sel['by_cohort'][c]['candidates_by_rule']['mean_minus_one_se']['config'],
            'T40':oldsel['by_cohort'][c]['chosen'],'T2':cfg,'T3':cfg,'T4':cfg}
        for r in pending:
            cache=cache_for(r);old,a4=a4_for(r);fits={};trajectories={};reuse={};audits={}
            frames,_,records,views,qa,_=capture_timed(model,r);assert qa==cache['qa'];del views
            audits['T0']=full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],a4['state'],cache['T0'])['audit']
            for name,cc in configs.items():
                arm=name if name in ('T2','T3','T4') else 'T1'
                signature=(arm,cc['lr'],cc['beta'],cc['steps'])
                if signature in reuse:
                    previous=reuse[signature];fits[name]=fits[previous];continue
                tr=run_trajectory(cache,dict(lr=cc['lr'],beta=cc['beta'],budgets=[cc['steps']]),arm)
                z=selected_fit(tr,cc['steps'],cache['T0']['boxes']);fits[name]=z;trajectories[name]=tr;reuse[signature]=name
                if arm!='T2':
                    audits[name]=full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],
                        {**a4['state'],**z['state']},z['final'])['audit']
            record(OUT/'eval'/keyfile(r['key']),dict(key=r['key'],cohort=c,group=r['group'],source=r['source'],
                panel=r['f41_panel'],T0=cache['T0'],frame_ids=cache['frame_ids'],fits=fits,trajectories=trajectories,
                configs=configs,audits=audits,selection_sha256=sha(OUT/'SELECTION.json'),GT_online=False,new_DINO=0))
            assert state_digest(model)==digest
            print('F41 eval',r['key'],{n:z['final']['physical_interval'] for n,z in fits.items()},flush=True)
            status(OUT/'STATUS.json',dict(stage='evaluation',last=r['key'],finished=False))
        del model;gc.collect();torch.cuda.empty_cache()


def summarize():
    from scripts.analyze_spatial10_components_v1 import labels_for,checked_score,summary
    from scripts.analyze_dense_support_v1 import behavior
    p=plan(True);sel=read(OUT/'SELECTION.json');rows=[r for rr in p['rows'].values() for r in rr if r['f41_role']=='evaluation']
    gt=labels_for(rows);results=[];names=p['selection']['controls']
    for r in rows:
        path=OUT/'eval'/keyfile(r['key']);assert recorded(path);x=load(path);cache=cache_for(r)
        native=x['T0'];ids=x['frame_ids'];native_interval=[ids[native['indices'][0]],ids[native['indices'][1]]+1]
        arms={'T0':checked_score(native['boxes'],gt[r['key']],ids,native['indices'])[0]};details={}
        for n,z in x['fits'].items():
            arms[n],_=checked_score(z['final']['boxes'],gt[r['key']],ids,z['final']['indices'])
            assert arms[n]['sIoU']==arms['T0']['sIoU']
            details[n]=dict(interval=z['final']['physical_interval'],state_delta=z['state_delta'],best_step=z['best_step'],
                loss=z['final']['loss'],behavior=behavior(native_interval,z['final']['physical_interval']))
        results.append(dict(key=r['key'],cohort=x['cohort'],group=r['group'],source=r['source'],panel=r['f41_panel'],
            query=r['input']['caption'],arms=arms,details=details,native_interval=native_interval,
            GT_interval=gt[r['key']].get('temporal_interval'),path=str(path),sha256=sha(path)))
    sums={}
    for c in p['rows']:
        rr=[r for r in results if r['cohort']==c];sums[c]={}
        for panel,ss in [('all',rr)]+[(n,[r for r in rr if r['panel']==n]) for n in ('F40_repeated_evaluation','additional_locked_evaluation')]:
            sums[c][panel]=summary(ss,names,METRICS,[(n,'T0') for n in names if n!='T0']+[('T1',n) for n in ('T40','T2','T3','T4')])
    write(OUT/'EVAL_RESULTS.json',dict(rows=results,summary=sums,selection=sel,historical_exposure=True,untouched=False))
    status(OUT/'STATUS.json',dict(stage='evaluated_pending_audit',finished=False))
    print(json.dumps({c:{pa:{n:100*v['vIoU_corrected']['mean'] for n,v in s['arms'].items()} for pa,s in panels.items()} for c,panels in sums.items()},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture-dev','equivalence','search','choose','capture-eval','evaluate','summarize']);a=ap.parse_args()
    if a.stage=='prepare':prepare();return
    if a.stage=='choose':choose();return
    if a.stage=='summarize':summarize();return
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.stage=='capture-dev':capture('development')
        elif a.stage=='capture-eval':capture('evaluation')
        else:globals()[a.stage]()
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
