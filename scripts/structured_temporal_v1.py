"""Bounded F43 execution. Labels are confined to development selection/scoring.

prepare -> development -> select -> validation -> full-replay.
No scheduler, production promotion, new A4 fit or expert call.
"""
import argparse
import copy
import fcntl
import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts.dense_support_tuning_v1 import keyfile,config_id,METRICS
from scripts.dense_support_generalization_v1 import cache_for,a4_for

OUT=ROOT/'artifacts/decota_structured_temporal_v1'
F41=ROOT/'artifacts/decota_dense_support_generalization_v1'
F42=ROOT/'artifacts/decota_dense_support_validation_selection_v1'
OWN=['vg_tta/structured_temporal_v1.py','scripts/structured_temporal_v1.py',
     'tests/test_structured_temporal_v1.py']
LRS=[.0001,.001,.01,.1]
BUDGETS=[1,5,10,30]


def prepare():
    from vg_tta.structured_temporal_v1 import DEFAULT
    p=read(F41/'LOCK.json');seal=read(F41/'COMPLETION.json')
    protected={**p['protected_pins'],**seal['files'],**seal['code_pins']}
    protected[str((F41/'COMPLETION.json').relative_to(ROOT))]=sha(F41/'COMPLETION.json')
    for f,h in protected.items():assert sha(ROOT/f)==h,f
    # SIGINT was sent to the exact previously inspected F42 worker. Preserve all
    # committed scores and unfinished fit artifacts, but do not call it complete.
    progress=read(F42/'STATUS.json');counts={}
    for cohort in p['rows']:
        files=sorted((F42/'scores'/cohort).glob('*.json'))
        counts[cohort]=dict(completed_candidates=len(files),planned_candidates=130,
                            score_pins={str(f.relative_to(ROOT)):sha(f) for f in files})
    pause=dict(status='paused_not_completed',reason='Latest user prioritizes new structured temporal mechanism.',
               worker_pid=3231541,signal='SIGINT',created=time.time(),last_progress=progress,cohorts=counts,
               preserve_partial_fits=True,independent_test=False)
    if not (F42/'PAUSE_F43.json').exists():write(F42/'PAUSE_F43.json',pause)
    status(F42/'STATUS.json',dict(stage='paused_for_F43',finished=False,paused=True,pause_receipt=str(F42/'PAUSE_F43.json')))
    rows=copy.deepcopy(p['rows'])
    for rr in rows.values():
        for r in rr:r['f43_role']='development' if r['f41_role']=='development' else 'second_validation'
    lock=dict(name='F43_Structured_Reliable_Temporal_Adaptation',created=time.time(),rows=rows,
        protected_pins=protected,code_pins={f:sha(ROOT/f) for f in OWN},
        attachment=dict(path='./private_authorization_notes/authorization.txt'),
        fixed=DEFAULT,default=dict(lr=.001,steps=5),spatial=p['spatial'],
        backbone='TA-STVG existing VidSTG -> HC1 and HC2 -> VidSTG source checkpoints',
        trainable='66306 temp_embed; A4 1792 actual states reused without fitting',
        teacher='same frozen dense final pred_actioness raw logits, complete valid offset grids',
        selection=dict(menu=[dict(lr=x,steps=n) for x in LRS for n in BUDGETS],
            development_sources_per_direction=32,criterion='max mean development source delta_v separately for B and C per target',
            ties='within1e-12: default config, then fewer steps, then log distance to .001',
            evaluation='HC25 and Vid24 former evaluation, now second validation after F42 exposure; do not reselect F43 on these results',
            matched='B_match uses C development-selected lr/budget to isolate the C package; B_tuned has its own same-budget search',
            no_source_exclusion=True,no_independent_test=True),
        arms=['T0','F39','F39_matched','A_raw','A','B_default','C_default','B_tuned','B_match','C',
              'R_loss','R_guard_loss','R_decision','B_no_margin','wrong_query','time_shift'],
        semantics=dict(interval='legal s<e, inclusive sampled end; physical [f_s,f_e+1)',
            calibration='unweighted median and MAD; even median averages middle two; not guaranteed semantic calibration',
            BCE='-sum omega [inside*log(a)+(1-inside)*log(1-a)]',
            T_A='direct argmax projected interval, no temporal network update',
            C='rho weighted NLL+margin, native guard, best actual decoded G; earliest-state ties',
            guard='output-state selection only; reject loss-increasing optimizer proposals with full Adam restoration',
            reliability='offset IoU times sigmoids of the two projected top-two margins; not a correctness probability'),
        caps=dict(trajectories=1800,backwards=45000,new_DINO=0,new_space_fits=0,full_model_replays=250,wall_seconds=7200),
        historical_exposure=True,GT_online=False,registries_change=False,automation=False,corruption=False)
    lock['attachment']['sha256']=sha(lock['attachment']['path'])
    write(OUT/'LOCK.json',lock)
    write(OUT/'PARENT_AUDIT.json',dict(F41_verified_files=len(protected),F41_audit=read(F41/'AUDIT.json'),
          F41_seal_sha256=sha(F41/'COMPLETION.json'),F42_pause=str(F42/'PAUSE_F43.json')))
    status(OUT/'STATUS.json',dict(stage='prepared',finished=False))
    print('F43 prepared',len(protected),'protected files',[(c,len(rr)) for c,rr in rows.items()],flush=True)


def plan():
    p=read(OUT/'LOCK.json')
    runtime=dict(p['code_pins'])
    amendment=OUT/'METADATA_COMPATIBILITY_FIX.json'
    if amendment.exists():
        fix=read(amendment)
        assert fix['original_code_pins']==p['code_pins']
        runtime.update(fix['runtime_code_pins'])
    for f,h in runtime.items():assert sha(ROOT/f)==h,f
    return p


def record(path,x):
    save(path,x);write(path.with_suffix('.json'),dict(sha256=sha(path),created=time.time()))


def recorded(path):return path.exists() and path.with_suffix('.json').exists() and sha(path)==read(path.with_suffix('.json'))['sha256']


def path_for(r,lr,kind):return OUT/'fits'/r['f43_role']/kind/config_id(dict(lr=lr))/keyfile(r['key'])


def ensure_fit(r,cache,lr,kind,budgets):
    from vg_tta.structured_temporal_v1 import fit
    from vg_tta.dense_support_tuning_v1 import HeadReplay,move
    path=path_for(r,lr,kind)
    if recorded(path):
        x=load(path);assert set(budgets)<=set(x['budgets']);return path,x
    it=HeadReplay(cache)
    teacher=it.teacher
    if kind=='wrong':teacher=move(cache['wrong_teacher'],'cuda')
    if kind=='shift':teacher=move(cache['rolled_teacher'],'cuda')
    x=fit(it,cache['records'],cache['frame_ids'],teacher,lr=lr,budgets=budgets,
          reliability=kind in ['R','wrong','shift'],gamma=0. if kind=='NLL' else None)
    x.update(key=r['key'],kind=kind,role=r['f43_role'])
    record(path,x);return path,x


def metrics_for(r,cache,result,gt):
    from scripts.analyze_spatial10_components_v1 import checked_score
    m,_=checked_score(cache['T0']['boxes'],gt,cache['frame_ids'],result['indices'])
    return m


def development():
    from scripts.analyze_spatial10_components_v1 import labels_for
    from vg_tta.structured_temporal_v1 import selected_fit
    p=plan();started=time.time()
    for cohort,rr in p['rows'].items():
        rows=[r for r in rr if r['f43_role']=='development'];labels=labels_for(rows)
        scores={}
        for lr in LRS:
            for kind in ['B','R']:
                for i,r in enumerate(rows):
                    cache=cache_for(r);path,tr=ensure_fit(r,cache,lr,kind,BUDGETS)
                    base=metrics_for(r,cache,cache['T0'],labels[r['key']])
                    for budget in BUDGETS:
                        selector='loss' if kind=='B' else 'guard_decision'
                        s=selected_fit(tr,budget,cache['T0']['boxes'],selector)
                        m=metrics_for(r,cache,s['final'],labels[r['key']])
                        cid=config_id(dict(lr=lr,steps=budget,kind=kind))
                        scores.setdefault(cid,dict(config=dict(lr=lr,steps=budget),kind=kind,rows=[]))['rows'].append(
                            dict(key=r['key'],group=r['group'],delta_v=m['vIoU_corrected']-base['vIoU_corrected'],
                                 metrics=m,T0=base,parameter_changed=s['state_delta']>0,
                                 interval_changed=s['final']['indices']!=cache['T0']['indices'],fit=str(path),selector=selector))
                    status(OUT/'STATUS.json',dict(stage='development',cohort=cohort,lr=lr,kind=kind,completed=i+1,total=32,finished=False))
                print('F43 dev',cohort,lr,kind,flush=True)
        for x in scores.values():x['mean_delta_v']=sum(r['delta_v'] for r in x['rows'])/len(x['rows'])
        write(OUT/'development'/f'{cohort}.json',list(scores.values()))
    write(OUT/'DEVELOPMENT_COMPLETED.json',dict(created=time.time(),seconds=time.time()-started,trajectories=512,backwards=15360))


def select():
    p=plan();out={}
    for cohort in p['rows']:
        scores=read(OUT/'development'/f'{cohort}.json');out[cohort]={}
        for kind in ['B','R']:
            rr=[r for r in scores if r['kind']==kind];v=max(r['mean_delta_v'] for r in rr)
            ties=[r for r in rr if v-r['mean_delta_v']<=1e-12]
            import math
            best=min(ties,key=lambda r:(r['config']!=p['default'],r['config']['steps'],abs(math.log10(r['config']['lr']/.001))))
            out[cohort][kind]=dict(config=best['config'],development_delta_v=best['mean_delta_v'])
    write(OUT/'SELECTION.json',dict(created=time.time(),choices=out,data_role='32 development sources per direction only',
          historical_exposure=True,second_validation_not_used_for_F43_selection=True))
    print(json.dumps(out,indent=2),flush=True)


def inspect_arm(path,tr,budget,selector,cache):
    from vg_tta.structured_temporal_v1 import selected_fit
    s=selected_fit(tr,budget,cache['T0']['boxes'],selector)
    return s['final'],dict(path=str(path),budget=budget,selector=selector,state_delta=s['state_delta'],
            best_step=s['best_step'],parameter_TTA=True,kind=tr['kind'])


def validation():
    from scripts.analyze_spatial10_components_v1 import labels_for
    from vg_tta.structured_temporal_v1 import project,projected_indices
    from vg_tta.dense_support_tuning_v1 import HeadReplay
    from scripts.dense_support_tuning_v1 import run_fit
    p=plan();selection=read(OUT/'SELECTION.json')['choices'];outputs=[]
    # One source at a time, no cross-source states; also run fixed default diagnostics on development.
    for cohort,rr in p['rows'].items():
        labels=labels_for(rr);chosen=selection[cohort]
        for r in rr:
            dest=OUT/'results'/keyfile(r['key']).replace('.pt','.json')
            if dest.exists():outputs.append(read(dest));continue
            cache=cache_for(r);gt=labels[r['key']];it=HeadReplay(cache)
            values={'T0':cache['T0']};refs={};evidence={}
            for name,cal in [('A_raw',False),('A',True)]:
                ev=project(it.zero['logits'],it.teacher,cache['records'],calibrated=cal)
                values[name]=dict(indices=projected_indices(ev,cache['frame_ids']))
                evidence[name]=dict(rho=ev['rho'],offset_iou=ev['offset_iou'],
                    offsets=[{k:t[k] for k in ['interval','native_interval','native_drop','projection_margin','median','mad','mad_zero','target_indices']}
                             for t in ev['offsets']])
            tasks={}
            def add(name,kind,lr,budget,selector):
                tasks.setdefault((kind,lr),[]).append((name,budget,selector))
            add('B_default','B',.001,5,'loss');add('C_default','R',.001,5,'guard_decision')
            bc,cc=chosen['B']['config'],chosen['R']['config']
            add('B_tuned','B',bc['lr'],bc['steps'],'loss')
            add('B_match','B',cc['lr'],cc['steps'],'loss')
            for name,selector in [('C','guard_decision'),('R_loss','loss'),('R_guard_loss','guard_loss'),('R_decision','decision')]:
                add(name,'R',cc['lr'],cc['steps'],selector)
            add('B_no_margin','NLL',.001,5,'loss')
            add('wrong_query','wrong',cc['lr'],cc['steps'],'guard_decision')
            add('time_shift','shift',cc['lr'],cc['steps'],'guard_decision')
            for (kind,lr),jobs in tasks.items():
                budgets=BUDGETS if r['f43_role']=='development' and kind in ['B','R'] else sorted(set(j[1] for j in jobs))
                path,tr=ensure_fit(r,cache,lr,kind,budgets)
                for name,budget,selector in jobs:values[name],refs[name]=inspect_arm(path,tr,budget,selector,cache)
            for name,cfg in [('F39',dict(lr=.01,beta=1.,steps=5)),('F39_matched',dict(lr=.001,beta=.1,steps=5))]:
                path=OUT/'old_objective'/name/keyfile(r['key'])
                if recorded(path):old=load(path)
                else:old=run_fit(cache,cfg);record(path,old)
                values[name]=old['final'];refs[name]=dict(path=str(path),old=True,state_delta=old['state_delta'],best_step=old['best_step'])
            arms={n:metrics_for(r,cache,v,gt) for n,v in values.items()}
            assert len({m['sIoU'] for m in arms.values()})==1
            x=dict(key=r['key'],cohort=cohort,group=r['group'],source=r['source'],role=r['f43_role'],
                previous_panel=r['f41_panel'],caption=r['input']['caption'],GT_interval=gt['interval'],
                frame_ids=cache['frame_ids'],native_indices=cache['T0']['indices'],arms=arms,fit_references=refs,
                intervals={n:[cache['frame_ids'][v['indices'][0]],cache['frame_ids'][v['indices'][1]]+1] for n,v in values.items()},
                projection=evidence,GT_online=False,spatial_unchanged=True,
                wrong_query=cache['wrong_query'] if 'wrong_query' in cache else r['F39_wrong_query'],
                time_shift=[t.get('roll') for t in cache['rolled_teacher']])
            write(dest,x);outputs.append(x)
            status(OUT/'STATUS.json',dict(stage='all_arms',completed=len(outputs),total=113,last=r['key'],finished=False))
            print('F43 arms',r['key'],flush=True)
    write(OUT/'ALL_SOURCE_RESULTS.json',dict(rows=outputs,selection=selection,independent_test=False))


def full_replay():
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.structured_temporal_v1 import selected_fit,fit
    from vg_tta.dense_support_tuning_v1 import HeadReplay
    p=plan();data={r['key']:r for r in read(OUT/'ALL_SOURCE_RESULTS.json')['rows']}
    selected=['C','B_tuned','wrong_query','time_shift']
    for cohort,rr in p['rows'].items():
        model=model_for(cohort);digest=state_digest(model)
        for r in [x for x in rr if x['f43_role']=='second_validation']:
            dest=OUT/'full_replay'/keyfile(r['key']).replace('.pt','.json')
            if dest.exists():continue
            cache=cache_for(r);_,a4=a4_for(r)
            frames,_,records,views,qa,_=capture_timed(model,r);assert qa==cache['qa']
            audits={}
            for name in selected:
                ref=data[r['key']]['fit_references'][name];tr=load(ref['path'])
                s=selected_fit(tr,ref['budget'],cache['T0']['boxes'],ref['selector'])
                audits[name]=full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],
                               {**a4['state'],**s['state']},s['final'])['audit']
            assert state_digest(model)==digest
            write(dest,dict(key=r['key'],audits=audits,source_restored=True))
            del frames,views
            print('F43 real model',r['key'],flush=True)
        # Real-cache zero-LR backward and zero-step controls, no metric selection.
        r=rr[0];cache=cache_for(r);noops=[]
        for lr,budget in [(0.,5),(.001,0)]:
            it=HeadReplay(cache);tr=fit(it,cache['records'],cache['frame_ids'],it.teacher,lr=lr,budgets=[budget],reliability=True)
            s=selected_fit(tr,budget,cache['T0']['boxes'])
            assert s['state_delta']==0 and s['final']['indices']==cache['T0']['indices']
            noops.append(dict(lr=lr,budget=budget,backwards=tr['backwards'],state_delta=0,exact=True))
            record(OUT/'noops'/cohort/f'lr{lr}_steps{budget}.pt',tr)
        write(OUT/'noops'/f'{cohort}.json',noops)
        del model;gc.collect();torch.cuda.empty_cache()
    status(OUT/'STATUS.json',dict(stage='models_complete_pending_audit',finished=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','development','select','validation','full-replay']);a=ap.parse_args()
    if a.stage=='prepare':prepare();return
    if a.stage=='select':select();return
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:globals()[a.stage.replace('-','_')]()
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
