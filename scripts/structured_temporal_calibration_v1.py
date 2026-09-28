"""F44 bounded 3x3 calibration, then P0/P1/P2 and frozen-signal controls."""
import argparse
import copy
import fcntl
import gc
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.dense_support_tuning_v1 import keyfile,config_id,METRICS
from scripts.dense_support_generalization_v1 import cache_for,a4_for
from scripts.structured_temporal_v1 import record,recorded

OUT=ROOT/'artifacts/decota_structured_calibration_v1'
PARENT=ROOT/'artifacts/decota_structured_temporal_v1'
OWN=['vg_tta/structured_temporal_calibration_v1.py',
     'scripts/structured_temporal_calibration_v1.py','tests/test_structured_temporal_calibration_v1.py']


def prepare():
    from vg_tta.structured_temporal_calibration_v1 import FIXED,MENU
    parent=read(PARENT/'LOCK.json');seal=read(PARENT/'COMPLETION.json')
    protected={**parent['protected_pins'],**seal['files'],**seal['code_pins'],
               str((PARENT/'COMPLETION.json').relative_to(ROOT)):sha(PARENT/'COMPLETION.json')}
    for path,h in protected.items(): assert sha(ROOT/path)==h,path
    assert read(PARENT/'AUDIT.json')['status']=='pass'
    rows=copy.deepcopy(parent['rows'])
    for rr in rows.values():
        for r in rr:r['f44_role']=r['f43_role']
    old=read(PARENT/'SELECTION.json')['choices']
    numerical={c:old[c]['R']['config'] for c in rows}
    assert numerical=={'hcstvg1_test':{'lr':.001,'steps':5},'vidstg_test':{'lr':.1,'steps':5}}
    attachment='./private_authorization_notes/authorization.txt'
    lock=dict(name='F44_R53_Structured_Temporal_Interface_Calibration',created=time.time(),
        attachment=dict(path=attachment,sha256=sha(attachment)),rows=rows,
        protected_pins=protected,code_pins={f:sha(ROOT/f) for f in OWN},
        fixed=FIXED,menu=MENU,numerical=numerical,spatial=parent['spatial'],
        source_checkpoints=parent['backbone'],historical_exposure=True,GT_online=False,
        selection=dict(criterion='maximum source mean P1 delta_v on 32 development sources per target',
            ties='within1e-12 prefer alpha1/lambda1; then larger lambda, then larger alpha',
            second_validation='25 HC +24 Vid; all nine prespecified sensitivity cells measured after config lock, never reselect',
            P0_P2='same chosen teacher, no independent hyperparameter search; same optimizer settings for P1/P2',
            independent_test=False),
        closed=['new spatial experiments','reliability weighting','decision best state','native trust guard',
                'ASA attributes','crop','warp','coverage','local boundary','shared TTA','EMA','SAM',
                'entropy','memory','temporal augmentation','teacher ensemble','threshold gates'],
        controls=['T0','P0_direct_projection','P1_actual_66306_head','P2_free_logits',
                  'wrong_query','time_shift','F43_C','F43_B_match'],
        caps=dict(fits=1600,backwards=9000,full_model_replays=180,wall_seconds=7200,new_DINO=0,new_space_fits=0),
        no_new_models=True,automation=False,corruption=False,registry_changes=False,
        science='locked design choices are not universal necessity; no independent test or SOTA claim')
    write(OUT/'LOCK.json',lock)
    write(OUT/'PARENT_AUDIT.json',dict(status='pass',verified_files=len(protected),
         parent_audit=read(PARENT/'AUDIT.json'),parent_seal=sha(PARENT/'COMPLETION.json')))
    status(OUT/'STATUS.json',dict(stage='prepared',finished=False))
    print('F44 prepared',len(protected),'protected dependencies',flush=True)


def plan():
    p=read(OUT/'LOCK.json')
    pins=dict(p['code_pins'])
    amendment=OUT/'COHORT_ROUTING_FIX.json'
    if amendment.exists():
        fix=read(amendment);assert fix['original_code_pins']==p['code_pins']
        pins.update(fix['runtime_code_pins'])
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    return p


def fit_path(r,cfg,kind):
    return OUT/'fits'/r['f44_role']/kind/config_id(cfg)/keyfile(r['key'])


def ensure_fit(r,cache,cfg,kind,numerical):
    from vg_tta.structured_temporal_calibration_v1 import fit
    from vg_tta.dense_support_tuning_v1 import HeadReplay,move
    from vg_tta.dense_support_temporal_v1 import OutputReplay
    path=fit_path(r,cfg,kind)
    if recorded(path):return path,load(path)
    head=HeadReplay(cache);it=OutputReplay(head.zero) if kind=='P2' else head
    teacher=move(cache['wrong_teacher'],'cuda') if kind=='wrong_query' else (
            move(cache['rolled_teacher'],'cuda') if kind=='time_shift' else head.teacher)
    x=fit(it,cache['records'],cache['frame_ids'],teacher,cfg,
          **numerical,output_control=kind=='P2')
    x.update(key=r['key'],kind=kind,role=r['f44_role'])
    record(path,x);return path,x


def score(cache,gt,indices):
    from scripts.analyze_spatial10_components_v1 import checked_score
    return checked_score(cache['T0']['boxes'],gt,cache['frame_ids'],indices)[0]


def grid_for_row(r,cache,gt,p,cohort):
    from vg_tta.structured_temporal_calibration_v1 import projected_indices
    cells={};baseline=score(cache,gt,cache['T0']['indices'])
    for cfg in p['menu']:
        path,tr=ensure_fit(r,cache,cfg,'P1',p['numerical'][cohort])
        direct=projected_indices(tr['evidence'],cache['frame_ids'])
        projected=score(cache,gt,direct);param=score(cache,gt,tr['final']['indices'])
        cells[config_id(cfg)]=dict(config=cfg,key=r['key'],group=r['group'],fit=str(path),
            T0=baseline,P0=projected,P1=param,delta_v=param['vIoU_corrected']-baseline['vIoU_corrected'],
            P0_interval=[cache['frame_ids'][direct[0]],cache['frame_ids'][direct[1]]+1],
            P1_interval=tr['final']['physical_interval'],state_delta=tr['state_delta'],best_step=tr['best_step'])
    return cells


def development():
    from scripts.analyze_spatial10_components_v1 import labels_for
    p=plan();started=time.time()
    for cohort,rr in p['rows'].items():
        rows=[r for r in rr if r['f44_role']=='development'];gt=labels_for(rows);scores={}
        for i,r in enumerate(rows):
            cache=cache_for(r);cells=grid_for_row(r,cache,gt[r['key']],p,cohort)
            write(OUT/'grid'/keyfile(r['key']).replace('.pt','.json'),dict(key=r['key'],role='development',cells=cells))
            for cid,cell in cells.items():scores.setdefault(cid,dict(config=cell['config'],rows=[]))['rows'].append(cell)
            print('F44 grid development',cohort,i+1,len(rows),r['key'],flush=True)
            status(OUT/'STATUS.json',dict(stage='development',cohort=cohort,completed=i+1,total=32,finished=False))
        for item in scores.values():item['mean_delta_v']=sum(r['delta_v'] for r in item['rows'])/32
        write(OUT/'development'/f'{cohort}.json',scores)
    write(OUT/'DEVELOPMENT_COMPLETED.json',dict(created=time.time(),seconds=time.time()-started,sources=64,fits=576))


def choose(scores):
    best=max(x['mean_delta_v'] for x in scores.values())
    ties=[x for x in scores.values() if best-x['mean_delta_v']<=1e-12]
    return min(ties,key=lambda x:(x['config']!={'center_fraction':1.,'prior_weight':1.},
                 -x['config']['prior_weight'],-x['config']['center_fraction']))


def select():
    p=plan();selection={}
    assert (OUT/'DEVELOPMENT_COMPLETED.json').exists()
    for c in p['rows']:
        data=read(OUT/'development'/f'{c}.json');best=choose(data)
        selection[c]=dict(config=best['config'],numerical=p['numerical'][c],development_delta_v=best['mean_delta_v'])
    write(OUT/'SELECTION.json',dict(created=time.time(),choices=selection,role='development only',
        historical_exposure=True,not_independent_test=True,do_not_reselect=True))
    print(selection,flush=True)


def validation():
    from scripts.analyze_spatial10_components_v1 import labels_for
    from vg_tta.structured_temporal_calibration_v1 import projected_indices
    p=plan();choices=read(OUT/'SELECTION.json')['choices'];outputs=[]
    old={r['key']:r for r in read(PARENT/'ALL_SOURCE_RESULTS.json')['rows']}
    for cohort,rr in p['rows'].items():
        labels=labels_for(rr);cfg=choices[cohort]['config']
        for r in rr:
            dest=OUT/'results'/keyfile(r['key']).replace('.pt','.json')
            if dest.exists():outputs.append(read(dest));continue
            cache=cache_for(r);gt=labels[r['key']]
            gridfile=OUT/'grid'/keyfile(r['key']).replace('.pt','.json')
            cells=read(gridfile)['cells'] if gridfile.exists() else grid_for_row(r,cache,gt,p,cohort)
            if not gridfile.exists():write(gridfile,dict(key=r['key'],role=r['f44_role'],cells=cells))
            refs={};values={'T0':cache['T0']};evs={}
            for name in ['P1','P2','wrong_query','time_shift']:
                path,tr=ensure_fit(r,cache,cfg,name,p['numerical'][cohort]);values[name]=tr['final']
                refs[name]=dict(path=str(path),best_step=tr['best_step'],state_delta=tr['state_delta'],
                    parameter_TTA=tr['parameter_TTA'],parameters=tr['parameters'])
                evs[name]=[{k:t[k] for k in ['median','mad','mad_zero','center_fraction','interval','native_interval']}
                           for t in tr['evidence']['offsets']]
                if name=='P1':values['P0']=dict(indices=projected_indices(tr['evidence'],cache['frame_ids']))
            basecell=cells[config_id(dict(center_fraction=1.,prior_weight=1.))]
            oldr=old[r['key']]
            # Important integration equivalence: after removing rho/guard/decision,
            # the original alpha=1/lambda=1 is B_match, NOT F43's full C.
            for m in METRICS:
                assert basecell['P1'][m]==oldr['arms']['B_match'][m],(r['key'],m)
            arms={n:score(cache,gt,v['indices']) for n,v in values.items()}
            intervals={n:[cache['frame_ids'][v['indices'][0]],cache['frame_ids'][v['indices'][1]]+1] for n,v in values.items()}
            for n,oldn in [('F43_C','C'),('F43_B_match','B_match')]:
                arms[n]=oldr['arms'][oldn];intervals[n]=oldr['intervals'][oldn]
            assert len({m['sIoU'] for m in arms.values()})==1
            x=dict(key=r['key'],cohort=cohort,source=r['source'],group=r['group'],role=r['f44_role'],
                caption=r['input']['caption'],GT_interval=gt['interval'],frame_ids=cache['frame_ids'],
                arms=arms,intervals=intervals,fit_references=refs,projection=evs,grid_reference=str(gridfile),
                selected_config=cfg,numerical=p['numerical'][cohort],GT_online=False,spatial_invariant=True,
                wrong_query=cache.get('wrong_query',r.get('F39_wrong_query')),historical_exposure=True)
            write(dest,x);outputs.append(x)
            print('F44 validation/controls',len(outputs),113,r['key'],flush=True)
            status(OUT/'STATUS.json',dict(stage='grid_and_controls',completed=len(outputs),total=113,finished=False))
    write(OUT/'ALL_SOURCE_RESULTS.json',dict(rows=outputs,selection=choices,independent_test=False))


def full_replay():
    import torch
    from scripts.run_closure_v1 import model_for
    from scripts.run_spatial10_components_v1 import capture_timed
    from scripts.audit_parametric_reinsertion_v1 import full_prediction
    from vg_tta.foreground_runtime import state_digest
    from vg_tta.dense_support_tuning_v1 import HeadReplay
    from vg_tta.dense_support_temporal_v1 import OutputReplay
    from vg_tta.structured_temporal_calibration_v1 import fit
    p=plan();data={r['key']:r for r in read(OUT/'ALL_SOURCE_RESULTS.json')['rows']}
    choices=read(OUT/'SELECTION.json')['choices']
    for cohort,rr in p['rows'].items():
        model=model_for(cohort);digest=state_digest(model)
        for r in [x for x in rr if x['f44_role']=='second_validation']:
            dest=OUT/'full_replay'/keyfile(r['key']).replace('.pt','.json')
            if dest.exists():continue
            cache=cache_for(r);_,a4=a4_for(r)
            frames,_,records,views,qa,_=capture_timed(model,r);assert qa==cache['qa']
            audits={}
            for name in ['P1','wrong_query','time_shift']:
                tr=load(data[r['key']]['fit_references'][name]['path'])
                audits[name]=full_prediction(model,frames,cache['frame_ids'],r['input'],r['subject'],
                    {**a4['state'],**tr['state']},tr['final'])['audit']
            assert state_digest(model)==digest
            write(dest,dict(key=r['key'],audits=audits,source_restored=True))
            del frames,views
            print('F44 real model',r['key'],flush=True)
        r=rr[0];cache=cache_for(r);cfg=choices[cohort]['config'];audits=[]
        for kind in ['P1','P2']:
            for lr,steps in [(0.,5),(p['numerical'][cohort]['lr'],0)]:
                head=HeadReplay(cache);it=OutputReplay(head.zero) if kind=='P2' else head
                tr=fit(it,cache['records'],cache['frame_ids'],head.teacher,cfg,lr=lr,steps=steps,output_control=kind=='P2')
                assert tr['state_delta']==0 and tr['final']['indices']==cache['T0']['indices']
                tr.update(key=r['key'],kind=kind,role='noop')
                record(OUT/'noops'/cohort/f'{kind}_lr{lr}_steps{steps}.pt',tr)
                audits.append(dict(kind=kind,lr=lr,steps=steps,exact=True,backwards=tr['backwards']))
        write(OUT/'noops'/f'{cohort}.json',audits)
        del model;gc.collect();torch.cuda.empty_cache()
    status(OUT/'STATUS.json',dict(stage='models_complete_pending_audit',finished=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','development','select','validation','full-replay']);a=ap.parse_args()
    if a.stage=='prepare':prepare();return
    if a.stage=='select':select();return
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:globals()[a.stage.replace('-','_')]()
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
