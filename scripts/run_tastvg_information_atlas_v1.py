"""Finite CPU characterization of already sealed temporal latent caches."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys,json,time,hashlib,collections,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_information_atlas_math_v1 import *
torch.set_num_threads(2)
BASE=ROOT/'artifacts/tastvg_temporal_information_atlas_v1'
PREV=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
PUB=ROOT/'results/tastvg_temporal_information_atlas/2026-10-03'
DATASETS=['vidstg','hc2']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def status(name,**kwargs):write(BASE/'STATUS.json',dict(status=name,time=time.time(),pid=os.getpid(),**kwargs))
def load(p):
    p=Path(p);r=read(p.with_suffix('.json'));assert sha(p)==r['sha256']
    # Trusted locally generated, SHA-bound arrays; no downloaded pickle files.
    return torch.load(p,map_location='cpu',weights_only=False)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);torch.save(x,p)
    write(p.with_suffix('.json'),dict(sha256=sha(p),time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
def verify():
    r=read(BASE/'RUNTIME_LOCK.json')
    for p,h in {**r['code'],**r['metadata']}.items():assert sha(ROOT/p)==h,p
    assert not torch.cuda.is_initialized();return r
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def prepare():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(PREV/'FINAL_COMPLETION.json')['status']=='completed_verified_publication'
    code=['protocols/tastvg_temporal_information_atlas_v1.md','scripts/tastvg_information_atlas_math_v1.py',
          'scripts/run_tastvg_information_atlas_v1.py','scripts/test_tastvg_information_atlas_v1.py']
    metadata={};payloads={}
    paths=[PREV/'SOURCE_INPUTS.json',PREV/'SOURCE_GT.json',PREV/'COHORT.json',PREV/'FINAL_COMPLETION.json',
           PREV/'RUNTIME_LOCK.json',ROOT/'methods/CURRENT_METHOD.json']
    for ds in DATASETS:
        for sub in ['source_features','target_features']:
            for p in sorted((PREV/ds/sub).glob('*.json')):
                r=read(p);f=p.with_suffix('.pt');assert sha(f)==r['sha256']
                payloads[str(f.relative_to(ROOT))]=r['sha256'];paths.append(p)
        paths.extend([PREV/ds/'SOURCE_FEATURE_BARRIER.json',PREV/ds/'TARGET_FEATURE_BARRIER.json',
            ROOT/f'artifacts/tastvg_current_correction_views_v1/{ds}/PLAN.json',
            POOL/ds/'GT_LABELS_search.json',POOL/ds/'GT_LABELS_confirm.json'])
    for p in paths:metadata[str(p.relative_to(ROOT))]=sha(p)
    config=dict(name='TA-STVG Temporal Latent Information Atlas v1',stage='P0 CPU characterization',
        predecessor_commit='107e1ef9b87bdf5e1b3b07cfe06bf660e3e9284e',alphas=ALPHAS,
        source_splits={'vidstg':[95,31],'hc2':[48,16]},frame_tasks=FRAME_TASKS,candidate_tasks=CANDIDATE_TASKS,
        candidate_views={k:len(v) for k,v in VIEWS.items()},frame_views={'Hidden':256,'Geometry':1},
        hidden_interface='final sixth temporal decoder temp_embed input; cached observed-frame 256D',
        source_checkpoint_sha256=read(ROOT/'results/tastvg_temporal_latent_quality/2026-10-03/CONFIG.json')['checkpoint_state_sha256'],
        source_validation_selects_alpha=True,validation_refit=False,
        target_GT_for_fit=False,target_schema_preflight_span_inspections=4,
        controls='equal-source weighted geometry and within-source training-label shuffle; intercept-only null',
        delta='zero-intercept relative features: source native anchor / target actual A8; also absolute-tIoU prediction differences',
        frame_domain='existing observation clip origin=first fid, width=last fid+1-first fid',
        target_cache_coverage={'vidstg':{'search_sources':16,'confirm_sources':8,'cells':144},
                               'hc2':{'search_sources':14,'confirm_sources':7,'cells':144}},
        bootstrap_draws=10000,seed=20261003,model_calls=0,expert_calls=0,GPU_calls=0,
        no_new_top1_outputs=True,historical_exposure=True,production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'))
    write(PUB/'CONFIG.json',config)
    metadata[str((PUB/'CONFIG.json').relative_to(ROOT))]=sha(PUB/'CONFIG.json')
    write(BASE/'RUNTIME_LOCK.json',dict(code={f:sha(ROOT/f) for f in code},metadata=metadata,payloads=payloads,time=time.time()))
    status('prepared_pending_source_fit',source_queries=190,target_cached_cells=288)
    print('ATLAS_LOCKED',len(payloads),'cached payloads',flush=True)

def source_items(ds):
    rr=read(PREV/'SOURCE_INPUTS.json')[ds];gt=read(PREV/'SOURCE_GT.json')[ds];ans=[]
    for r in rr:
        z=load(PREV/ds/'source_features'/f'{r["index"]:04}.pt')
        assert z['split']==r['split'] and z['frame_ids']==r['frame_ids']
        h=z['hidden'].double().numpy();ids=np.asarray(z['frame_ids']);anchor=0
        frame=frame_labels(ids,gt[str(r['index'])])
        cand=candidate_labels([c['physical_interval'] for c in z['candidates']],gt[str(r['index'])],anchor)
        ans.append(dict(dataset=ds,source_index=r['index'],domain='source',panel=r['split'],
            order='source',condition='clean',cell=f'source/{ds}/{r["index"]}',h=h,x=np.asarray(z['x']),
            geometry=np.asarray(z['geometry']),frame=frame,candidate=cand,anchor=anchor,
            frames=len(ids),source_payload=f'{r["index"]:04}.pt'))
    return ans

def candidate_x(item,view,task):
    a=item['geometry'] if view=='Geometry' else item['x'][:,VIEWS[view]]
    if task=='delta':a=a-a[item['anchor']]
    return a
def frame_x(item,view):
    if view=='Hidden':return item['h']
    ids=item.get('frame_ids')
    if ids is not None:
        return ((ids-ids[0])/(ids[-1]+1-ids[0]))[:,None]
    return item['frame']['position'][:,None]
def model_id(family,view,task,control):return f'{family}/{view}/{task}/{control}'
def split_column(model,j):
    m=dict(model);m['weight']=model['weight'][:,j].copy();m['bias']=float(model['bias'][j]);return m

def fit():
    verify();tick=time.time();all_summary={};model_sha={}
    forbidden=['GT_LABELS_','test_annotations','valv2_proc','target_features','target_metrics']
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            if any(v in str(args[0]) for v in forbidden):raise PermissionError('Source fitting cannot read target assets or labels')
    sys.addaudithook(guard)
    for ds in DATASETS:
        items=source_items(ds);tr=[r for r in items if r['panel']=='train'];va=[r for r in items if r['panel']=='validation']
        models={};details={}
        for family,views in [('candidate',list(VIEWS)),('frame',['Hidden','Geometry'])]:
            for view in views:
                tasks=['precision','recall','tiou'] if family=='candidate' else ['position','start_distance','end_distance']
                for special in [False,True]:
                    selected_tasks=(['delta'] if family=='candidate' else ['phase']) if special else tasks
                    def assemble(rr):
                        xx=[];ys=[];ss=[]
                        for r in rr:
                            x=candidate_x(r,view,'delta' if special else 'tiou') if family=='candidate' else frame_x(r,view)
                            labels=r[family];mask=np.isfinite(labels[selected_tasks[0]])
                            if not mask.any():continue
                            real=np.column_stack([labels[t][mask] for t in selected_tasks]);shuf=[]
                            for t in selected_tasks:
                                y,_,_=shuffle_labels(labels[t],f'{ds}/{r["source_index"]}',
                                    'candidate' if family=='candidate' else ('phase' if t=='phase' else 'frame'),
                                    mask=mask,anchor=r['anchor'] if t=='delta' else None)
                                shuf.append(y[mask])
                            xx.append(x[mask]);ys.append(np.column_stack([real,*shuf]));ss.extend([r['source_index']]*int(mask.sum()))
                        return np.concatenate(xx),np.concatenate(ys),np.asarray(ss)
                    x,y,g=assemble(tr);vx,vy,vg=assemble(va)
                    # Validation targets are always the real labels, including shuffled-fit controls.
                    vy[:,len(selected_tasks):]=vy[:,:len(selected_tasks)]
                    path,stat,at=ridge_path(x,y,g,vx,vy,vg,intercept=not(family=='candidate' and special))
                    for j in range(y.shape[1]):
                        t=selected_tasks[j%len(selected_tasks)];control='real' if j<len(selected_tasks) else 'shuffle'
                        name=model_id(family,view,t,control);models[name]=split_column(path[at[j]],j)
                        details[name]=dict(selected_alpha=ALPHAS[at[j]],selected_index=at[j],dimensions=x.shape[1],
                            training_rows=len(y),validation_rows=len(vy),training_sources=len(np.unique(g)),validation_sources=len(np.unique(vg)),
                            path=[dict(alpha=z['alpha'],validation_MSE=z['validation_MSE'][j],training_MSE=z['training_MSE'][j],equation_error=z['equation_error'][j]) for z in stat])
                    print('ATLAS_RIDGE',ds,family,view,'special' if special else 'ordinary',flush=True)
                if family=='frame':
                    x=np.concatenate([frame_x(r,view) for r in tr]);vx=np.concatenate([frame_x(r,view) for r in va])
                    y=np.concatenate([r['frame']['event'] for r in tr]);vy=np.concatenate([r['frame']['event'] for r in va])
                    g=np.concatenate([np.full(r['frames'],r['source_index']) for r in tr]);vg=np.concatenate([np.full(r['frames'],r['source_index']) for r in va])
                    sy=np.concatenate([shuffle_labels(r['frame']['event'],f'{ds}/{r["source_index"]}','frame')[0] for r in tr])
                    for control,yy in [('real',y),('shuffle',sy)]:
                        path,stat,at=logistic_path(x,yy,g,vx,vy,vg);name=model_id(family,view,'event',control);models[name]=path[at]
                        details[name]=dict(selected_alpha=ALPHAS[at],selected_index=at,dimensions=x.shape[1],
                            training_rows=len(y),validation_rows=len(vy),training_sources=len(np.unique(g)),validation_sources=len(np.unique(vg)),path=stat)
                        print('ATLAS_LOGISTIC',ds,view,control,'selected',ALPHAS[at],flush=True)
        # Source-balanced fitted intercept-only nulls, not target-derived constants.
        null={}
        for family,tasks in [('frame',FRAME_TASKS),('candidate',CANDIDATE_TASKS)]:
            for t in tasks:
                values=[np.mean(r[family][t][np.isfinite(r[family][t])]) for r in tr if np.isfinite(r[family][t]).any()]
                null[f'{family}/{t}']=float(np.mean(values))
        save(BASE/ds/'FROZEN_ATLAS.pt',dict(models=models,null=null));model_sha[ds]=sha(BASE/ds/'FROZEN_ATLAS.pt')
        all_summary[ds]=details;write(BASE/ds/'FIT_SUMMARY.json',details)
        assert len(models)==68,(ds,len(models))
    write(PUB/'SOURCE_PATHS.json',all_summary)
    write(BASE/'FIT_SEAL.json',dict(status='all_136_probes_frozen',time=time.time(),probe_hashes=model_sha,
        source_only_fit=True,target_GT_for_fit=False,target_readout_not_started=True,CPU_wall_seconds=time.time()-tick,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
    status('source_fit_complete_pending_target_readout',models=136,target_GT_for_fit=False)

def compute(item,models):
    scores={}
    for name,m in models.items():
        family,view,task,control=name.split('/')
        x=candidate_x(item,view,task) if family=='candidate' else frame_x(item,view)
        scores[name]=predict(m,x)
        if m['model']=='logistic':scores[name+'/logit']=(x-m['mean'])/m['std']@m['weight']+m['bias']
    for view in VIEWS:
        for control in ['real','shuffle']:
            v=scores[model_id('candidate',view,'tiou',control)]
            scores[model_id('candidate',view,'delta_from_absolute',control)]=v-v[item['anchor']]
    return scores

def target_items(ds,labels=False):
    plans=read(ROOT/f'artifacts/tastvg_current_correction_views_v1/{ds}/PLAN.json')['rows']
    gt={s:read(POOL/ds/f'GT_LABELS_{s}.json') for s in ['search','confirm']} if labels else None
    rows=[]
    for p in sorted((PREV/ds/'target_features').glob('*.pt')):
        z=load(p);ids=np.asarray(z['frame_ids']);h=z['hidden'].double().numpy();r=plans[z['source_id']]
        item=dict(dataset=ds,source_index=z['source_id'],domain='target',panel=z['split'],order=z['order'],
            condition=z['condition'],cell=z['cell_key'],h=h,x=np.asarray(z['x']),geometry=np.asarray(z['geometry']),
            anchor=z['anchor_index'],frames=len(ids),frame_ids=ids,source_payload=p.name,
            state_pre_sha256=z['A_state_pre_sha256'],state_post_sha256=z['A_state_post_sha256'])
        assert r['frame_ids']==z['frame_ids'] and z['source_A_temporal_bitwise_parity'] and z['A_spatial_bitwise_parity']
        if labels:
            span=gt[z['split']][str(z['source_id'])]['span']
            item['frame']=frame_labels(ids,span)
            intervals=[[ids[i],ids[j]+1] for i,j in z['candidate_indices']]
            item['candidate']=candidate_labels(intervals,span,item['anchor'])
        rows.append(item)
    assert len(rows)==144;return rows

def readout():
    verify();tick=time.time();fit=read(BASE/'FIT_SEAL.json');paths={}
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)) and 'GT_LABELS_' in str(args[0]):
            raise PermissionError('Frozen target readout cannot read labels')
    sys.addaudithook(guard)
    for ds in DATASETS:
        assert sha(BASE/ds/'FROZEN_ATLAS.pt')==fit['probe_hashes'][ds]
        models=load(BASE/ds/'FROZEN_ATLAS.pt')['models'];rows=[]
        for item in source_items(ds):
            if item['panel']=='validation':rows.append(dict(cell=item['cell'],predictions=compute(item,models)))
        for item in target_items(ds,labels=False):rows.append(dict(cell=item['cell'],predictions=compute(item,models)))
        save(BASE/ds/'SEALED_READOUT.pt',rows);paths[ds]=sha(BASE/ds/'SEALED_READOUT.pt')
        print('ATLAS_READOUT',ds,len(rows),flush=True)
    write(BASE/'GLOBAL_READOUT_SEAL.json',dict(status='all_readouts_sealed',time=time.time(),files=paths,
        probes=fit['probe_hashes'],source_fit_seal_sha256=sha(BASE/'FIT_SEAL.json'),target_GT_read_in_readout=False,
        CPU_wall_seconds=time.time()-tick))
    status('readout_sealed_pending_GT_characterization')

def describe(item,scores,null):
    metrics={};extra={}
    for name,pred in scores.items():
        if name.endswith('/logit'):continue
        family,view,task,control=name.split('/')
        lab='delta' if task=='delta_from_absolute' else task;y=item[family][lab]
        mask=np.isfinite(y);metrics[name]=moments(y[mask],pred[mask],task=='event',
            scores.get(name+'/logit',None)[mask] if name+'/logit' in scores else None)
        if family=='frame' and task in ['start_distance','end_distance']:
            # Remaining GT-boundary error after subtracting known absolute position.
            # Algebraically equals distance MAE; averaged inferred boundary adds
            # a query-level endpoint estimate that is independent of frame variation.
            pos=item['frame']['position'];difference=pos-pred if task=='start_distance' else pos+pred
            truth=pos-y if task=='start_distance' else pos+y
            extra[name]=dict(endpoint_abs_error=float(abs(difference.mean()-truth.mean())),
                inferred_endpoint_std=float(difference.std()),true_endpoint_std=float(truth.std()))
    for family,tasks in [('frame',FRAME_TASKS),('candidate',CANDIDATE_TASKS)]:
        for t in tasks:
            y=item[family][t];mask=np.isfinite(y);p=np.full(len(y),null[f'{family}/{t}'])
            metrics[f'{family}/Null/{t}/real']=moments(y[mask],p[mask],t=='event')
    return dict(dataset=item['dataset'],domain=item['domain'],panel=item['panel'],source_index=item['source_index'],
        order=item['order'],condition=item['condition'],cell=item['cell'],frames=item['frames'],candidates=32,
        event_frames=int(item['frame']['event'].sum()),event_both_classes=len(np.unique(item['frame']['event']))==2,
        anchor_kind='native' if item['domain']=='source' else 'A8',anchor_index=item['anchor'],metrics=metrics,
        position_removed_endpoints=extra)

def groups(rows):
    out={'source_validation':[r for r in rows if r['domain']=='source']}
    for label,mode in [('target_clean','clean'),('target_corrupt','corrupt')]:
        out[label]=[r for r in rows if r['domain']=='target' and (r['condition']=='clean')==(mode=='clean')]
        for panel in ['search','confirm']:
            out[label+'_'+panel]=[r for r in out[label] if r['panel']==panel]
    return out

def summarize(rows):
    result={};names=list(rows[0]['metrics'])
    for label,rr in groups(rows).items():
        total=dict(cells=len(rr),sources=len({r['source_index'] for r in rr}),frames=sum(r['frames'] for r in rr),
            event_frames=sum(r['event_frames'] for r in rr),no_event_support=sum(r['event_frames']==0 for r in rr),
            event_single_class=sum(not r['event_both_classes'] for r in rr))
        metrics={n:bootstrap_summary(rr,n) for n in names};difference={}
        for name in names:
            family,view,task,control=name.split('/')
            if view in ['Geometry','Null']:continue
            field='auc' if task=='event' else 'r2'
            for other,tag in [(f'{family}/Geometry/{task}/{control}','over_geometry'),
                              (f'{family}/{view}/{task}/shuffle','over_shuffle')]:
                if other in names and control=='real':difference[name+'/'+tag]=paired_difference(rr,name,other,field)
        result[label]=dict(coverage=total,metrics=metrics,paired_differences=difference)
        print('ATLAS_SUMMARY',label,len(rr),flush=True)
    return result

def diagnose():
    verify();tick=time.time();seal=read(BASE/'GLOBAL_READOUT_SEAL.json');coverage={}
    for ds in DATASETS:
        assert sha(BASE/ds/'SEALED_READOUT.pt')==seal['files'][ds]
        frozen=load(BASE/ds/'FROZEN_ATLAS.pt');readout={r['cell']:r['predictions'] for r in load(BASE/ds/'SEALED_READOUT.pt')}
        items=[r for r in source_items(ds) if r['panel']=='validation']+target_items(ds,labels=True)
        rows=[]
        for r in items:
            rows.append(describe(r,readout[r['cell']],frozen['null']))
        write(PUB/ds/'ROWS.json',rows);write(PUB/ds/'SUMMARY.json',summarize(rows))
        coverage[ds]=dict(source_validation_queries=sum(r['domain']=='source' for r in rows),
            target_cells=sum(r['domain']=='target' for r in rows),models=len(frozen['models']))
    write(PUB/'LABEL_JOIN.json',dict(time=time.time(),readout_seal_time=seal['time'],
        raw_target_annotations_read=False,cached_target_span_labels=True,target_GT_used_for_fit=False,
        preflight_schema_inspections_disclosed=4))
    write(PUB/'RESOURCES.json',dict(source_fit=read(BASE/'FIT_SEAL.json')['CPU_wall_seconds'],
        readout=seal['CPU_wall_seconds'],diagnosis=time.time()-tick,new_GPU_calls=0,new_model_calls=0,new_expert_calls=0,
        CUDA_initialized=torch.cuda.is_initialized(),new_backbone_calls=0,new_candidate_calls=0,
        views=6,probes_per_dataset=68,coverage=coverage))
    status('completed_pending_root_audit_report_publication',coverage=coverage)

def main():
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='fit':fit()
    elif action=='readout':readout()
    elif action=='diagnose':diagnose()
    else:raise ValueError(action)
    assert not torch.cuda.is_initialized()
if __name__=='__main__':
    try:main()
    except Exception:
        if BASE.exists():
            write(BASE/'FAILURE.json',dict(time=time.time(),action=sys.argv[1],traceback=traceback.format_exc()))
            status('failed',action=sys.argv[1])
        raise
