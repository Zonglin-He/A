"""Score exact sealed baseline outputs on CPU; no model, optimizer or downloads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
import sys,time,gzip,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_paper_baseline_scoring_common_v1 import *
from scripts.audit_decota_paper_baseline_scoring_v1 import metric
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,box_iou
from vg_tta.tastvg_paper48_metrics_v1 import xyxy,DenseMetric
from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric

FIELDS=['After_v','After_t','After_s','After_R0.3','After_R0.5','Before_v','Before_t','Before_s',
        'delta_total_v','delta_total_t','delta_total_s','delta_inherited_v','delta_current_v',
        'inherited_boxes_v','inherited_time_v','current_boxes_v','current_time_v',
        'correct_to_wrong_0.3','wrong_to_correct_0.3','correct_to_wrong_0.5','wrong_to_correct_0.5']

def statistics(rows):
    queries=sorted(set(r['query_id'] for r in rows));sources=sorted(set(r['source_id'] for r in rows))
    qi={q:i for i,q in enumerate(queries)};si={s:i for i,s in enumerate(sources)}
    data=np.zeros((len(queries),len(FIELDS)));ns=np.zeros(len(queries));qsrc=np.zeros(len(queries),dtype=int)
    for r in rows:
        i=qi[r['query_id']];data[i]+=[r[f] for f in FIELDS];ns[i]+=1;qsrc[i]=si[r['source_id']]
    assert (ns==3).all();data/=ns[:,None]
    assert len({(r['query_id'],r['order']) for r in rows})==len(rows)
    count=np.bincount(qsrc,minlength=len(sources)).astype(float)
    sums=np.stack([np.bincount(qsrc,weights=data[:,j],minlength=len(sources)) for j in range(len(FIELDS))],axis=1)
    matrix=sums/count[:,None];rng=np.random.default_rng(SEED);b=[];bq=[]
    order_query={};order_source={}
    for o in ['order1','order2','order3']:
        rr=[r for r in rows if r['order']==o];v=np.array([[r[f] for f in FIELDS] for r in rr]);s=np.array([si[r['source_id']] for r in rr])
        oc=np.bincount(s,minlength=len(sources))
        order_query[o]=v.mean(0)
        order_source[o]=np.stack([np.bincount(s,weights=v[:,j],minlength=len(sources))/oc for j in range(len(FIELDS))],axis=1).mean(0)
    for _ in range(100):
        ids=rng.integers(0,len(sources),(100,len(sources)))
        b.append(matrix[ids].mean(1));bq.append(sums[ids].sum(1)/count[ids].sum(1)[:,None])
    ci=np.quantile(np.concatenate(b),[.025,.975],axis=0);qci=np.quantile(np.concatenate(bq),[.025,.975],axis=0)
    out={}
    for j,f in enumerate(FIELDS):
        a=matrix[:,j];z=dict(source_macro=float(a.mean()),query_macro=float(data[:,j].mean()),
             ci95_source=ci[:,j].tolist(),ci95_query_clustered=qci[:,j].tolist(),
             orders={o:float(order_query[o][j]) for o in order_query},
             order_source_macro={o:float(order_source[o][j]) for o in order_source})
        if f.startswith('delta_'):
            z.update(gross_gain_pp=float(np.maximum(a,0).mean()*100),gross_loss_pp=float(-np.minimum(a,0).mean()*100),
               harm_gt5pp_sources=int((a<-.05).sum()),harm_gt20pp_sources=int((a<-.20).sum()),
               harm_gt5pp_queries=int((data[:,j]<-.05).sum()),harm_gt20pp_queries=int((data[:,j]<-.20).sum()))
        out[f]=z
    return dict(sources=len(sources),queries=len(queries),logical_rows=len(rows),metrics=out)

def emit_scores(boxes,row,gt,span,iv,ds,official,check_official):
    tube=DenseTube(boxes,row,gt,span,clip=ds=='hc2');v=tube.score(iv)
    z=metric(boxes,row,gt,span,iv,ds=='hc2');assert max(abs(v[k]-z[k]) for k in v)<2e-10
    if check_official:
        corners=xyxy(boxes,row['input']['width'],row['input']['height'])
        if ds=='hc2':corners=np.maximum(corners,0)
        w=official(corners,row['frame_ids'],iv,gt,span)
        for k,f in [('v','m_vIoU'),('t','m_tIoU'),('s','sIoU_dense_GT')]:assert abs(v[k]-w[f])<2e-10,(k,v[k],w[f])
    return v,tube

def qdiagnosis(row,gt,span,expert):
    ids=sorted(gt);b=np.array([gt[f] for f in ids]);center=(b[:,:2]+b[:,2:])/2
    if len(ids)>1:
        speed=np.linalg.norm(np.diff(center,axis=0)/np.array([row['input']['width'],row['input']['height']]),axis=1)
        motion=float(np.mean(speed/(np.diff(ids)/row['input']['fps'])))
    else:motion=0.
    quality=[];unknown=0
    for anchor in expert['anchors']['single4']:
        fid=row['frame_ids'][anchor['position']]
        if fid in gt:quality.append(float(box_iou(xyxy(anchor['box'],row['input']['width'],row['input']['height']),gt[fid])))
        else:unknown+=1
    return dict(query_type=row['query_type'],development_source_exposed=row['development_source_exposed'],
       GT_duration_seconds=(span[1]-span[0])/row['input']['fps'],GT_track_motion_per_second=motion,
       admitted_known_GT_frames=len(quality),admitted_unknown_GT_frames=unknown,
       admitted_expert_GT_iou=float(np.mean(quality)) if quality else None,
       admitted_frames=len(expert['anchors']['single4']))

def run(ds):
    import torch
    torch.set_num_threads(2)
    runtime,barrier=verify();out=PUB/ds;out.mkdir(parents=True,exist_ok=True)
    if (BASE/(ds+'_COMPLETION.json')).exists(): return
    exposure=BASE/(ds+'_GT_EXPOSURE.json')
    write(exposure,dict(status='authorized_postseal_GT_exposure',evaluation_barrier_sha256=sha(BASE/'EVALUATION_BARRIER.json'),
           selected_original_prediction_barriers=9,EATA_inference_stays_paused=True,time=time.time()))
    plan=read(PAPER/ds/'PLAN.json');dense,spans,gtproof=truth(ds,plan);n=plan['queries'];job='t1_ours_vid' if ds=='vidstg' else 't1_ours_hc2'
    sources={s:i for i,s in enumerate(sorted(set(r['source'] for r in plan['rows'])))}
    official=DenseMetric() if ds=='vidstg' else HC2DenseMetric()
    tick=time.time();source={};diagnoses={};rows_by_method=collections.defaultdict(list);audits={};total=0
    def row_metrics(method,parent,order,arrival,before,biv,after,aiv,compute,details):
        nonlocal total
        row=plan['rows'][parent];gt=dense[parent];span=spans[parent];check=parent%101==0
        pre,pt=emit_scores(before,row,gt,span,biv,ds,official,check);post,at=emit_scores(after,row,gt,span,aiv,ds,official,check)
        frozen=source[parent]['score'];iv=source[parent]['interval'];box_pre=pt.score(iv);box_after=at.score(biv)
        r=dict(method=method,dataset=ds,query_id=parent,source_id=sources[row['source']],order=order,arrival=arrival,
          **diagnoses[parent],**details,compute=compute)
        for name,vals in [('Before',pre),('After',post),('Frozen',frozen)]:
            r.update({name+'_'+k:v for k,v in vals.items()})
            for t in [.3,.5]:r[name+'_R'+str(t)]=float(vals['v']>t)
        r.update(delta_total_v=post['v']-frozen['v'],delta_total_s=post['s']-frozen['s'],delta_total_t=post['t']-frozen['t'],
          delta_inherited_v=pre['v']-frozen['v'],delta_current_v=post['v']-pre['v'],
          inherited_boxes_v=box_pre['v']-frozen['v'],inherited_time_v=pre['v']-box_pre['v'],
          current_boxes_v=box_after['v']-pre['v'],current_time_v=post['v']-box_after['v'],
          GT_time_readout_headroom_v=at.score(span)['v']-post['v'])
        for t in [.3,.5]:
            r['correct_to_wrong_'+str(t)]=float(frozen['v']>t and post['v']<=t)
            r['wrong_to_correct_'+str(t)]=float(frozen['v']<=t and post['v']>t)
        assert abs(r['delta_total_v']-sum(r[k] for k in ['inherited_boxes_v','inherited_time_v','current_boxes_v','current_time_v']))<1e-12
        rows_by_method[method].append(r);total+=1
    for parent,row in enumerate(plan['rows']):
        a,m,rc=load_prediction(f'table1_stateless/{job}/{parent:05}.npz');ia,im,ir=load_prediction(f'{job}/inputs/clean/{parent:05}.npz')
        assert m['input_sha256']==ir['sha256'] and np.array_equal(a['Source'],ia['native_boxes'])
        assert m['control_lock_sha256']==sha(PAPER/'table1_stateless/RUNTIME_LOCK.json')
        assert m['frame_ids']==row['frame_ids']==im['frame_ids'] and m['interval']==im['interval'] and m['parent']==parent
        from vg_tta.decota_paper_controls_v1 import dino_refine
        refined,control=dino_refine(ia['native_boxes'],im['frame_ids'],im['expert']);assert np.array_equal(refined,a['DINO_Refine']) and control==m['control']
        fs,_=emit_scores(a['Source'],row,dense[parent],spans[parent],m['interval'],ds,official,parent%101==0)
        source[parent]=dict(score=fs,interval=m['interval'],boxes=a['Source'])
        diagnoses[parent]=qdiagnosis(row,dense[parent],spans[parent],im['expert'])
        ref,rm,_=load_prediction(f'table1_target_reference/{ds}/{parent:05}.npz')
        assert rm['input_metadata_sha256']==digest(row['input']) and rm['frame_ids']==row['frame_ids']
        for order,seq in plan['orders'].items():
            # Index maps are built once below; official query identity, not arrival identity.
            at=arrival_maps[ds][order][parent]
            for method,key in [('Source Only','Source'),('DINO-Refine','DINO_Refine')]:
                row_metrics(method,parent,order,at,a['Source'],m['interval'],a[key],m['interval'],
                   dict(cached_readout=True,new_model_forwards=0,new_DINO_calls=0,
                        shared_uncached_DINO_calls=m['shared_uncached_DINO_calls'] if method=='DINO-Refine' else 0,
                        shared_expert_seconds=m['shared_expert_seconds'] if method=='DINO-Refine' else 0),dict(optimizer_steps=0,recoveries=0))
            row_metrics('Target-trained reference',parent,order,at,a['Source'],m['interval'],ref['Reference'],rm['interval'],
                        dict(rm['compute'],stateless_prediction_reused_three_orders=True),dict(optimizer_steps=0,recoveries=0))
        if (parent+1)%100==0:
            status(BASE/(ds+'_STATUS.json'),dict(status='scoring_sealed_stateless_controls',dataset=ds,done=parent+1,total=n,GT_read=True,time=time.time()))
            print('STATELESS_GT_SCORE',ds,parent+1,n,flush=True)
    methods=['TENT','SAR']+(['EATA'] if ds=='hc2' else [])
    for method in methods:
        dest=PAPER/'baselines'/f'{method}_{ds}';scope=read(dest/'PARAMETER_SCOPE.json');count=0;coord=0;samples=0;changed=0;math_checks=[]
        from vg_tta.decota_paper_baseline_math_v1 import audit_updates
        for parent in [0,1]:
            z=torch.load(PAPER/'baselines/smoke'/job/method/f'{parent:05}.pt',map_location='cpu',weights_only=False)
            math_checks.append(audit_updates(z['audit']))
        source_state=None
        for order,seq in plan['orders'].items():
            previous=None;previous_sha=None;previous_state_hash=None
            for arrival,parent in enumerate(seq):
                a,m,rc=load_prediction(f'baselines/{method}_{ds}/{order}/{arrival:05}.npz');row=plan['rows'][parent]
                assert m['method']==method and m['dataset']==ds and m['parent']==parent and m['order']==order and m['arrival']==arrival
                assert not m['GT_read'] and m['port_lock_sha256']==sha(PAPER/'baselines/RUNTIME_LOCK.json')
                assert m['frame_ids']==row['frame_ids'] and m['input_metadata_sha256']==digest(row['input'])
                assert m['previous_payload_sha256']==previous_sha
                sb,sa=a['state_before'],a['state_after'];assert sb.shape==sa.shape==(scope['count'],) and np.isfinite(sb).all() and np.isfinite(sa).all()
                if previous is not None:
                    assert np.array_equal(sb,previous) and m['parameter_before_sha256']==previous_state_hash
                else:
                    if source_state is None:source_state=sb.copy()
                    assert np.array_equal(sb,source_state)
                    assert digest(sb.tolist())==m['parameter_before_sha256']
                    assert np.array_equal(a['Before'],source[parent]['boxes']) and m['interval_before']==source[parent]['interval']
                # Every after-state hash is independently recomputed, including skips.
                if np.array_equal(sb,sa):
                    assert m['parameter_after_sha256']==m['parameter_before_sha256']
                else:
                    assert digest(sa.tolist())==m['parameter_after_sha256'];changed+=1
                coord+=2*len(sb)
                if 'optimizer_sample' in m:
                    spec=m['optimizer_sample'];f=PAPER/spec['path'];assert sha(f)==spec['sha256']
                    state=torch.load(f,map_location='cpu',weights_only=False)
                    flat=np.concatenate([p.numpy().reshape(-1) for p in state['parameters']]).astype(np.float32)
                    assert np.array_equal(flat,sa) and state['arrivals']==arrival+1 and state['method']==method
                    samples+=1
                ct=m['audit']['counts'];assert not m['audit']['GT_read'] and m['audit']['arrival']==arrival+1
                row_metrics(method+'-STVG',parent,order,arrival,a['Before'],m['interval_before'],a['After'],m['interval_after'],m['compute'],
                            dict(optimizer_steps=ct['optimizer_steps'],backward_steps=ct['backward_steps'],recoveries=ct['recoveries'],
                              reliable_update=ct['optimizer_steps']>0,skip_counts=dict(collections.Counter(t.get('skip','updated') for t in m['audit']['trace'])),
                              recorded_state_changed=not np.array_equal(sb,sa)))
                previous=sa;previous_sha=rc['sha256'];previous_state_hash=m['parameter_after_sha256'];count+=1
                if count%100==0:
                    status(BASE/(ds+'_STATUS.json'),dict(status='scoring_and_auditing_saved_online_state',method=method,dataset=ds,done=count,total=n*3,GT_read=True,time=time.time()))
                    print('BASELINE_GT_SCORE',ds,method,count,n*3,flush=True)
        audits[method]=dict(arrivals=count,state_coordinate_checks=coord,changed_states=changed,optimizer_snapshots=samples,
            qualification_math=math_checks,full_saved_state_chain=True,all_after_state_hashes_verified=True,
            production_gradients_retained=False,production_optimizer_math_or_model_Jacobian_independently_replayed=False)
    summaries={}
    for method,rows in rows_by_method.items():
        filename=method.lower().replace(' ','_').replace('-','_')+'.jsonl.gz'
        with gzip.open(out/filename,'xt',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
        s=statistics(rows);s['row_file']=filename;s['one_direction_only']=method=='EATA-STVG';summaries[method]=s
    write(out/'SUMMARY.json',dict(dataset=ds,target_split='validation' if ds=='hc2' else 'test',queries=n,
         sources=plan['sources'],logical_rows=total,methods=summaries,GT_provenance=gtproof,
         official_function_checks='each stateless query and online arrival whose ordinal is divisible by 101; all arrays also use independent vector geometry',
         dense_formula_checks_all_readouts=True,recorded_state_audits=audits,CPU_seconds=time.time()-tick,
         exposure='historically exposed complete official target cohort; not independent confirmation',time=time.time()))
    write(BASE/(ds+'_COMPLETION.json'),dict(status='scored_saved_state_and_dense_audited',rows=total,
         files={str(p.relative_to(ROOT)):sha(p) for p in out.iterdir() if p.is_file()},time=time.time()))

arrival_maps={}
def main():
    verify()
    for ds in ['vidstg','hc2']:
        p=read(PAPER/ds/'PLAN.json');arrival_maps[ds]={o:{parent:i for i,parent in enumerate(seq)} for o,seq in p['orders'].items()}
    if len(sys.argv)>1:run(sys.argv[1]);return
    for ds in ['vidstg','hc2']:run(ds)
    merge()
def merge():
    verify();datasets={};seconds=0
    for ds in ['vidstg','hc2']:
        assert read(BASE/(ds+'_COMPLETION.json'))['status']=='scored_saved_state_and_dense_audited'
        d=read(PUB/ds/'SUMMARY.json');datasets[ds]=d['methods'];seconds+=d['CPU_seconds']
    write(PUB/'SUMMARY.json',dict(datasets=datasets,logical_rows=217221,paired_source_bootstrap=10000,seed=SEED,
          EATA_VidSTG='paused_not_scored',EATA_HC2='one_direction_supplement',OPD_main='not_full_cohort_evaluated',
          actual_scoring_CPU_seconds=seconds,all_original_paper_stages_complete=False,time=time.time()))
    write(BASE/'SCORING_COMPLETION.json',dict(status='all_selected_scoring_complete_pending_root_statistics_report_publication',
          rows=217221,summary_sha256=sha(PUB/'SUMMARY.json'),time=time.time()))

if __name__=='__main__':main()
