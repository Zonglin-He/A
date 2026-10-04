"""After the global prediction barrier: CPU dense scores and pipeline diagnosis."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,collections,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *
from scripts.tastvg_cpu_handoff_v1 import verify_cpu
from scripts.audit_tastvg_dta_oracle_r1_v1 import summary
from scripts.audit_tastvg_negative_evidence_v1 import metric as independent_metric

ARMS=('F','T','S','A','U')
CONTRASTS=(('A','F'),('A','T'),('S','F'),('T','F'),('A','S'))

def labels(direction):
    """The first and only input-label join, called after seal()."""
    p=verify(direction);spec=read(BASE/'GT_INPUT_LOCK.json')[direction]
    ap=Path(spec['annotation']);assert sha(ap)==spec['annotation_sha256'];a=read(ap)
    rows=a if p['target_dataset']=='hc2' else a['videos'];gt={};spans={}
    for r in p['rows']:
        v=rows[r['parent_index']]
        assert v['caption'].lower()==r['input']['caption']
        assert str(v['original_video_id'])==str(r['input']['original_video_id'])
        i=r['parent']
        if p['target_dataset']=='hc2':
            start=int(v['tube_start_frame'])-1;end=start+len(v['trajectory'])-1
            spans[i]=[start,end]
            gt[i]={start+j:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])]
                   for j,(x,y,w,h) in enumerate(v['trajectory'])}
        else:
            start,end=map(int,(v['tube_start_frame'],v['tube_end_frame']));spans[i]=[start,end]
            tr=a['trajectories'][str(v['original_video_id'])][str(v['target_id'])]
            gt[i]={int(fid):[z['bbox'][0],z['bbox'][1],z['bbox'][0]+z['bbox'][2],z['bbox'][1]+z['bbox'][3]]
                   for fid,z in tr.items() if start<=int(fid)<end}
        assert gt[i] and end>start
    return gt,spans

def tails(rows,a,b):
    if not rows:return dict(cells=0)
    d=np.array([r[a+'_v']-r[b+'_v'] for r in rows])
    z=dict(cells=len(rows),improved=int((d>1e-12).sum()),degraded=int((d< -1e-12).sum()),
           unchanged=int((abs(d)<=1e-12).sum()),gross_gain_pp=float(np.maximum(d,0).mean()*100),
           gross_loss_pp=float(-np.minimum(d,0).mean()*100),severe_harm_gt5pp=int((d<-.05).sum()),
           severe_harm_gt20pp=int((d<-.2).sum()))
    for t in [.3,.5]:
        good=np.array([r[b+'_v']>t for r in rows]);after=np.array([r[a+'_v']>t for r in rows]);n=int(good.sum())
        z[str(t)]=dict(correct_before=n,correct_to_wrong=int((good&~after).sum()),
            wrong_to_correct=int((~good&after).sum()),conditional_correct_damage=float((good&~after).sum()/n) if n else None)
    return z

def aggregate(rows,fields):
    z=summary(rows,fields)
    z['tails']={a+'_minus_'+b:tails(rows,a,b) for a,b in CONTRASTS}
    z['orders']={o:summary([r for r in rows if r['order']==o],fields) for o in ['order0','order1','order2']}
    z['order_sample_SD']={k:float(np.std([v['metrics'][k]['mean'] for v in z['orders'].values()
        if v['sources']],ddof=1)) if sum(bool(v['sources']) for v in z['orders'].values())>1 else None for k in fields}
    return z

def gap_recovery(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['source_id']].append([r['A_v']-r['F_v'],r['U_v']-r['F_v']])
    x=np.array([np.mean(groups[i],0) for i in sorted(groups)]);mu=x.mean(0)
    rng=np.random.default_rng(20261004);b=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    eligible=b[:,1]>1e-8;ratio=b[eligible,0]/b[eligible,1]
    return dict(numerator=float(mu[0]),denominator=float(mu[1]),
        denominator_ci95=np.percentile(b[:,1],[2.5,97.5]).tolist(),
        ratio=float(mu[0]/mu[1]) if mu[1]>1e-8 else None,
        positive_denominator_bootstrap_draws=int(eligible.sum()),
        ratio_ci95_conditional_on_positive_denominator=np.percentile(ratio,[2.5,97.5]).tolist() if len(ratio) else None,
        stable_positive_denominator=bool(np.percentile(b[:,1],2.5)>1e-8),
        not_a_mathematical_upper_bound=True)

def recency(distance):
    if distance is None:return 'no_prior_write'
    if distance==1:return '1'
    if distance<=3:return '2-3'
    if distance<=7:return '4-7'
    return '>=8'

def tiou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0]));return inter/max(a[1]-a[0]+b[1]-b[0]-inter,1e-12)

def spatial_frame_quality(pred,row,truth,span,target):
    from scripts.audit_tastvg_negative_evidence_v1 import corners,iou
    frames=np.array(sorted(truth));ids=np.array(row['frame_ids'])
    boxes=corners(pred['boxes'])*np.array([row['input']['width'],row['input']['height']]*2)
    if target=='hc2':boxes=np.maximum(boxes,0)
    dense=np.stack([np.interp(frames,ids,boxes[:,j]) for j in range(4)],-1)
    values=iou(dense,np.array([truth[int(f)] for f in frames]))
    values[(frames<ids[0])|(frames>ids[-1])]=0
    keep=(frames>=span[0])&(frames<span[1])
    return frames[keep],values[keep]

def characteristics(row,truth,span):
    boxes=np.array(list(truth.values()),float);w=row['input']['width'];h=row['input']['height']
    area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)/(w*h)
    ids=row['frame_ids'];rate=row['input']['fps'];length=(ids[-1]-ids[0]+1)/rate
    return dict(event_duration_seconds=(span[1]-span[0])/rate,observed_clip_seconds=length,
        event_fraction=(span[1]-span[0])/(ids[-1]-ids[0]+1),GT_box_area_fraction=float(area.mean()),
        caption_words=len(row['input']['caption'].split()),observed_frames=len(ids),
        input_grid_GT_frame_fraction=sum(span[0]<=j<span[1] for j in ids)/len(ids),
        small_object=float(area.mean())<.02,short_event=(span[1]-span[0])/(ids[-1]-ids[0]+1)<.25)

def diagnose(rows,steps):
    ex=[r for r in rows if r['expert_scheduled']];z={}
    z['path_transitions']={a+'_minus_'+b:tails(rows,a,b) for a,b in
        [('IB','F'),('S','IB'),('A','S'),('A','F'),('A','T')]}
    z['temporal']={}
    for threshold in [.3,.5]:
        z['temporal'][str(threshold)]=dict(scheduled=len(ex),
            good_candidate_present=sum(r['temporal_oracle_v']>threshold for r in ex),
            good_candidate_missed=sum(r['temporal_oracle_v']>threshold and r['A_v']<=threshold for r in ex),
            no_good_candidate=sum(r['temporal_oracle_v']<=threshold for r in ex),
            native_correct_destroyed=sum(r['S_v']>threshold and r['A_v']<=threshold for r in ex),
            native_wrong_rescued=sum(r['S_v']<=threshold and r['A_v']>threshold for r in ex))
    # Tube correctness is conditional on the fixed boxes; it cannot by itself
    # identify absent temporal support when the spatial trajectory is poor.
    z['temporal_tIoU']={str(t):dict(scheduled=len(ex),
        good_candidate_present=sum(r['temporal_oracle_t']>t for r in ex),
        good_candidate_missed=sum(r['temporal_oracle_t']>t and r['A_t']<=t for r in ex),
        no_good_candidate=sum(r['temporal_oracle_t']<=t for r in ex),
        native_correct_destroyed=sum(r['S_t']>t and r['A_t']<=t for r in ex),
        native_wrong_rescued=sum(r['S_t']<=t and r['A_t']>t for r in ex)) for t in [.3,.5]}
    z['evidence']=dict(scheduled=len(ex),empty=sum(r['valid_expert_frames']==0 for r in ex),
        nonempty_without_event=sum(r['valid_expert_frames']>0 and r['valid_event_frames']==0 for r in ex),
        measures={f:summary([r for r in ex if r[f] is not None],[f]) for f in
            ['expert_event_frame_precision','expert_event_GT_IoU']})
    z['arrival_update']=summary(ex,['net_update_A_v','net_update_GT_v'])
    z['spatial']={}
    for at in sorted({s['step'] for s in steps}):
        ss=[s for s in steps if s['step']==at]
        z['spatial'][str(at)]=dict(steps=len(ss),empty_evidence=sum(s['valid_frames']==0 for s in ss),
            no_event_reference=sum(s['valid_frames']>0 and s['valid_event_frames']==0 for s in ss),
            flat_rewards=sum(s['flat_rewards'] for s in ss),
            loss_down_GT_time_harm=sum(s['loss_decreased'] and s['post_GT_v']<s['pre_GT_v']-1e-12 for s in ss),
            loss_down_A_time_harm=sum(s['loss_decreased'] and s['post_A_v']<s['pre_A_v']-1e-12 for s in ss),
            severe_GT_time_harm=sum(s['post_GT_v']<s['pre_GT_v']-.05 for s in ss),
            good_top_but_harm=sum(s['top_best_GT_v']>s['pre_GT_v']+1e-12 and s['post_GT_v']<s['pre_GT_v']-1e-12 for s in ss),
            unique_good_top_but_harm=sum(len(s['top_indices'])==1 and s['top_best_GT_v']>s['pre_GT_v']+1e-12 and s['post_GT_v']<s['pre_GT_v']-1e-12 for s in ss),
            mean_post_minus_pre_GT_v=float(np.mean([s['post_GT_v']-s['pre_GT_v'] for s in ss])))
        z['spatial'][str(at)]['correct_support']={str(t):dict(
            present=sum(s['oracle_GT_v']>t for s in ss),
            absent=sum(s['oracle_GT_v']<=t for s in ss),
            top_group_misses=sum(s['oracle_GT_v']>t and s['top_best_GT_v']<=t for s in ss),
            correct_update_destroyed=sum(s['pre_GT_v']>t and s['post_GT_v']<=t for s in ss)) for t in [.3,.5]}
        for field in ['observed_GT_sIoU_delta','unobserved_GT_sIoU_delta']:
            eligible=[s for s in ss if s[field] is not None]
            z['spatial'][str(at)][field]=summary(eligible,[field])
    for f in ['delta_boxes','delta_native_interval','delta_fast','A_minus_F_v','A_minus_T_v',
              'headroom_GT_time','headroom_GT_space','headroom_A_GT_time','headroom_A_GT_space']:
        z.setdefault('paired_source_effects',{})[f]=summary(rows,[f])
    return z

def run(direction):
    import torch
    verify_cpu(BASE)
    torch.set_num_threads(2);bar=seal();p=verify(direction);out=BASE/direction;target=p['target_dataset']
    start=time.time();tick=time.monotonic();gt,spans=labels(direction)
    write(out/'GT_EXPOSURE.json',dict(time=start,global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        GT_used_for_updates=False,GT_used_for_selection=False,labels_sha256=sha(BASE/'GT_INPUT_LOCK.json')))
    from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    from methods.decota_final_simplified_v1.tensors import state_hash
    official=DenseMetric() if target=='vidstg' else HC2DenseMetric()
    rows=[];steps=[];compute=collections.Counter();states=0;max_dense_error=0.
    frozen_cache={};native_center=read(out/'SUPPORT.json')['center_sha256']
    for order,seq in p['orders'].items():
        previous=native_center
        for at,parent in enumerate(seq):
            f=out/'online'/order/f'{at:05}.pt.gz';rf=f.with_suffix('.json')
            assert sha(f)==read(rf)['sha256'];x=loadz(f);r=p['rows'][parent];g=gt[parent];span=spans[parent]
            assert x['parent']==parent and x['arrival']==at and x['order']==order and x['expert_scheduled']==(at%4==0)
            assert x['pre_sha']==previous==state_hash(x['pre_state'])
            assert x['post_sha']==state_hash(x['post_state']);previous=x['post_sha'];states+=1
            reference=checked(out/'target_reference'/f'{parent:05}.pt')['prediction']
            def m(pred,interval=None,check=False):
                nonlocal max_dense_error
                val=independent_metric(pred,r,g,span,target,interval)
                if check:
                    b=xyxy(pred['boxes'],r['input']['width'],r['input']['height'])
                    if target=='hc2':b=np.maximum(b,0)
                    lit=official(b,r['frame_ids'],pred['physical_interval'] if interval is None else interval,g,span)
                    for a,k in [('v','m_vIoU'),('t','m_tIoU'),('s','sIoU_dense_GT')]:
                        err=abs(val[a]-lit[k]);assert err<1e-10,(parent,a,err);max_dense_error=max(max_dense_error,err)
                return val
            preds=dict(F=x['source_native'],T=x['fast_only'],S=x['slow'],A=x['final'],U=reference)
            z=dict(source_id=parent,order=order,arrival=at,expert_scheduled=x['expert_scheduled'],
                updated=x['updated'],nearest_write_distance=x['nearest_write_distance'],
                recency=recency(x['nearest_write_distance']),pre_state_sha256=x['pre_sha'],post_state_sha256=x['post_sha'],
                displacement=x['displacement'],displacement_from_source=x['displacement_from_source'],
                **characteristics(r,g,span))
            for a,pred in preds.items():
                v=m(pred,check=True)
                z.update({a+'_'+k:val for k,val in v.items()})
                z[a+'_at03']=float(v['v']>.3);z[a+'_at05']=float(v['v']>.5)
            b=m(x['slow'],x['source_native']['physical_interval']);z.update({'IB_'+k:v for k,v in b.items()})
            z.update(delta_boxes=z['IB_v']-z['F_v'],delta_native_interval=z['S_v']-z['IB_v'],
                delta_fast=z['A_v']-z['S_v'],GT_time_v=m(x['slow'],span)['v'],GT_space_v=z['S_t'],joint_GT_v=1.)
            z['headroom_GT_time']=z['GT_time_v']-z['S_v'];z['headroom_GT_space']=z['GT_space_v']-z['S_v']
            z['headroom_A_GT_time']=z['GT_time_v']-z['A_v'];z['headroom_A_GT_space']=z['A_t']-z['A_v']
            for a,b in CONTRASTS:
                for k in ('v','t','s','at03','at05'):z[a+'_minus_'+b+'_'+k]=z[a+'_'+k]-z[b+'_'+k]
            assert abs(z['A_minus_F_v']-z['delta_boxes']-z['delta_native_interval']-z['delta_fast'])<1e-12
            if x['expert_scheduled']:
                er=read(out/'experts/spatial/clean'/f'{parent:05}.json');e=load(out/'experts'/er['cache'])
                valid=np.asarray(e['valid'],bool);ids=np.array(r['frame_ids']);positions=np.flatnonzero(valid)
                event=[j for j in positions if span[0]<=ids[j]<span[1] and int(ids[j]) in g]
                eiou=[]
                from scripts.audit_tastvg_negative_evidence_v1 import iou,corners
                for j in event:
                    truth=np.array(g[int(ids[j])])/np.array([r['input']['width'],r['input']['height']]*2)
                    eiou.append(float(iou(corners(np.asarray(e['boxes'])[j]),truth)))
                z.update(valid_expert_frames=len(positions),valid_event_frames=len(event),
                    expert_event_frame_precision=len(event)/len(positions) if len(positions) else None,
                    expert_event_GT_IoU=float(np.mean(eiou)) if eiou else None)
                td=x['temporal'];cm=[m(x['slow'],q['physical_interval']) for q in td['candidates']]
                tr=read(out/'experts/temporal/clean'/f'{parent:05}.json');te=load(out/'experts'/tr['cache'])
                z.update(temporal_candidate_v=[q['v'] for q in cm],temporal_candidate_t=[q['t'] for q in cm],
                    temporal_selected=td['selected'],temporal_scores=td['scores'],temporal_oracle_v=max(q['v'] for q in cm),
                    temporal_oracle_t=max(q['t'] for q in cm),temporal_regret=max(q['v'] for q in cm)-z['A_v'],
                    expert_proposal_best_t=max([tiou(q,span) for q in te['proposals']],default=0.))
                z['temporal_coverage_gap']=z['GT_time_v']-z['temporal_oracle_v']
                for j,st in enumerate(x['update_steps']):
                    scores=[m(c['prediction'],x['final']['physical_interval'])['v'] for c in st['candidates']]
                    gts=[m(c['prediction'],span)['v'] for c in st['candidates']]
                    u=st['update'];rw=None if st['rewards'] is None else np.array(st['rewards'])
                    top=[] if rw is None else np.flatnonzero(rw>=rw.max()-1e-12).tolist()
                    a=m(st['prediction'],x['final']['physical_interval'])['v'];b=m(st['post_prediction'],x['final']['physical_interval'])['v']
                    aGT=m(st['prediction'],span)['v'];bGT=m(st['post_prediction'],span)['v']
                    ff,preq=spatial_frame_quality(st['prediction'],r,g,span,target)
                    fp,postq=spatial_frame_quality(st['post_prediction'],r,g,span,target);assert np.array_equal(ff,fp)
                    observed=np.isin(ff,ids[positions]);dq=postq-preq
                    steps.append(dict(source_id=parent,order=order,arrival=at,step=j,pre_A_v=a,post_A_v=b,
                        pre_GT_v=aGT,post_GT_v=bGT,candidate_A_v=scores,candidate_GT_v=gts,
                        top_indices=top,top_best_GT_v=max([gts[k] for k in top],default=aGT),
                        top_first_GT_v=gts[top[0]] if top else aGT,oracle_GT_v=max(gts),
                        observed_GT_sIoU_delta=float(dq[observed].mean()) if observed.any() else None,
                        unobserved_GT_sIoU_delta=float(dq[~observed].mean()) if (~observed).any() else None,
                        rewards=None if rw is None else rw.tolist(),valid_frames=len(positions),valid_event_frames=len(event),
                        flat_rewards=bool(rw is not None and np.ptp(rw)<=1e-12),
                        loss_decreased=bool(u is not None and u['loss_after']<u['loss_before']-1e-12),
                        loss_before=u['loss_before'] if u else None,loss_after=u['loss_after'] if u else None,
                        gradient_norm=u['global_gradient_norm'] if u else 0.,
                        pre_state_sha256=st['pre_state_sha256'],post_state_sha256=st['post_state_sha256']))
                first=x['update_steps'][0]
                z['spatial_oracle_A_v']=max(m(c['prediction'],x['final']['physical_interval'])['v'] for c in first['candidates'])
                z['spatial_oracle_GT_v']=max(m(c['prediction'],span)['v'] for c in first['candidates'])
                z['net_update_A_v']=m(x['post_prediction'],x['final']['physical_interval'])['v']-z['A_v']
                z['net_update_GT_v']=m(x['post_prediction'],span)['v']-z['GT_time_v']
            compute.update(x['compute']);rows.append(z)
            if len(rows)%48==0:print('CPU_SCORE',direction,len(rows),p['total'],flush=True)
    assert states==p['total'] and sum(r['expert_scheduled'] for r in rows)==p['expert_scheduled_total']
    fields=[a+'_'+k for a in ARMS for k in ['v','t','s','at03','at05']]
    fields += [a+'_minus_'+b+'_'+k for a,b in CONTRASTS for k in ['v','t','s','at03','at05']]
    fields += ['delta_boxes','delta_native_interval','delta_fast','headroom_GT_time','headroom_GT_space']
    panels={}
    for name in ['all','expert','nonexpert']:
        rr=[r for r in rows if name=='all' or r['expert_scheduled']==(name=='expert')]
        panels[name]=aggregate(rr,fields)
    byrecency={name:summary([r for r in rows if r['recency']==name],['A_minus_T_v','S_minus_F_v'])
               for name in ['no_prior_write','1','2-3','4-7','>=8']}
    byshape={name:aggregate([r for r in rows if r[field]==value],fields) for name,field,value in
        [('small_object','small_object',True),('larger_object','small_object',False),
         ('short_event','short_event',True),('longer_event','short_event',False)]}
    result=dict(panels=panels,recency=byrecency,characteristic_slices=byshape,gap_recovery=gap_recovery(rows))
    diagnosis=diagnose(rows,steps)
    config=dict(direction=direction,source_dataset=p['source_dataset'],target_dataset=target,
        queries=p['queries'],sources=p['sources'],arrivals=p['total'],scheduled=p['expert_scheduled_total'],
        source_checkpoint_sha256=p['source_checkpoint_sha256'],target_checkpoint_sha256=p['target_checkpoint_sha256'],
        params=p['params'],parameters=1792,condition='clean',orders=p['order_seeds'],availability=25,
        method_names={'F':'Frozen','T':'Fast-only','S':'Spatial-only','A':'Full A','U':'Target-trained reference'},
        historically_exposed=True,target_hyperparameter_tuning=False,production_promoted=False,
        predict_before_write=True,GT_used_for_updates=False,GT_used_for_selection=False,
        bootstrap_seed=20261004,bootstrap_draws=10000,global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'))
    d=PUB/direction;d.mkdir(parents=True,exist_ok=True)
    for name,value in [('CONFIG',config),('ROWS',rows),('STEP_ROWS',steps),('SUMMARY',result),('PIPELINE_DIAGNOSIS',diagnosis)]:
        write(d/(name+'.json'),value)
    write(d/'CASES.json',{f:dict(good=sorted(rows,key=lambda r:-r[f])[:8],failure=sorted(rows,key=lambda r:r[f])[:8])
        for f in ['A_minus_F_v','A_minus_T_v','delta_boxes','delta_fast']})
    write(d/'PREDICTION_BARRIER.json',dict(GT_read=False,time=bar['time'],score_start_time=start,
        global_barrier_sha256=config['global_barrier_sha256'],runtime_lock_sha256=config['runtime_lock_sha256']))
    write(d/'RESOURCES.json',dict(online_compute=dict(compute),CPU_wall_seconds=time.monotonic()-tick,
        CUDA_initialized=torch.cuda.is_initialized(),new_GPU_calls=0,max_dense_error=max_dense_error,
        GPU_stage_resources={stage:read(out/(stage+'_RESOURCES.json')) for stage in ['CAPTURE','TARGET_REFERENCE','ONLINE']},
        expert_allocation_files={stage:[read(f) for f in (out/'experts/allocations').glob('*.json')
            if read(f)['stage']==stage] for stage in ['spatial','temporal']}))
    write(out/'SCORE_COMPLETION.json',dict(status='scored_diagnosed_pending_root_audit_publication',
        cells=len(rows),steps=len(steps),time=time.time()))
    print('SCORED_DIAGNOSED',direction,len(rows),len(steps),flush=True)

if __name__=='__main__':run(sys.argv[1])
