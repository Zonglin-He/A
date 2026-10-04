"""Only after the global seal: independent attention math and official GT metrics."""
import sys, time, collections, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_privileged_p0_common_v1 import *
import numpy as np
import torch


def softmax(z):
    x=np.exp(z-z.max(-1,keepdims=True));return x/x.sum(-1,keepdims=True)


def support_numpy(raw_boxes,raw_scores):
    b=np.asarray(raw_boxes,dtype=np.float64);scores=np.asarray(raw_scores,dtype=np.float64)
    xy=np.clip(np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],0,1)
    valid=np.isfinite(b).all(1)&np.isfinite(scores)&(scores>=.35)&(xy[:,2:]>xy[:,:2]).all(1)
    idx=np.flatnonzero(valid);xy=xy[idx]
    boxes=np.c_[(xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]]
    weights=softmax(scores[idx][None])[0] if len(idx) else np.empty(0)
    return idx,boxes,weights


def field_numpy(boxes,weights,h,w,padding):
    pad=np.asarray(padding,bool).reshape(h,w);valid=~pad
    vh=valid.any(1).sum();vw=valid.any(0).sum()
    assert np.array_equal(valid,(np.arange(h)[:,None]<vh)&(np.arange(w)[None,:]<vw))
    yy,xx=np.meshgrid((np.arange(h)+.5)/vh,(np.arange(w)+.5)/vw,indexing='ij')
    field=np.zeros((h,w))
    for box,weight in zip(boxes,weights):
        field+=weight*np.exp(-.5*(((xx-box[0])/(box[2]/2))**2+((yy-box[1])/(box[3]/2))**2))
    field[pad]=1
    return field.flatten()


def dense_independent(boxes,row,truth,span,interval,dataset):
    # Deliberately scalar interpolation/area arithmetic, independent of scorer.
    boxes=np.asarray(boxes,dtype=np.float64);width=row['input']['width'];height=row['input']['height']
    xy=np.c_[boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2]*np.array([width,height,width,height])
    if dataset=='hc2':xy=np.maximum(xy,0)
    ids=np.array(row['frame_ids']);ious={}
    for fid,target in sorted(truth.items()):
        if fid<ids[0] or fid>ids[-1]:ious[fid]=0.;continue
        j=int(np.searchsorted(ids,fid))
        if j==0 or ids[j]==fid:p=xy[j]
        else:
            a=(fid-ids[j-1])/(ids[j]-ids[j-1]);p=(1-a)*xy[j-1]+a*xy[j]
        q=np.asarray(target);inter=max(min(p[2],q[2])-max(p[0],q[0]),0)*max(min(p[3],q[3])-max(p[1],q[1]),0)
        union=max(p[2]-p[0],0)*max(p[3]-p[1],0)+max(q[2]-q[0],0)*max(q[3]-q[1],0)-inter
        ious[fid]=float(inter/union) if union>0 else 0.
    s,e=map(int,interval);g,h=span;overlap=max(min(e,h)-max(s,g),0)
    v=sum(i for f,i in ious.items() if max(s,g)<=f<min(e,h))/max(max(e,h)-min(s,g),1)
    return dict(v=v,t=overlap/(e-s+h-g-overlap),s=sum(ious.values())/len(ious)),ious


FIELDS=['ordinary_v','privileged_v','delta_v','ordinary_s','privileged_s','delta_s',
        'ordinary_t','privileged_t','gross_gain_v','gross_loss_v']


def summarize(rows):
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['source_id']].append([r[k] for k in FIELDS])
    ids=sorted(grouped);x=np.array([np.mean(grouped[i],0) for i in ids])
    rng=np.random.default_rng(20261004);boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=20261004,
        metrics={k:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),
            source_values={str(i):float(v) for i,v in zip(ids,x[:,j])}) for j,k in enumerate(FIELDS)},
        harm_gt5pp=sum(r['delta_v']<-.05 for r in rows),harm_gt20pp=sum(r['delta_v']<-.20 for r in rows),
        gain_gt5pp=sum(r['delta_v']>.05 for r in rows),gain_gt20pp=sum(r['delta_v']>.20 for r in rows),
        correctness={str(t):dict(ordinary_correct=sum(r['ordinary_v']>=t for r in rows),
            privileged_correct=sum(r['privileged_v']>=t for r in rows),
            destroyed=sum(r['ordinary_v']>=t and r['privileged_v']<t for r in rows),
            recovered=sum(r['ordinary_v']<t and r['privileged_v']>=t for r in rows)) for t in [.3,.5]},
        active_evidence_frames=sum(r['active_evidence_frames'] for r in rows),
        empty_evidence_cells=sum(r['active_evidence_frames']==0 for r in rows))


def all_summaries(rows):
    result={}
    for ds in DATASETS:
        result[ds]={}
        for split in ['search','confirm','pooled']:
            rr=[r for r in rows if r['dataset']==ds and (split=='pooled' or r['split']==split)]
            conditions=sorted({r['condition'] for r in rr})
            result[ds][split]=dict(corruption=summarize([r for r in rr if r['condition']!='clean']),
                clean=summarize([r for r in rr if r['condition']=='clean']),
                conditions={c:summarize([r for r in rr if r['condition']==c]) for c in conditions})
    return result


def public_check(directory):
    p=Path(directory);rows=read(p/'ROWS.json');assert len(rows)==192
    assert len({(r['dataset'],r['source_id'],r['condition']) for r in rows})==192
    checks=0
    for r in rows:
        for k in ['v','s']:
            assert abs(r['delta_'+k]-(r['privileged_'+k]-r['ordinary_'+k]))<1e-14;checks+=1
        assert r['ordinary_t']==r['privileged_t'] and r['parameter_updates']==r['backwards']==0;checks+=2
        assert r['gross_gain_v']==max(r['delta_v'],0) and r['gross_loss_v']==max(-r['delta_v'],0);checks+=2
        assert 0<=r['active_evidence_frames']<=r['requested_frames']<=4;checks+=1
    expected=all_summaries(rows);assert expected==read(p/'SUMMARY.json');checks+=6*8*len(FIELDS)*3
    decision=read(p/'DECISION.json');gate=all(expected[d][s]['corruption']['metrics']['delta_v']['mean']>0 for d in DATASETS for s in ['search','confirm'])
    assert decision['P0_gate_pass']==gate and decision['method_promoted'] is False
    return dict(status='pass',checks=checks,cells=192,summary_and_bootstrap_recomputed=True)


def run():
    from scripts.tastvg_decota_c1_common_v1 import cache
    from vg_tta.tastvg_oracle_event5_v1 import official
    torch.set_num_threads(4);seal=verify_seal();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    cpu_files=['scripts/score_audit_tastvg_privileged_p0_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
        'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py']
    write(BASE/'CPU_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in cpu_files},
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
    tick=time.time();rows=[];diagnostics=[];checks=0;maxattn=maxbias=maxmetric=0.
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');labels={s:read(POOL/ds/f'GT_LABELS_{s}.json') for s in ['search','confirm']}
        for row in p['rows']:
            for cond in p['conditions']:
                f=BASE/ds/'predictions'/cond/f"{row['ordinal']:05}.pt";x=checked(f);pred=x['prediction'];ev=x['evidence']
                assert x['dataset']==ds and x['source_id']==row['ordinal'] and x['split']==row['split'] and x['condition']==cond
                assert pred['weights_unchanged'] and pred['parameter_updates']==pred['backwards']==0 and not x['GT_read']
                data,rc=cache(ds,row['pool_parent'],cond)
                assert x['pixel_sha256']==rc['pixel_sha256'] and torch.equal(pred['ordinary']['boxes'],data['prediction']['boxes'])
                assert x['native']['physical_interval']==data['prediction']['physical_interval']
                native=data['prediction'];assert ev['positions']==[r['position'] for r in ev['observations']]
                observations={};candidates=[]
                for obs in ev['observations']:
                    assert native['indices'][0]<=obs['position']<=native['indices'][1]
                    idx,boxes,weights=support_numpy(obs['raw_boxes'],obs['raw_scores'])
                    actual=obs['support'];assert idx.tolist()==actual['indices'].tolist()
                    if len(idx):
                        np.testing.assert_allclose(boxes,actual['boxes'],atol=1e-7,rtol=0)
                        np.testing.assert_allclose(weights,actual['weights'],atol=1e-7,rtol=0)
                    observations[obs['position']]=(boxes,weights);candidates.append(len(idx));checks+=len(obs['raw_scores'])+len(idx)*5+1
                allocation={a:dict(observed_visual_mass=[],unobserved_visual_mass=[],evidence_mass=[]) for a in ['ordinary','privileged']}
                for arm in allocation:
                    for offset,z in enumerate(pred[arm]['attention']):
                        h,w=data['views'][offset]['info']['fea_map_size'];nvis=h*w
                        masks=data['views'][offset]['info']['encoded_mask'].numpy()[:,:nvis]
                        for layer in z['layers']:
                            raw=layer['raw_logits'].numpy().astype(np.float64);bias=layer['bias'].numpy().astype(np.float64)
                            expect=np.zeros_like(bias);fieldmap={}
                            for local,pad in enumerate(masks):
                                pos=2*local+offset
                                if pos in observations and len(observations[pos][0]):
                                    field=field_numpy(*observations[pos],h,w,pad);fieldmap[local]=field
                                    if arm=='privileged':expect[local,:nvis]=np.where(~pad,np.log(field+1e-6),0)
                            error=float(np.max(np.abs(bias-expect)));maxbias=max(maxbias,error);assert error<2e-4,error
                            attn=softmax(raw+bias[:,None,:]).mean(1)
                            actual=layer['attention'].numpy().astype(np.float64)
                            error=float(np.max(np.abs(actual-attn)));maxattn=max(maxattn,error);assert error<2e-6,error
                            np.testing.assert_allclose(actual.sum(-1),1,atol=3e-7,rtol=0)
                            checks+=bias.size+attn.size
                            for local,aa in enumerate(actual):
                                observed=2*local+offset in observations;key='observed_visual_mass' if observed else 'unobserved_visual_mass'
                                allocation[arm][key].append(float(aa[:nvis].sum()))
                                if local in fieldmap:allocation[arm]['evidence_mass'].append(float(np.dot(aa[:nvis],fieldmap[local])/max(aa[:nvis].sum(),1e-30)))
                if not any(candidates):assert torch.equal(pred['ordinary']['boxes'],pred['privileged']['boxes'])
                gt=labels[row['split']][str(row['pool_parent'])];truth={int(k):v for k,v in gt['truth'].items()};span=gt['span']
                metrics={};ious={}
                for arm in ['ordinary','privileged']:
                    boxes=pred[arm]['boxes'].numpy();metrics[arm]=official(boxes,row,truth,span,native['physical_interval'],ds)
                    independent,ious[arm]=dense_independent(boxes,row,truth,span,native['physical_interval'],ds)
                    error=max(abs(independent[k]-metrics[arm][k]) for k in ['v','t','s']);maxmetric=max(maxmetric,error);assert error<2e-12,error;checks+=len(truth)+3
                requested=[row['frame_ids'][o['position']] for o in ev['observations']]
                observed_gt=[i for i in requested if i in truth];unobserved_gt=[i for i in truth if i not in requested]
                expert_quality=[]
                for pos,(boxes,weights) in observations.items():
                    fid=row['frame_ids'][pos]
                    if not len(boxes) or fid not in truth:continue
                    scale=np.array([row['input']['width'],row['input']['height']]*2)
                    xy=np.c_[boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2]*scale
                    gtbox=np.asarray(truth[fid]);inter=np.maximum(np.minimum(xy[:,2:],gtbox[2:])-np.maximum(xy[:,:2],gtbox[:2]),0).prod(-1)
                    union=(xy[:,2:]-xy[:,:2]).prod(-1)+np.maximum(gtbox[2:]-gtbox[:2],0).prod()-inter
                    iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
                    expert_quality.append(dict(proposals=len(boxes),best_IoU=float(iou.max()),weighted_IoU=float(np.dot(iou,weights))))
                frame_changes=(pred['ordinary']['boxes']!=pred['privileged']['boxes']).any(-1)
                posset={o['position'] for o in ev['observations']}
                dv=metrics['privileged']['v']-metrics['ordinary']['v'];dsv=metrics['privileged']['s']-metrics['ordinary']['s']
                r=dict(dataset=ds,split=row['split'],source_id=row['ordinal'],condition=cond,
                    **{a+'_'+k:v for a in metrics for k,v in metrics[a].items()},delta_v=dv,delta_s=dsv,
                    gross_gain_v=max(dv,0),gross_loss_v=max(-dv,0),requested_frames=len(requested),
                    active_evidence_frames=sum(n>0 for n in candidates),proposal_counts=candidates,
                    raw_proposals=sum(len(o['raw_scores']) for o in ev['observations']),
                    observed_GT_event_frames=len(observed_gt),sampling_frames=len(row['frame_ids']),
                    parameter_updates=0,backwards=0,worker_cell_seconds=x['seconds'],
                    prediction_receipt_sha256=sha(f.with_suffix('.json')))
                rows.append(r)
                diagnostics.append(dict(dataset=ds,split=row['split'],source_id=row['ordinal'],condition=cond,
                    changed_observed_frames=sum(bool(frame_changes[i]) for i in posset),
                    changed_unobserved_frames=sum(bool(frame_changes[i]) for i in range(len(frame_changes)) if i not in posset),
                    max_normalized_box_displacement=float((pred['privileged']['boxes']-pred['ordinary']['boxes']).abs().max()),
                    observed_expert_quality=expert_quality,
                    observed_GT_event_box_IoU_delta=float(np.mean([ious['privileged'][i]-ious['ordinary'][i] for i in observed_gt])) if observed_gt else None,
                    unobserved_GT_event_box_IoU_delta=float(np.mean([ious['privileged'][i]-ious['ordinary'][i] for i in unobserved_gt])) if unobserved_gt else None,
                    attention={a:{k:float(np.mean(v)) if v else None for k,v in allocation[a].items()} for a in allocation}))
        print('CPU_AUDIT',ds,'cells',len(rows),'checks',checks,flush=True)
    assert len(rows)==192 and not torch.cuda.is_initialized()
    summary=all_summaries(rows)
    gate=all(summary[d][s]['corruption']['metrics']['delta_v']['mean']>0 for d in DATASETS for s in ['search','confirm'])
    write(PUB/'ROWS.json',rows);write(PUB/'DIAGNOSTICS.json',diagnostics);write(PUB/'SUMMARY.json',summary)
    write(PUB/'DECISION.json',dict(status='P0_positive_conditional' if gate else 'P0_NO_GO_for_this_fixed_intervention',
        P0_gate_pass=gate,method_promoted=False,OPD_started=False,LN_consolidation_started=False,
        scope='This alpha1/tau1 half-box field and frozen final spatial decoder only; no universal impossibility conclusion',
        criterion='All four corrupt development/confirmation dataset means positive, predeclared before GT',
        continue_episodic_only_after_P0_audit=gate))
    write(PUB/'COST.json',dict(cells=192,new_DINO=seal['new_DINO'],DINO_cap=768,backwards=0,
        parameter_updates=0,full_parity_inputs=2,all_other_frozen_prefixes_hash_reused=True,
        GPU_worker_seconds=seal['seconds'],CPU_score_audit_seconds=time.time()-tick,
        datasets=seal['costs'],wall_time_includes_loading_IO=True,not_pure_GPU_kernel_time=True))
    write(PUB/'PROTOCOL_BINDING.json',dict(runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        global_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        defaults=verify()['defaults'],expert_checkpoint_sha256=verify()['expert_checkpoint_sha256'],
        datasets={ds:dict(configuration=read(BASE/ds/'PLAN.json')['configuration'],
            checkpoint_state_sha256=read(BASE/ds/'PLAN.json')['checkpoint_state_sha256'],
            sources=16,search_sources=8,confirm_sources=8,cells=96,conditions=read(BASE/ds/'PLAN.json')['conditions']) for ds in DATASETS},
        historical_exposure=True,expert_schedule='every cell, max four native-interval sampled frames',
        native_time_fixed=True,final_six_spatial_layers_only=True,parameter_updates=0,
        worker_GT_read=False,score_GT_read=True,no_private_box_attention_or_media_export=True))
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,cells=192,
        max_attention_CPU_error=maxattn,max_bias_CPU_error=maxbias,max_official_metric_error=maxmetric,
        global_barrier_verified=True,all_native_predictions_bitwise=True,all_weights_frozen=True,
        temporal_readout_unchanged=True,CPU_only=True,seconds=time.time()-tick))
    write(PUB/'PUBLIC_AUDIT.json',public_check(PUB))
    status(BASE/'STATUS.json',dict(status='CPU_scored_audited_pending_visual_publication',cells=192,
        P0_gate_pass=gate,GT_read=True,time=time.time()))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','public']);p.add_argument('directory',nargs='?');args=p.parse_args()
    if args.stage=='run':run()
    else:print(public_check(args.directory or PUB))
