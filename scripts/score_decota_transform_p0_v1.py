"""Post-global-seal CPU evaluation; fixed decisions never enter input inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_transform_common_v1 import *
from scripts.score_decota_optimizer_posterior_v1 import stats
from vg_tta.decota_transform_p0_v1 import *
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
import numpy as np

FIELDS=['v','t','s','native_v','native_t','delta_v','delta_t','delta_full','delta_event','before_v','correction_v']

def groups(rows,fields):
    out={}
    for ds in DATASETS:
        out[ds]={}
        for split in ['search','confirm']:
            out[ds][split]={}
            for arm in sorted({r['arm'] for r in rows}):
                rr=[r for r in rows if r['dataset']==ds and r['split']==split and r['arm']==arm]
                gg={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean']}
                gg.update({o:[r for r in rr if r['condition']!='clean' and r['order']==o] for o in ['order1','order2']})
                gg.update({c:[r for r in rr if r['condition']==c] for c in sorted({r['condition'] for r in rr})})
                gg['non_directional_corruption']=[r for r in rr if r['condition']!='clean' and not r['directional']]
                ss={k:stats(v,fields) for k,v in gg.items()}
                delta='correction_v' if 'correction_v' in fields else 'delta_v'
                for k,rrr in gg.items():
                    ss[k]['tails']=dict(harm_gt5pp=sum(r[delta]<-.05 for r in rrr),harm_gt20pp=sum(r[delta]<-.2 for r in rrr),
                        gain_gt5pp=sum(r[delta]>.05 for r in rrr),zero_change=sum(r[delta]==0 for r in rrr))
                out[ds][split][arm]=ss
    return out

def association(rows,score='delta_event',target='correction_v'):
    if not rows:return dict(cells=0,sources=0,correlation=None,AUC=None)
    ids=sorted({r['source_id'] for r in rows});ix={s:i for i,s in enumerate(ids)}
    ri=np.array([ix[r['source_id']] for r in rows]);n=len(ids)
    xx=np.array([r[score] for r in rows]);yy=np.array([r[target] for r in rows]);counts=np.bincount(ri,minlength=n)
    active=yy!=0;helped=yy>0
    order=np.argsort(xx[active],kind='stable');xs=xx[active][order]
    starts=np.flatnonzero(np.r_[True,np.diff(xs)!=0]) if len(xs) else np.array([],int)
    def calc(node):
        w=node[:,ri]/counts[ri];sw=w.sum(1)
        mx=(w*xx).sum(1)/sw;my=(w*yy).sum(1)/sw
        dx=xx[None,:]-mx[:,None];dy=yy[None,:]-my[:,None]
        vx=(w*dx*dx).sum(1)/sw;vy=(w*dy*dy).sum(1)/sw
        cov=(w*dx*dy).sum(1)/sw
        corr=np.divide(cov,np.sqrt(np.maximum(vx*vy,0)),out=np.full(len(w),np.nan),where=(vx>1e-24)&(vy>1e-24))
        auc=np.full(len(w),np.nan)
        if len(starts):
            ww=w[:,active][:,order];h=helped[active][order]
            pos=np.add.reduceat(ww*h,starts,axis=1);neg=np.add.reduceat(ww*~h,starts,axis=1)
            num=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1);den=pos.sum(1)*neg.sum(1)
            auc=np.divide(num,den,out=auc,where=den>0)
        return corr,auc
    point=calc(np.ones((1,n)));rng=np.random.default_rng(20261005);boots=[[],[]]
    for _ in range(40):
        c=rng.multinomial(n,np.full(n,1/n),size=250);v=calc(c)
        for j in range(2):boots[j].extend(v[j][np.isfinite(v[j])].tolist())
    def val(j):
        return dict(mean=float(point[j][0]) if np.isfinite(point[j][0]) else None,
                    ci95=np.quantile(boots[j],[.025,.975]).tolist() if boots[j] else None,valid_draws=len(boots[j]))
    return dict(cells=len(rows),sources=n,correlation=val(0),AUC=val(1),zero_utility=int((yy==0).sum()),
        help=int((yy>0).sum()),harm=int((yy<0).sum()),ROC_excludes_zero_utility=True,source_equal=True,
        source_ids=ids,draws=10000,seed=20261005,no_supervised_fit=True,score=score,target=target)

def associations(rows):
    out={}
    for ds in DATASETS:
        out[ds]={}
        for split in ['search','confirm']:
            out[ds][split]={}
            for arm in ['episodic','online100']:
                rr=[r for r in rows if r['dataset']==ds and r['split']==split and r['arm']==arm]
                gg={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean'],
                    'non_directional_corruption':[r for r in rr if r['condition']!='clean' and not r['directional']]}
                out[ds][split][arm]={k:{f:association(v,f) for f in ['delta_event','delta_full']} for k,v in gg.items()}
    return out

def decisions(ts,ass):
    tg={};sg={}
    for ds in DATASETS:
        a=ts[ds]['confirm']['consensus_episodic']['corruption']['metrics'];b=ts[ds]['search']['consensus_episodic']['corruption']['metrics']
        tg[ds]=all(a[k]['ci95'][0]>0 and b[k]['mean']>0 for k in ['delta_t','delta_v'])
        good=[]
        for stream in ['episodic','online100']:
            aa=ass[ds]['confirm'][stream]['corruption']['delta_event'];bb=ass[ds]['search'][stream]['corruption']['delta_event']
            good.append(all(aa[k]['ci95'] is not None and aa[k]['ci95'][0]>threshold and bb[k]['mean'] is not None and bb[k]['mean']>threshold for k,threshold in [('correlation',0),('AUC',.5)]))
        sg[ds]=all(good)
    return dict(temporal=dict(qualified=all(tg.values()),dataset_gates=tg),spatial=dict(qualified=all(sg.values()),dataset_gates=sg),
        conditional_temporal514='pending_separate_lock' if all(tg.values()) else 'skipped_P0_not_qualified',
        conditional_acceptance='pending_separate_lock' if all(sg.values()) else 'skipped_P0_not_qualified',
        new_backwards=0,new_expert=0,production_promoted=False,working_point_retained=not(all(tg.values()) or all(sg.values())),
        future_benchmarks_started=False)

def run():
    import torch
    torch.set_num_threads(4);verify();bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['status']=='sealed' and not bar['GT_read'] and len(bar['files'])==576
    for f,h in bar['files'].items():assert sha(BASE/f)==h
    pins=['scripts/score_decota_transform_p0_v1.py','vg_tta/decota_transform_p0_v1.py','vg_tta/tastvg_oracle_event5_v1.py','scripts/score_decota_optimizer_posterior_v1.py']
    labels={str((c1.POOL/ds/f'GT_LABELS_{s}.json').relative_to(ROOT)):sha(c1.POOL/ds/f'GT_LABELS_{s}.json') for ds in DATASETS for s in ['search','confirm']}
    write(BASE/'CPU_LOCK.json',dict(pins={f:sha(ROOT/f) for f in pins},labels=labels,barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
    write(BASE/'GT_EXPOSURE.json',dict(time=time.time(),barrier_time=bar['time'],barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),GT_online=False))
    labs={(ds,sp):read(c1.POOL/ds/f'GT_LABELS_{sp}.json') for ds in DATASETS for sp in ['search','confirm']}
    plans={ds:read(BASE/ds/'PLAN.json') for ds in DATASETS};tr=[];sr=[];cost=bar['cost'];checks=0;cache={};tick=time.time()
    for cell in read(BASE/'COHORT.json')['cells']:
        ds,split,parent,cond,order=cell['dataset'],cell['split'],cell['parent'],cell['condition'],cell['order'];row=plans[ds]['rows'][parent]
        key=(ds,parent,cond)
        if key not in cache:cache[key]=checked(BASE/'predictions'/ds/cond/f'{parent:05}.pt')
        x=cache[key];lab=labs[ds,split][str(parent)];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span'];native=x['native_interval']
        label=dict(dataset=ds,split=split,condition=cond,order=order,arrival=cell['arrival'],source_id=parent,directional=x['directional'])
        for stream in ['episodic','online100']:
            ref=old.checked(ROOT/cell['references'][stream]);b=ref['after'].numpy();d=DenseTube(b,row,truth,span,clip=ds=='hc2')
            nat=d.score(native)
            for name,iv in [('consensus',x['consensus_interval']),('shift',x['temporal_views']['shift']['mapped']['interval']),('crop',x['temporal_views']['crop']['mapped']['interval'])]:
                m=d.score(iv);off=official(b,row,truth,span,iv,ds);assert max(abs(m[k]-off[k]) for k in m)<2e-12;checks+=3
                tr.append(dict(**label,arm=name+'_'+stream,**m,native_v=nat['v'],native_t=nat['t'],delta_v=m['v']-nat['v'],delta_t=m['t']-nat['t'],
                    native_interval=native,predicted_interval=iv,crop_effective=x['temporal_views']['crop']['spec']['effective'],
                    shift_clipped=x['temporal_views']['shift']['mapped']['clipped'],shift_invalid=x['temporal_views']['shift']['mapped']['invalid']))
            before=official(ref['before'].numpy(),row,truth,span,native,ds);after=official(b,row,truth,span,native,ds);assert abs(nat['v']-after['v'])<2e-12;checks+=4
            kk=stream+'_'+order;cc=x['consistency'][kk]
            sr.append(dict(**label,arm=stream,before_v=before['v'],v=after['v'],t=after['t'],s=after['s'],correction_v=after['v']-before['v'],
                delta_event=cc['delta_event'],delta_full=cc['delta_full'],C_before=cc['before'],C_after=cc['after'],
                selected_step=ref['fit']['selected_step'],loss_gain=ref['fit']['path'][0]['loss']-ref['fit']['path'][ref['fit']['selected_step']]['loss'],
                state_sha256=x['spatial_views']['flip']['results'][kk+'_after']['state_sha256'],payload_sha256=sha(BASE/'predictions'/ds/cond/f'{parent:05}.pt'),
                prestate_reference_sha256=sha(ROOT/cell['references'][stream])))
        # Spatially-native control isolates pure temporal readout from correction strength.
        ref=old.checked(ROOT/cell['references']['episodic']);b=ref['native']['boxes'].numpy();d=DenseTube(b,row,truth,span,clip=ds=='hc2');nat=d.score(native);m=d.score(x['consensus_interval'])
        tr.append(dict(**label,arm='consensus_frozen',**m,native_v=nat['v'],native_t=nat['t'],delta_v=m['v']-nat['v'],delta_t=m['t']-nat['t'],
            native_interval=native,predicted_interval=x['consensus_interval'],crop_effective=x['temporal_views']['crop']['spec']['effective'],shift_clipped=x['temporal_views']['shift']['mapped']['clipped'],shift_invalid=x['temporal_views']['shift']['mapped']['invalid']))
    assert len(tr)==8064 and len(sr)==2304
    ts=groups(tr,['v','t','s','native_v','native_t','delta_v','delta_t']);ss=groups(sr,['v','before_v','correction_v','delta_event','delta_full']);ass=associations(sr);dec=decisions(ts,ass)
    configuration=dict(datasets={ds:dict(source_checkpoint=plans[ds]['configuration']['checkpoint'],
        checkpoint_sha256=plans[ds]['configuration']['checkpoint_sha256'],source_dataset=plans[ds]['configuration']['source_dataset'],
        checkpoint_state_sha256=plans[ds]['checkpoint_state_sha256'],development_sources=32,confirmation_sources=16,
        development_queries=32,confirmation_queries=16,orders=2,conditions=plans[ds]['conditions'],
        frame_counts=[len(row['frame_ids']) for row in plans[ds]['rows']]) for ds in DATASETS},
        original_input_sampling='Paper48 unchanged',corruption_pixels_matched=True,expert_rate=1.,new_expert_calls=0,new_parameter_updates=0,
        backbone_capture_precision='official AMP float16 prefix',private_interface_reconstruction='source encoder norm/native decoder float32, original parity verified',
        spatial_parameters=1792,Adam_lr=.03,old_steps=10,old_state_selector='first own-loss minimum step0..10',
        persistent_update='old LN delta1/16; no new write',temporal_parameters_updated=0,historical_exposure=True)
    for name,value in [('temporal/ROWS.json',tr),('spatial/ROWS.json',sr),('temporal/SUMMARY.json',ts),('spatial/SUMMARY.json',ss),('spatial/ASSOCIATIONS.json',ass),('DECISION.json',dec),('COST.json',cost),('CONFIGURATION.json',configuration)]:write(PUB/name,value)
    counts={ds:dict(unique_inputs=288,directional_unique=sum(x['directional'] for k,x in cache.items() if k[0]==ds),crop_noop=sum(not x['temporal_views']['crop']['spec']['effective'] for k,x in cache.items() if k[0]==ds),
        shift_clipped=sum(x['temporal_views']['shift']['mapped']['clipped'] for k,x in cache.items() if k[0]==ds),shift_invalid=sum(x['temporal_views']['shift']['mapped']['invalid'] for k,x in cache.items() if k[0]==ds)) for ds in DATASETS}
    gt_crop={}
    for ds in DATASETS:
        panels={}
        for sp in ['search','confirm']:
            q=[]
            for row in plans[ds]['rows']:
                parent=row['ordinal']
                if str(parent) not in labs[ds,sp]:continue
                span=labs[ds,sp][str(parent)]['span']
                for cond in plans[ds]['conditions']:
                    x=cache[ds,parent,cond];s=x['temporal_views']['crop']['spec'];ids=row['frame_ids'];cover=[ids[s['a']],ids[s['b']-1]+1]
                    q.append(dict(condition=cond,source_id=parent,GT_event_contained=cover[0]<=span[0] and cover[1]>=span[1],
                                  original_GT_event_contained=ids[0]<=span[0] and ids[-1]+1>=span[1],
                                  effective=s['effective'],GT_event_overlap_fraction=max(0,min(cover[1],span[1])-max(cover[0],span[0]))/(span[1]-span[0])))
            panels[sp]=dict(inputs=len(q),event_not_fully_contained=sum(not z['GT_event_contained'] for z in q),effective_crop_inputs=sum(z['effective'] for z in q),
                            event_not_contained_effective=sum(z['effective'] and not z['GT_event_contained'] for z in q),
                            original_sample_support_incomplete=sum(not z['original_GT_event_contained'] for z in q),
                            newly_cut_GT_event=sum(z['original_GT_event_contained'] and not z['GT_event_contained'] for z in q),GT_diagnostic_only=True)
        gt_crop[ds]=panels
    write(PUB/'COVERAGE.json',dict(unique_inputs=576,logical_cells=1152,temporal_rows=len(tr),spatial_rows=len(sr),dataset_counts=counts,historical_exposure=True,
        crop_actual_GT_support_diagnostic=gt_crop,
        full_stream_TTA_not_rerun=True,new_optimization=False,GT_after_global_seal=True))
    write(PUB/'SCORING_AUDIT.json',dict(status='pass',official_dense_checks=checks,seconds=time.time()-tick,GT_after_global_seal=True,
        inputs_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),parameters_unchanged=True))
    write(BASE/'DECISION.json',dec);status(BASE/'STATUS.json',dict(status='scored_pending_root_audit',time=time.time(),decision=dec))
    archive('全部真实预测封存后CPU dense/源聚类统计已完成；时间资格='+str(dec['temporal']['qualified'])+'、空间资格='+str(dec['spatial']['qualified'])+'，等待独立审计')
    print('TRANSFORM_DECISION',dec,flush=True)

if __name__=='__main__':run()
