"""CPU paired observation/selection/execution diagnosis after both update seals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.tastvg_oracle_event5_common_v1 import *
import numpy as np,collections
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
from vg_tta.tastvg_paper48_metrics_v1 import xyxy
from vg_tta.tastvg_current_correction_views_v1 import select

def summarize(rows):
    fields=['A_GT','A_actual','spatial_oracle_GT','spatial_oracle_A']
    fields += [f'{b}_{kind}_{interval}' for b in BRANCHES for kind in ['select','temp'] for interval in ['GT','A']]
    fields += ['delta_'+f for f in fields if '_select_' in f or '_temp_' in f]
    fields += [f'GT_minus_{b}_{kind}_{interval}' for b in ['R','U2'] for kind in ['select','temp'] for interval in ['GT','A']]
    result={}
    for group in ['corruption','clean']:
        result[group]={}
        for sample in ['eligible_matched','all_scheduled_with_noop_unsupported']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and (sample!='eligible_matched' or r['eligible'])]
            z=source_summary(rr,fields);z['quality']={};z['execution']={}
            for b in BRANCHES:
                ee=[r['evidence'][b] for r in rr if b in r['evidence']]
                z['quality'][b]=dict(requests=len(ee),empty_requests=sum(e['valid_frames']==0 for e in ee),
                    empty_observed_frames=sum(e['empty_observed_frames'] for e in ee),observed_frames=sum(e['observed_frames'] for e in ee),
                    event_observed_frames=sum(e['event_observed_frames'] for e in ee),event_valid_frames=sum(e['event_valid_frames'] for e in ee),
                    event_scorable_valid_frames=sum(e['event_scorable_valid_frames'] for e in ee),
                    event_annotation_missing=sum(e['event_annotation_missing'] for e in ee),
                    event_box_GT_IoU_valid_weighted=(sum(e['event_iou_sum'] for e in ee)/sum(e['event_scorable_valid_frames'] for e in ee)
                        if sum(e['event_scorable_valid_frames'] for e in ee) else None),
                    event_box_GT_IoU_observed_zero_empty=(sum(e['event_iou_sum'] for e in ee)/sum(e['event_scorable_observed_frames'] for e in ee)
                        if sum(e['event_scorable_observed_frames'] for e in ee) else None))
                z['execution'][b]={}
                for interval,base in [('GT','A_GT'),('A','A_actual')]:
                    k=f'delta_{b}_temp_{interval}';sel=f'delta_{b}_select_{interval}'
                    def val(r,name):return r[name]
                    z['execution'][b][interval]=dict(improved=sum(r[k]>1e-12 for r in rr),harmed=sum(r[k]<-1e-12 for r in rr),
                        serious_harm_gt5pp=sum(r[k]<-.05 for r in rr),selected_better_but_update_harm=sum(r[sel]>1e-12 and r[k]<-1e-12 for r in rr),
                        unique_preference_selected_better_update_harm=sum(r[sel]>1e-12 and r[k]<-1e-12 and r['evidence'].get(b,{}).get('top_tie_count',0)==1 for r in rr),
                        gross_gain_pp=float(np.mean([max(r[k],0) for r in rr])*100) if rr else None,
                        gross_loss_pp=float(np.mean([max(-r[k],0) for r in rr])*100) if rr else None,
                        correctness={str(t):dict(correct_before=sum(r[base]>t for r in rr),
                            destroyed=sum(r[base]>t>=r[f'{b}_temp_{interval}'] for r in rr),
                            rescued=sum(r[f'{b}_temp_{interval}']>t>=r[base] for r in rr)) for t in [.3,.5]})
            result[group][sample]=z
    return result

def run():
    import torch
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_ur_write_decomposition_v1 import sgd
    from scripts.audit_tastvg_ur_write_v1 import independent_rewards,independent_geometry,logsoftmax,ranks
    torch.set_num_threads(2);verified(include_GT=True);tick=time.monotonic();checks=collections.Counter();maxerr=0.
    gb=read(BASE/'GLOBAL_INTERVENTION_BARRIER.json');assert gb['status']=='sealed' and gb['donors']==288
    for ds in DATASETS:
        f=BASE/ds/'updates/PREDICTION_BARRIER.json';assert sha(f)==gb['datasets'][ds]
        for q,h in read(f)['files'].items():assert sha(f.parent/q)==h
    if not (BASE/'POST_INTERVENTION_GT_EXPOSURE.json').exists():
        write(BASE/'POST_INTERVENTION_GT_EXPOSURE.json',dict(scope='sealed query-only GT-event outputs, CPU metrics and evidence quality',
            GT_routing_plan_preceded_inference=True,raw_GT_not_opened_by_workers=True,barrier_sha256=sha(BASE/'GLOBAL_INTERVENTION_BARRIER.json'),time=time.time()))
    else:assert read(BASE/'POST_INTERVENTION_GT_EXPOSURE.json')['barrier_sha256']==sha(BASE/'GLOBAL_INTERVENTION_BARRIER.json')
    events=read(BASE/'EVENT_PLAN.json')['events'];total=0
    for ds in DATASETS:
        p=plan(ds);cfg=p['params']
        for split in SPLITS:
            labels=read(POOL/ds/f'GT_LABELS_{split}.json');rows=[]
            for c in [c for c in events if c['dataset']==ds and c['split']==split]:
                x=cached_payload(c);y=cached_payload(c,'round2');old=old_a(c)
                new=checked(BASE/ds/'updates/predictions'/f'{prefix(c)}.pt')
                assert new['persistent_unchanged'] and new['pre_sha']==old['pre_sha'] and new['persistent_post_sha']==old['post_sha']
                assert sha(payload_path(c))==new['cached_probes_sha256'] and torch.equal(new['A']['boxes'],x['A']['boxes'])
                row=p['rows'][c['parent']];g=labels[str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()};span=g['span'];ia=x['A']['physical_interval']
                r=dict(dataset=ds,split=split,parent=c['parent'],source_id=c['parent'],condition=c['condition'],order=c['order'],
                    arrival=c['arrival'],expert_scheduled=True,eligible=c['eligible'],available_event_sample_frames=c['available'],evidence={})
                tubes=[DenseTube(z['boxes'],row,truth,span,ds=='hc2') for z in x['probes']]
                probesGT=[s.score(span)['v'] for s in tubes];probesA=[s.score(ia)['v'] for s in tubes]
                r.update(A_GT=probesGT[0],A_actual=probesA[0],spatial_oracle_GT=max(probesGT),spatial_oracle_A=max(probesA),
                    probe_GT_v=probesGT,probe_A_v=probesA)
                upost=old['update_steps'][0].get('post_prediction')
                if upost is None:
                    uf=BASE/ds/'cached_U_readout/predictions'/f'{prefix(c)}.pt';uv=checked(uf)
                    assert uv['cached_first_post_state_sha256']==state_hash(old['update_steps'][0]['post_state'])
                    upost=uv['prediction'];checks['cached_U_omitted_post_readouts']+=1
                temp={'U':upost,'R':x['predictions']['R_temp'],'U2':y['predictions']['U2_temp']}
                sels={'U':x['selections']['U'],'R':x['selections']['R'],'U2':y['selections']['U2']}
                if c['eligible']:
                    er=new['evidence'];assert sha(ROOT/er['cache'])==er['cache_sha256'];ev=load(ROOT/er['cache'])
                    np.testing.assert_array_equal(ev['positions'],c['positions']);assert len(set(ev['positions']))==5
                    assert all(span[0]<=row['frame_ids'][i]<span[1] for i in ev['positions'])
                    rew=independent_rewards(x['probes'],ev)
                    assert (rew is None)==(new['rewards'] is None)
                    if rew is not None:
                        np.testing.assert_allclose(rew,new['rewards'],rtol=0,atol=1e-12);meta=new['metadata']
                        d=independent_geometry(x['A']['boxes'],[z['boxes'] for z in x['probes']],new['coefficients'])
                        lp=logsoftmax(-d/cfg['student_temperature']);lq=logsoftmax(-ranks(rew)/cfg['teacher_temperature'])
                        np.testing.assert_allclose(d,meta['distances'],atol=2e-5,rtol=2e-5)
                        np.testing.assert_allclose(np.exp(lp),meta['p'],atol=2e-6,rtol=2e-5)
                        np.testing.assert_allclose(np.exp(lq),meta['q'],atol=2e-7,rtol=1e-6)
                        np.testing.assert_allclose(np.sum(np.exp(lp)*(lp-lq)),meta['loss'],atol=2e-5,rtol=2e-5)
                        checks['independent_loss_scalars']+=28
                    assert select(rew)==new['selection'];assert new['inner_steps']==1
                    expected=sgd(x['pre_state'],new['gradients'],cfg['lr']) if rew is not None else x['pre_state']
                    assert state_hash(expected)==new['temporary_sha']==state_hash(new['temporary_state'])
                    for n,v in x['pre_state'].items():
                        np.testing.assert_allclose((v.double().numpy()-cfg['lr']*new['gradients'][n].double().numpy()).astype(np.float32),
                            new['temporary_state'][n],atol=1e-6,rtol=2e-6);checks['independent_SGD_coordinates']+=v.numel()
                    temp['GT_event']=new['post_prediction'];sels['GT_event']=new['selection']
                else:temp['GT_event']=x['A'];sels['GT_event']=select(None)
                for branch in BRANCHES:
                    if branch=='GT_event':
                        evid,receipt=(ev,new['evidence']) if c['eligible'] else (None,None)
                    else:evid,receipt=cached_evidence(c,branch)
                    sel=sels[branch];index=sel['selected']
                    if evid is not None:
                        rr=independent_rewards(x['probes'],evid);assert sel==select(rr)
                        if branch!='GT_event':np.testing.assert_allclose(rr,x['rewards'][branch] if branch in x['rewards'] else y['rewards'][branch],atol=1e-12,rtol=0) if rr is not None else None
                        pos=evid['positions'];valid=np.asarray(evid['valid'],bool);ebox=xyxy(evid['boxes'],row['input']['width'],row['input']['height'])
                        ep=[i for i in pos if span[0]<=row['frame_ids'][i]<span[1]];sc=[i for i in ep if row['frame_ids'][i] in truth];vp=[i for i in sc if valid[i]]
                        quality=[float(box_iou(ebox[i],truth[row['frame_ids'][i]])) for i in vp]
                        r['evidence'][branch]=dict(valid_frames=int(valid.sum()),observed_frames=len(pos),
                            empty_observed_frames=sum(not valid[i] for i in pos),event_observed_frames=len(ep),
                            event_valid_frames=int(sum(valid[i] for i in ep)),event_scorable_observed_frames=len(sc),
                            event_scorable_valid_frames=len(vp),event_annotation_missing=len(ep)-len(sc),
                            event_iou_sum=sum(quality),event_IoU_valid_mean=float(np.mean(quality)) if quality else None,
                            rewards=np.asarray(rr).tolist() if rr is not None else None,selected=index,top_tie_count=len(sel['top_ties']),
                            selection_reason=sel['reason'],new_call=receipt.get('new_call',False),
                            input_sha256=receipt['input_sha256'])
                    chosen=x['probes'][index];after=temp[branch];after_score=DenseTube(after['boxes'],row,truth,span,ds=='hc2')
                    for interval,name,baseline in [(span,'GT',r['A_GT']),(ia,'A',r['A_actual'])]:
                        vs=tubes[index].score(interval);vt=after_score.score(interval)
                        r[f'{branch}_select_{name}']=vs['v'];r[f'{branch}_temp_{name}']=vt['v']
                        r[f'delta_{branch}_select_{name}']=vs['v']-baseline;r[f'delta_{branch}_temp_{name}']=vt['v']-baseline
                        for boxes,ref in [(chosen['boxes'],vs),(after['boxes'],vt)]:
                            m=official(boxes,row,truth,span,interval,ds)
                            for k in ['v','t','s']:
                                error=abs(m[k]-ref[k]);assert error<1e-10;maxerr=max(maxerr,error);checks['official_metric_scalars']+=1
                    checks['observation_branches']+=1
                for b in ['R','U2']:
                    for kind in ['select','temp']:
                        for interval in ['GT','A']:
                            r[f'GT_minus_{b}_{kind}_{interval}']=r[f'GT_event_{kind}_{interval}']-r[f'{b}_{kind}_{interval}']
                if c['eligible']:
                    r['GT_event_loss']=new['metadata']['loss'] if new['metadata'] else None
                    r['GT_event_displacement']=new['displacement']
                rows.append(r);total+=1
                if total%24==0:print('CPU_EVENT_SCORE',total,288,flush=True)
            out=PUBLIC/'experiment2'/split/ds
            write(out/'INTERVENTION_ROWS.json',rows);write(out/'SUMMARY.json',summarize(rows))
            write(out/'CASES.json',{f:dict(positive=sorted(rows,key=lambda r:-r[f])[:5],negative=sorted(rows,key=lambda r:r[f])[:5])
                for f in ['GT_minus_R_select_GT','GT_minus_R_temp_GT','GT_minus_U2_select_GT','GT_minus_U2_temp_GT','delta_GT_event_temp_GT','delta_GT_event_temp_A']})
    assert total==288 and not torch.cuda.is_initialized();verified(include_GT=True)
    audit=dict(status='pass',donors=288,eligible=sum(c['eligible'] for c in events),checks=dict(checks),
        max_official_error=maxerr,CUDA_initialized=False,post_global_intervention_seal=True,
        worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'EXPERIMENT2_ROOT_AUDIT.json',audit);write(PUBLIC/'experiment2/ROOT_AUDIT.json',audit)
    status(BASE/'STATUS.json',dict(status='completed_pending_report_publication',time=time.time()))
    archive('GT-event观察与一步临时更新已全部封存并CPU评分，官方指标与梯度/SGD链核验通过，进入报告与公开收尾')
    print('EXPERIMENT2_COMPLETE',audit,flush=True)
if __name__=='__main__':run()
