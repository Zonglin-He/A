"""Post-global-seal CPU qualification with an independent numerical readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from tastvg_reference_common_v1 import *
import numpy as np,collections
from vg_tta.tastvg_reference_selection_v1 import student_frames,decision,pairwise

def summarize(rows):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    fields=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ['cell','source_id','parent','arrival']]
    out={}
    for group in ['corruption','clean']:
        rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')]
        z=dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),metrics={},counts={})
        for k in fields:
            take=[r for r in rr if r[k] is not None]
            if take:
                a=source_summary(take,[k]);z['metrics'][k]={**a['metrics'][k], 'available_cells':len(take),'available_sources':a['sources']}
        for strategy in ['Uniform','Routed']:
            z['counts'][strategy]=dict(empty_arrivals=sum(r[strategy+'_empty'] for r in rr),no_event_valid=sum(r[strategy+'_no_event_valid'] for r in rr),
                no_event_valid_nonempty=sum(r[strategy+'_no_event_valid'] and not r[strategy+'_empty'] for r in rr),
                nonempty_arrivals=sum(not r[strategy+'_empty'] for r in rr),valid_frames=sum(r[strategy+'_valid_frames'] for r in rr))
            for scope in ['fixed','joint']:
                a=z['counts'][strategy][scope]={}
                for threshold in [.3,.5]:
                    a[str(threshold)]=dict(support_correct=sum(max(r[scope+'_utilities'])>threshold for r in rr),
                        missed_correct=sum(max(r[scope+'_utilities'])>threshold and r[strategy+'_'+scope+'_v']<=threshold for r in rr),
                        no_correct_support=sum(max(r[scope+'_utilities'])<=threshold for r in rr),
                        native_correct_destroyed=sum(r[scope+'_utilities'][0]>threshold and r[strategy+'_'+scope+'_v']<=threshold for r in rr))
        dv=np.array([r['delta_fixed_v'] for r in rr]);z.update(cell_gross_gain_pp=float(np.maximum(dv,0).mean()*100),cell_gross_loss_pp=float(-np.minimum(dv,0).mean()*100),
            changed_selection=sum(r['Routed_selected']!=r['Uniform_selected'] for r in rr),positive_cells=int((dv>1e-12).sum()),negative_cells=int((dv< -1e-12).sum()),
            raw_quantile_duplicate_cells=sum(r['raw_quantile_unique']<5 for r in rr),actual_distinct_frames=sorted(set(r['actual_unique'] for r in rr)))
        out[group]=z
    return out

def independent_rewards(candidates,expert):
    positions=np.flatnonzero(expert['valid']);out=[]
    if not len(positions):return None
    for b in candidates:
        values=[]
        for i in positions:
            a,c=np.asarray(b[i],float),np.asarray(expert['boxes'][i],float)
            alo,ahi=a[:2]-a[2:]/2,a[:2]+a[2:]/2;clo,chi=c[:2]-c[2:]/2,c[:2]+c[2:]/2
            inter=max(0,min(ahi[0],chi[0])-max(alo[0],clo[0]))*max(0,min(ahi[1],chi[1])-max(alo[1],clo[1]))
            union=a[2]*a[3]+c[2]*c[3]-inter;values.append(inter/max(union,1e-12))
        out.append(float(np.mean(values)))
    return out

def run():
    import torch
    torch.set_num_threads(2);tick=time.monotonic();lock=verify();bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['new_specialist_calls']==62 and bar['routed_cells']==60 and not bar['GT_read']
    for ds,h in bar['datasets'].items():
        assert sha(BASE/ds/'PREDICTION_BARRIER.json')==h
        for f,v in read(BASE/ds/'PREDICTION_BARRIER.json')['files'].items():assert sha(BASE/f)==v
    from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
    from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
    from methods.decota_final_simplified_v1.tensors import state_hash
    all_summary={};checks=collections.Counter();max_error=0.;boundary=collections.Counter()
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');old=read(ROOT/p['row_plan']);label=lock['diagnostic_labels'][ds]
        assert sha(ROOT/label['path'])==label['sha256']
        write(BASE/ds/'GT_EXPOSURE.json',dict(time=time.time(),barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),historical_exposure=True,online_GT=False,labels_sha256=label['sha256']))
        gt=read(ROOT/label['path']);metric=DenseMetric() if ds=='vidstg' else HC2DenseMetric();rows=[]
        for c in p['cells']:
            row=old['rows'][c['parent']];x=load(ROOT/c['payload']);st=x['update_steps'][0]
            assert state_hash(x['pre_state'])==c['pre_state_sha256'];checks['sealed_A_state']+=1
            b=[z['prediction']['boxes'] for z in st['candidates']];assert len(b)==9 and np.array_equal(b[0],x['slow']['boxes'])
            route=student_frames(x['temporal']['candidates'],row['frame_ids']);assert route==c['router'];checks['router']+=1
            # Independent interval votes and literal CDF traversal.
            w=[sum(a['physical_interval'][0]<=fid<a['physical_interval'][1] for a in x['temporal']['candidates'])/len(x['temporal']['candidates']) for fid in row['frame_ids']]
            np.testing.assert_array_equal(w,route['weights'])
            q=[]
            for quant in [.1,.3,.5,.7,.9]:
                running=0.
                for i,v in enumerate(w):
                    running+=v
                    if running>=quant*sum(w)-1e-14:q.append(i);break
            assert q==route['raw_quantiles'];checks['independent_CDF_quantiles']+=5
            lab=gt[str(c['parent'])];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span'];ids=np.asarray(row['frame_ids']);clip=ds=='hc2'
            event=(ids>=span[0])&(ids<span[1]);scored=np.array([int(i) in truth for i in ids]);boundary[ds]+=int(not np.array_equal(event,scored))
            utilities={'fixed':[],'joint':[]}
            for i,bb in enumerate(b):
                fn=evaluator(bb,row,truth,span,clip);pix=xyxy(bb,row['input']['width'],row['input']['height']);pix=np.maximum(pix,0) if clip else pix
                for scope,idx in [('fixed',x['slow']['indices']),('joint',st['candidates'][i]['prediction']['indices'])]:
                    result=fn(idx);interval=[int(ids[idx[0]]),int(ids[idx[1]])+1]
                    a=metric(pix,ids,interval,truth,span);d=dense_official_metrics(pix,ids,interval,truth,span)
                    for name,code in [('m_vIoU','v'),('m_tIoU','t')]:
                        err=max(abs(result[code]-a[name]),abs(result[code]-d[name]));assert err<1e-10;max_error=max(max_error,err);checks['independent_dense_values']+=2
                    utilities[scope].append(result['v'])
            r={k:c[k] for k in ['cell','source_id','condition','order','arrival']}
            r.update(candidate_count=9,fixed_utilities=utilities['fixed'],joint_utilities=utilities['joint'],
                pre_state_sha256=c['pre_state_sha256'],raw_quantile_unique=route['raw_unique'],actual_unique=route['actual_unique'])
            for strategy in ['Uniform','Routed']:
                if strategy=='Uniform':ev=load(ROOT/c['uniform_cache']);dc=c['uniform_decision']
                else:
                    receipt=read(BASE/ds/'receipts'/f'{c["cell"]:03}_student_routed.json');ev=load(BASE/receipt['cache']);dc=receipt['decision']
                calculated=decision(b,ev['boxes'],ev['valid']);assert calculated==dc
                rr=independent_rewards(b,ev)
                if rr is None:assert dc['rewards'] is None
                else:np.testing.assert_allclose(rr,dc['rewards'],rtol=0,atol=1e-12);checks['independent_rewards']+=9
                assert dc['selected']==(0 if rr is None else int(np.argmax(rr)));checks['selection']+=1
                pos=ev['positions'];valid=np.asarray(ev['valid'],bool);assert len(set(pos))==5 and np.flatnonzero(valid).size<=5
                r.update({strategy+'_selected':dc['selected'],strategy+'_rewards':dc['rewards'],strategy+'_empty':int(not valid.any()),
                    strategy+'_valid_frames':int(valid.sum()),strategy+'_event_hits':int(event[pos].sum()),strategy+'_event_precision':float(event[pos].mean()),
                    strategy+'_scored_precision':float(scored[pos].mean()),strategy+'_valid_event_frames':int((valid&event).sum()),
                    strategy+'_no_event_valid':int(not (valid&event).any()),strategy+'_no_scored_valid':int(not (valid&scored).any())})
                for scope,u in utilities.items():
                    pair=pairwise(dc['rewards'],u);v=u[dc['selected']]
                    for name,value in dict(v=v,gain=v-u[0],regret=max(u)-v,pairwise=pair['pairwise_accuracy'],decisive=pair['decisive_coverage'],strict_pairs=pair['strict_GT_pairs']).items():r[strategy+'_'+scope+'_'+name]=value
                    r[strategy+'_'+scope+'_pairs']=pair['pairs']
            for field in ['fixed_v','fixed_gain','fixed_regret','fixed_pairwise','fixed_decisive','joint_v','joint_gain','joint_regret','joint_pairwise','joint_decisive','event_precision','scored_precision','empty','no_event_valid','valid_frames']:
                a,d=r['Uniform_'+field],r['Routed_'+field];r['delta_'+field]=d-a if a is not None and d is not None else None
            rows.append(r)
        assert len(rows)==30 and len({r['source_id'] for r in rows})==10
        write(PUBLIC/ds/'ROWS.json',rows);z=summarize(rows);write(PUBLIC/ds/'SUMMARY.json',z);all_summary[ds]=z
        cases=sorted(rows,key=lambda r:r['delta_fixed_v']);write(PUBLIC/ds/'CASES.json',dict(negative=cases[:5],positive=cases[-5:],selection_not_used_for_cohort=True))
    write(BASE/'ROOT_READBACK.json',dict(status='pass',checks=dict(checks),max_error=max_error,HC_endpoint_differences=dict(boundary),GT_only_after_global_seal=True,
        GPU_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-tick))
    write(PUBLIC/'SUMMARY.json',all_summary);write(PUBLIC/'RESOURCES.json',read(BASE/'RESOURCES.json'));write(PUBLIC/'ROOT_READBACK.json',read(BASE/'ROOT_READBACK.json'))
    assert not torch.cuda.is_initialized();status(BASE/'STATUS.json',dict(status='scored_pending_root_report_publication',done=62,total=62,time=time.time()))
    print('SCORED60matchedcells; independent audit',dict(checks),'maxerror',max_error)
if __name__=='__main__':run()
