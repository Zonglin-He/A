"""Root v3 audit: independent multi-step arithmetic, hashes, selection and scalar closure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,csv,json,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from scripts.tastvg_extended_common_v3 import BASE,verify,read,write,sha,anchor
from scripts.validate_tastvg_extended_selection_v3 import validate
from scripts.audit_tastvg_paper48_public_v1 import check_summary
from vg_tta.tastvg_paper48_metrics_v1 import source_summary
from methods.decota_final_simplified_v1.tensors import state_hash

def persist(p,x):
    if p.exists():assert read(p)==x,p
    else:write(p,x)

def run(ds):
    p=verify(ds);b=BASE/ds;initial_cfg=anchor(p);screen=read(BASE/'PLAN.json')['screen'];trials,sel=validate(b,screen,initial_cfg)
    assert read(b/'GT_EXPOSURE_confirm.json')['time']>sel['time']
    sources=[{p['rows'][i]['source'] for i in p['splits'][s]['orders']['order1']} for s in ['search','confirm']]
    assert len(sources[0])==32 and len(sources[1])==16 and not sources[0]&sources[1]
    persist(b/'SEARCH_DESIGN.json',dict(anchor=initial_cfg,screen=screen,search_sources=32,confirmation_sources=16,historically_exposed=True,max_scheduled=100))
    tasks=[(b/t['result_dir'],t) for t in trials if t['state']=='COMPLETE' and not t.get('reused_from')]
    for name in ['anchor','selected']:
        d=b/'confirmation'/name
        if (d/'REUSE.json').exists():
            r=read(d/'REUSE.json');src=b/r['source'];assert sha(src/r['receipt'])==r['sha256'];assert read(d/'REQUEST.json')['params']==read(src/'REQUEST.json')['params']
        elif not (d/'NUMERICAL_FAILURE.json').exists():tasks.append((d,None))
    hashes={};cells=steps=sgd=checks=param_checks=0;compute=collections.Counter();directions=collections.Counter()
    for d,t in tasks:
        req=read(d/'REQUEST.json');cfg=req['params'];split=req['split'];bar=read(d/'PREDICTION_BARRIER.json');au=read(d/'AUDIT.json');score=read(d/'SCORE.json');rows=read(d/'ROWS.json');summary=read(d/'SUMMARY.json')
        assert score['params']==cfg and score['audit_sha256']==sha(d/'AUDIT.json') and score['prediction_barrier_sha256']==sha(d/'PREDICTION_BARRIER.json')
        assert bar['GT_read'] is False and bar['model_restored'] and bar['time']<score['time']
        assert au['status']=='pass' and au['max_error']==0 and au['all_predictions_before_scoring']
        assert len(rows)==len(bar['files'])==bar['cells']==au['state_links']==p['splits'][split]['total']
        if t:assert t['objective']==score['objective'] and t['params']==cfg and score['time']<sel['time']
        else:assert sel['time']<bar['time'] and cfg=={'anchor':initial_cfg,'selected':sel['params']}[d.name]
        total=collections.Counter();updated=0;initial_hash=read(d/'SUPPORT.json')['center_sha256']
        for cond in p['conditions']:
            for order,seq in p['splits'][split]['orders'].items():
                prev=initial_hash
                for at,parent in enumerate(seq):
                    f=d/'online'/cond/order/f'{at:05}.json';rec=read(f);assert bar['files'][str(f.relative_to(d))]==sha(f) and sha(f.with_suffix('.pt'))==rec['sha256']
                    x=torch.load(f.with_suffix('.pt'),map_location='cpu',weights_only=False)
                    assert x['parent']==rec['parent']==parent and x['GT_read'] is False and x['expert_scheduled']==(at%4==0)
                    assert prev==rec['pre_sha']==x['pre_sha']==state_hash(x['pre_state']);assert rec['post_sha']==x['post_sha']==state_hash(x['post_state'])
                    trace=x['update_steps'];assert bool(trace)==x['expert_scheduled'] and len(trace)<=cfg['steps'];previous=x['pre_state'];valid=0
                    for step in trace:
                        assert state_hash(previous)==step['pre_state_sha256']==state_hash(step['pre_state'])
                        u=step['update']
                        if u is None:
                            assert all(torch.equal(v,step['post_state'][n]) for n,v in step['pre_state'].items())
                        else:
                            assert u['lr']==cfg['lr'] and u['teacher_temperature']==cfg['teacher_temperature'] and u['student_temperature']==cfg['student_temperature'];assert u['candidate_targets_detached'] and u['reward_detached']
                            dist=np.asarray(u['distances'],float);rank=np.asarray(u['rank'],float);assert len(dist)==len(rank)==2*cfg['direction_count']+1
                            lp=-dist/cfg['student_temperature'];lp-=np.logaddexp.reduce(lp);lq=-rank/cfg['teacher_temperature'];lq-=np.logaddexp.reduce(lq)
                            np.testing.assert_allclose(np.exp(lp),u['p'],atol=2e-7,rtol=1e-6);np.testing.assert_allclose(np.exp(lq),u['q'],atol=2e-7,rtol=1e-6)
                            np.testing.assert_allclose(np.sum(np.exp(lp)*(lp-lq)),u['loss_before'],atol=2e-5,rtol=2e-5)
                            for name,before in step['pre_state'].items():
                                expected=(before.double().numpy()-cfg['lr']*u['gradients'][name].double().numpy()).astype(np.float32)
                                np.testing.assert_allclose(expected,step['post_state'][name],atol=1e-6,rtol=2e-6);param_checks+=before.numel()
                            valid+=1;sgd+=1
                        assert state_hash(step['post_state'])==step['post_state_sha256'];previous=step['post_state'];steps+=1
                    assert state_hash(previous)==x['post_sha'];prev=x['post_sha'];assert x['updated']==any(s['updated'] for s in trace)
                    updated+=x['updated'];n=len(trace);c=x['compute'];expected=dict(inner_steps=n,spatial_candidate_replays=n*(2*cfg['direction_count']+1),native_replays=1+max(0,n-1)+n*(2*cfg['direction_count']+1)+2*valid,backward_calls=valid,spatial_provider_calls=int(x['expert_scheduled']),temporal_provider_calls=int(x['expert_scheduled']))
                    assert c==expected;total.update(c);cells+=1
        assert dict(total)==au['compute'] and updated==au['SGD_updates'];assert total['temporal_provider_calls']==au['teacher_checks']
        compute.update(total);directions[cfg['direction_count']]+=len(rows)
        for r in rows:
            assert r['parent']==p['splits'][split]['orders'][r['order']][r['arrival']] and r['expert_scheduled']==(r['arrival']%4==0)
            if not r['expert_scheduled']:assert not r['updated'] and r['delta_m_tIoU']==0
            for key in ['m_tIoU','m_vIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:
                assert 0<=r['Frozen_'+key]<=1+1e-12 and 0<=r['Ours_'+key]<=1+1e-12
                assert abs(r['Ours_'+key]-r['Frozen_'+key]-r['delta_'+key])<1e-12;checks+=2
        for group in ['clean','corruption']:
            for sub in ['all','nonexpert','expert']:
                rr=[r for r in rows if (r['condition']=='clean')==(group=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))];checks+=check_summary(rr,summary[group][sub])
        assert score['objective']==summary['corruption']['all']['metrics']['delta_m_vIoU']['mean']
        with (d/'SCALARS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
        for name in ['REQUEST.json','AUDIT.json','SCORE.json','SUMMARY.json','SCALARS.csv','PREDICTION_BARRIER.json']:hashes[str((d/name).relative_to(b))]=sha(d/name)
        print('VERIFIED',ds,d.relative_to(b),flush=True)
    # Paired per-source contrasts for all trials relative to the exact search anchor.
    anchor_rows=read(b/'search/screen_anchor/ROWS.json');paired={}
    def contrast(a,z):
        out={}
        for group in ['clean','corruption']:
            out[group]={}
            for sub in ['all','nonexpert','expert']:
                rr=[]
                for x,y in zip(a,z):
                    assert all(x[k]==y[k] for k in ['parent','order','condition','arrival','expert_scheduled','Frozen_m_vIoU'])
                    if (x['condition']=='clean')==(group=='clean') and (sub=='all' or x['expert_scheduled']==(sub=='expert')):rr.append(dict(parent=x['parent'],order=x['order'],delta_vs_anchor=y['Ours_m_vIoU']-x['Ours_m_vIoU']))
                out[group][sub]=source_summary(rr,['delta_vs_anchor'])
        return out
    for t in trials:
        if t['state']=='COMPLETE':paired[t['tag']]=contrast(anchor_rows,read(b/t['result_dir']/'ROWS.json'))
        else:
            assert t['objective'] is None and read(b/t['result_dir']/'STATUS.json')['status']=='numerical_failure'
    persist(b/'PAIRED_VS_ANCHOR.json',paired)
    sr=b/'confirmation/selected'
    if (sr/'REUSE.json').exists():sr=b/read(sr/'REUSE.json')['source']
    if (sr/'ROWS.json').exists():cf=contrast(read(b/'confirmation/anchor/ROWS.json'),read(sr/'ROWS.json'))
    else:cf=dict(status='unavailable_numerical_failure',no_reselection=True)
    persist(b/'CONFIRMATION_PAIRED.json',cf)
    result=dict(status='pass',dataset=ds,scheduled=len(trials),failures=sum(t['state']=='FAIL' for t in trials),actual_arrivals=cells,inner_step_state_chains=steps,independent_SGD_steps=sgd,independent_parameter_scalar_checks=param_checks,scalar_checks=checks,compute=dict(compute),hashes=hashes,selection_and_allocation_verified=True,confirmation_source_disjoint=True,GT_read=False,GPU_initialized=torch.cuda.is_initialized(),time=time.time())
    assert not result['GPU_initialized'];write(b/'ROOT_READBACK.json',result);print(json.dumps({k:v for k,v in result.items() if k!='hashes'},indent=2))

if __name__=='__main__':run(sys.argv[1])
