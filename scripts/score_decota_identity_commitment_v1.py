"""Independent loss/path/Adam/state/dense audit after each global barrier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import *
from scripts.run_decota_identity_commitment_v1 import references
from scripts.score_decota_actuation_scope_v1 import path_numpy,iou,path_energy
from scripts.score_decota_optimizer_posterior_v1 import energy,stats
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import flat,check_state
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
from vg_tta.c1_enabling_tricks_v1 import QUERY
from vg_tta.decota_actuation_scope_v1 import commit_state
from methods.decota_final_simplified_v1.tensors import state_hash
import numpy as np,torch
from scipy.special import logsumexp
ARMS=['top1','marginal','map_contrastive']
FIELDS=['v','t','s','vs_frozen_v','vs_before_v','vs_top1_v','gross_gain','gross_loss']

def observation_quality(ex,before,row,truth):
    ff,paths,lp,_=path_numpy(ex,before.numpy(),row['frame_ids']);ix=np.asarray(paths);vals=[];used=[]
    w,h=row['input']['width'],row['input']['height']
    for j,(pos,ev,_) in enumerate(ff):
        fid=row['frame_ids'][pos]
        if fid not in truth:continue
        a=np.asarray(truth[fid],float);gt=np.r_[(a[:2]+a[2:])/2/[w,h],(a[2:]-a[:2])/[w,h]]
        vals.append(np.array([iou(b,gt) for b in ev])[ix[:,j]]);used.append(j)
    z=dict(valid_observations=len(ff),GT_scored_observations=len(used),outside_GT_scored_support=len(ff)-len(used),
        path_count=len(paths),identity_accuracy_not_measured=True)
    if vals:
        quality=np.mean(vals,axis=0);z.update(MAP_observation_IoU=float(quality[int(lp.argmax())]),best_observation_IoU=float(quality.max()),posterior_expected_observation_IoU=float(np.exp(lp)@quality))
    return z

def objective(boxes,ex,before,ids,arm,setup=None):
    if arm=='top1':return energy(boxes,ex,'top1')
    ff,paths,lp,pa=path_numpy(ex,before,ids) if setup is None else setup
    if arm=='marginal':return path_energy(boxes,ff,paths,lp,'track')
    if len(paths)<2:return 0.
    ix=np.asarray(paths);comp=np.zeros(len(paths))
    for j,(pos,ev,_) in enumerate(ff):
        sims=np.array([iou(boxes[pos],b,True) for b in ev]);comp+=sims[ix[:,j]]/len(ff)
    return float(logsumexp(comp)-comp[int(lp.argmax())])

def audit_fit(z,initial,before,ex,row,arm):
    checks=check_state(z['initial'],initial);hist=z['path'];names=list(initial);assert sum(initial[n].numel() for n in names)==1792
    selected=min(range(len(hist)),key=lambda j:hist[j]['loss']);assert selected==z['selected_step'];checks+=check_state(z['state'],hist[selected]['state'])
    assert torch.equal(z['final'],hist[selected]['boxes']);setup=path_numpy(ex,before.numpy(),row['frame_ids']);ff,paths,lp,_=setup
    if arm=='map_contrastive':
        md=z['metadata'];assert md['map_index']==(int(lp.argmax()) if paths else None) and md['path_count']==len(paths)
        assert md['map_path']==(list(paths[int(lp.argmax())]) if paths else []) and md['no_competitor']==(len(paths)<2)
        assert np.allclose(md['probabilities'],np.exp(lp),rtol=0,atol=2e-6)
        if len(paths)<2:assert len(hist)==1 and z['gradient_calls']==0
    m=np.zeros(1792);v=np.zeros(1792)
    for j,h in enumerate(hist):
        actual=objective(h['boxes'].numpy(),ex,before.numpy(),row['frame_ids'],arm,setup);assert abs(actual-h['loss'])<8e-6,(arm,actual,h['loss']);checks+=1
        if 'update' not in h:continue
        u=h['update'];assert u['lr']==.03 and u['optimizer']=='adam';g=u['gradient'].numpy().astype(float)
        assert np.isfinite(g).all();m=.9*m+.1*g;v=.999*v+.001*g*g
        expected=-.03*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
        assert np.max(abs(expected-u['raw'].numpy()))<3e-6
        assert torch.equal(flat(hist[j+1]['state'],names)-flat(h['state'],names),u['raw']);checks+=1792*3
    return checks

def aggregate(rows,online=False):
    out={};fields=FIELDS if not online else ['v','t','s','vs_frozen_v','before_vs_frozen_v','vs_before_v','vs_episodic_v','gross_gain','gross_loss']
    tag='stream' if online else 'arm'
    for ds in DATASETS:
        out[ds]={}
        for sp in ['search','confirm']:
            out[ds][sp]={}
            for a in sorted({r[tag] for r in rows}):
                rr=[r for r in rows if r['dataset']==ds and r['split']==sp and r[tag]==a];groups={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean']}
                for order in ['order1','order2']:groups[order]=[r for r in rr if r['condition']!='clean' and r['order']==order]
                if online:
                    groups['expert_corrupt']=[r for r in rr if r['condition']!='clean' and r['expert']]
                    groups['nonexpert_corrupt']=[r for r in rr if r['condition']!='clean' and not r['expert']]
                for cond in sorted({r['condition'] for r in rr}):groups[cond]=[r for r in rr if r['condition']==cond]
                out[ds][sp][a]={}
                for name,q in groups.items():
                    z=stats(q,fields);ref='vs_frozen_v' if online else 'vs_before_v';z['tails']={f'harm_gt{n}pp':sum(r[ref]<-n/100 for r in q) for n in [5,20]};z['proxy_down_task_down']=sum(r.get('proxy_down_task_down',False) for r in q)
                    out[ds][sp][a][name]=z
    return out

def cpu_lock(stage):
    verify();name=BASE/f'{stage}_CPU_LOCK.json';pins={p:sha(ROOT/p) for p in ['scripts/score_decota_identity_commitment_v1.py','scripts/score_decota_actuation_scope_v1.py','scripts/score_decota_optimizer_posterior_v1.py','vg_tta/tastvg_oracle_event5_v1.py']}
    labels={str((c1.POOL/ds/f'GT_LABELS_{sp}.json').relative_to(ROOT)):sha(c1.POOL/ds/f'GT_LABELS_{sp}.json') for ds in DATASETS for sp in ['search','confirm']}
    write(name,dict(pins=pins,labels=labels,time=time.time()))
    return name

def score(stage):
    assert stage in ['matched','online'];verify();cpu=read(BASE/f'{stage}_CPU_LOCK.json')
    for f,h in {**cpu['pins'],**cpu['labels']}.items():assert sha(ROOT/f)==h
    bp=BASE/('MATCHED_GLOBAL_BARRIER.json' if stage=='matched' else 'ONLINE_GLOBAL_BARRIER.json');bar=read(bp);assert bar['status']=='sealed' and not bar['GT_read']
    for f,h in bar['files'].items():assert sha(BASE/f)==h
    write(BASE/f'{stage}_GT_EXPOSURE.json',dict(time=time.time(),barrier_sha256=sha(bp),GT_online=False))
    torch.set_num_threads(4);rows=[];diag=[];observation_rows=[];checks=0;tick=time.time();pub=PUB/stage;episodes={}
    cohorts=read(BASE/'COHORT.json')['cells'];online_lock=read(BASE/'ONLINE_LOCK.json') if stage=='online' else None
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');labels={sp:read(c1.POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        streams=['matched'] if stage=='matched' else online_lock['streams']
        for stream in streams:
            for split,sp in p['splits'].items():
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        prev=None;prevsha=None
                        for at,parent in enumerate(seq):
                            row=p['rows'][parent];f=BASE/stage/ds/split/cond/order/f'{at:05}.pt' if stage=='matched' else BASE/stage/ds/stream/split/cond/order/f'{at:05}.pt';x=checked(f)
                            assert x['parent']==parent and not x['GT_read'];ex=c1.checked(ROOT/x['evidence_path'])['expert'];assert sha(ROOT/x['evidence_path'])==x['evidence_sha256']
                            lab=labels[split][str(parent)];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span'];iv=x['native']['physical_interval']
                            dt=lambda boxes:DenseTube(boxes.numpy(),row,truth,span,clip=ds=='hc2');fr=dt(x['native']['boxes']).score(iv);before=dt(x['before']).score(iv)
                            if stage=='matched':
                                assert sha(ROOT/x['prestate'])==x['prestate_sha256'];old=r1.prior.checked(ROOT/x['prestate']);checks+=check_state(old['initial'],x['initial']);assert torch.equal(old['before'],x['before'])
                                a,b=references(x);fits={'top1':a,'marginal':b,'map_contrastive':x['fit']};top=dt(a['final']).score(iv)
                                observation_rows.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,**observation_quality(ex,x['before'],row,truth)))
                                for arm,z in fits.items():
                                    checks+=audit_fit(z,x['initial'],x['before'],ex,row,arm);metric=dt(z['final']).score(iv);off=official(z['final'].numpy(),row,truth,span,iv,ds);assert max(abs(off[k]-metric[k]) for k in off)<2e-12
                                    change=metric['v']-before['v'];rr=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=arm,**metric,
                                        before_v=before['v'],frozen_v=fr['v'],vs_frozen_v=metric['v']-fr['v'],vs_before_v=change,vs_top1_v=metric['v']-top['v'],gross_gain=max(change,0),gross_loss=max(-change,0),
                                        proxy_down_task_down=z['path'][-1]['loss']<z['path'][0]['loss'] and change<0,selected_step=z['selected_step'])
                                    rows.append(rr);diag.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,arm=arm,
                                        loss_path=[h['loss'] for h in z['path']],step_v=[dt(h['boxes']).score(iv)['v'] for h in z['path']],selected_step=z['selected_step'],gradient_calls=z['gradient_calls'],
                                        gradient_norms=[float(h['update']['gradient'].norm()) for h in z['path'] if 'update' in h],actual_step_norms=[float(h['update']['raw'].norm()) for h in z['path'] if 'update' in h],
                                        reused_control=arm!='map_contrastive'))
                            else:
                                assert x['query_reset'] and x['Adam_reset'] and x['previous_payload_sha256']==prevsha
                                init={n:torch.zeros_like(v) if n==QUERY else v for n,v in (prev if prev is not None else x['source_state']).items()};checks+=check_state(init,x['initial'])
                                expert=parent in online_lock['schedules'][ds][split][stream];assert expert==x['expert'] and torch.count_nonzero(x['initial'][QUERY])==0
                                if expert:
                                    checks+=audit_fit(x['fit'],x['initial'],x['before'],ex,row,online_lock['arm']);checks+=check_state(commit_state(x['initial'],x['fit']['state']),x['committed']);assert torch.equal(x['after'],x['fit']['final'])
                                else:checks+=check_state(x['initial'],x['committed']);assert torch.equal(x['after'],x['before']) and x['fit'] is None
                                if stream!='episodic':prev=x['committed'];prevsha=sha(f)
                                metric=dt(x['after']).score(iv);off=official(x['after'].numpy(),row,truth,span,iv,ds);assert max(abs(off[k]-metric[k]) for k in off)<2e-12
                                key=(ds,split,cond,order,at)
                                if stream=='episodic':episodes[key]=metric['v']
                                change=metric['v']-fr['v'];rows.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,stream=stream,expert=expert,**metric,
                                    frozen_v=fr['v'],before_v=before['v'],vs_frozen_v=change,before_vs_frozen_v=before['v']-fr['v'],vs_before_v=metric['v']-before['v'],vs_episodic_v=metric['v']-episodes[key],gross_gain=max(change,0),gross_loss=max(-change,0),
                                    payload_sha256=sha(f),prestate_sha256=state_hash(x['initial']),committed_state_sha256=state_hash(x['committed'])))
                                if 'reused_episodic' in x:assert sha(ROOT/x['reused_episodic'])==x['reuse_sha256']
                                diag.append(dict(dataset=ds,split=split,stream=stream,condition=cond,order=order,arrival=at,source_id=parent,expert=expert,
                                    fit_seconds=0. if 'reused_episodic' in x else x['fit_seconds'],actual_cached_backward_calls=x['actual_cached_backward_calls'],new_expert=0,new_backbone=0))
                        print('IDENTITY_ROOT_AUDIT',stage,ds,stream,split,cond,order,len(rows),checks,flush=True)
    expected=3456 if stage=='matched' else 13824;assert len(rows)==expected
    sums=aggregate(rows,stage=='online');write(pub/'ROWS.json',rows);write(pub/'DIAGNOSTICS.json',diag);write(pub/'SUMMARY.json',sums)
    write(pub/'ROOT_AUDIT.json',dict(status='pass',rows=len(rows),checks=checks,time=time.time(),seconds=time.time()-tick,independent_numpy_objective_and_Adam=True,
        sealed_before_GT=True,official_dense_verified=True,decoder_Jacobian_not_independently_replayed=True,all_negative_results_retained=True))
    if stage=='matched':
        write(pub/'OBSERVATION_QUALITY.json',observation_rows)
        gate={ds:sums[ds]['search']['map_contrastive']['corruption']['tails']['harm_gt20pp']==0 and sums[ds]['search']['map_contrastive']['corruption']['metrics']['vs_top1_v']['mean']>=0 and sums[ds]['search']['map_contrastive']['corruption']['metrics']['vs_top1_v']['ci95'][0]>=-.005 for ds in DATASETS}
        choice='map_contrastive' if all(gate.values()) else 'top1';selection=dict(arm=choice,search_gate=gate,uses_confirmation=False,time=time.time(),criterion_prelocked=True,method_promoted=False)
        write(BASE/'SEARCH_SELECTION.json',selection);write(pub/'DECISION.json',selection)
    else:
        schedules={}
        for ds in DATASETS:
            schedules[ds]={}
            for split in ['search','confirm']:
                schedules[ds][split]={}
                for rate in [25,50]:
                    ss=[f'seed{s}_budget{rate}' for s in range(5)];means=[sums[ds][split][s]['corruption']['metrics']['vs_frozen_v']['mean'] for s in ss]
                    rr=[r for r in rows if r['dataset']==ds and r['split']==split and r['stream'] in ss and r['condition']!='clean']
                    schedules[ds][split][str(rate)]=dict(schedule_means=means,mean=float(np.mean(means)),range=[min(means),max(means)],
                        source_bootstrap_conditional_on_five_schedules=stats(rr,['vs_frozen_v','before_vs_frozen_v','vs_before_v','vs_episodic_v']),
                        note='Average schedules within source before bootstrap; not 5x independent sources',source_rosters={s:online_lock['schedules'][ds][split][s] for s in ss})
        write(pub/'SCHEDULE_ROBUSTNESS.json',schedules);write(pub/'CONFIGURATION.json',online_lock)
    public_check(pub,stage)

def public_check(folder,stage):
    from scripts.decota_public_result_io_v1 import read as public_read
    folder=Path(folder);rows=public_read(folder/'ROWS.json');assert len(rows)==(3456 if stage=='matched' else 13824)
    assert aggregate(rows,stage=='online')==public_read(folder/'SUMMARY.json')
    for r in rows:
        assert 0<=r['v']<=1 and abs(r['vs_frozen_v']-(r['v']-r['frozen_v']))<1e-12
        assert abs(r['vs_before_v']-(r['v']-r['before_v']))<1e-12
    path=folder/'PUBLIC_AUDIT.json'
    if not path.exists():write(path,dict(status='pass',rows=len(rows),all_source_bootstrap_aggregates_recomputed=True,time=time.time()))
    return True

if __name__=='__main__':
    a=sys.argv[1];stage=sys.argv[2]
    if a=='lock':cpu_lock(stage)
    elif a=='score':score(stage)
    elif a=='public':public_check(sys.argv[3],stage)
