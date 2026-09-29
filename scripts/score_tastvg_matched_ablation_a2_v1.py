"""Post-barrier matched metrics, state/loss audit, all-order outcomes and harm tails."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_matched_ablation_a2_v1 import OUT,J01,S0,TEMP,NATIVE,verify
from scripts.score_tastvg_schedule_j01_v1 import independent_scores
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_tastvg_spatial_online_opd_s1_v1 import write_exposure
from methods.decota_final_simplified_v1.tensors import state_hash

def loss_dist(b,c,coeff):
    b=np.asarray(b,dtype=np.float64);c=np.asarray(c,dtype=np.float64)
    a,z=b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2
    e,f=c[...,:2]-c[...,2:]/2,c[...,:2]+c[...,2:]/2
    inter=np.maximum(np.minimum(z,f)-np.maximum(a,e),0).prod(-1)
    union=b[...,2:].prod(-1)+c[...,2:].prod(-1)-inter
    outer=(np.maximum(z,f)-np.minimum(a,e)).prod(-1)
    return coeff[0]*np.abs(b-c).sum(-1).mean(-1)+coeff[1]*(1-(inter/union-(outer-union)/outer)).mean(-1)

def describe(values):
    a=np.asarray(values,float);return dict(values=a.tolist(),mean=float(a.mean()),sample_std=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()),positive=int((a>0).sum()),negative=int((a<0).sum()),zero=int((a==0).sum()))

def aggregate(rows):
    result={};effects=[]
    for arm in sorted({r['arm'] for r in rows}):
        result[arm]={}
        for order in sorted({r['order'] for r in rows}):
            result[arm][order]={}
            for group in ['corruption','clean']+sorted({r['condition'] for r in rows if r['condition']!='clean'}):
                seq=[r for r in rows if r['arm']==arm and r['order']==order and (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
                result[arm][order][group]={}
                for subset in ['all','nonexpert']:
                    rr=[r for r in seq if subset=='all' or not r['expert_scheduled']];parents=sorted({r['parent'] for r in rr});keys=['s','t','v','delta_s','delta_t','delta_v','minus_final_s','minus_final_v']
                    src={parent:{key:float(np.mean([r[key] for r in rr if r['parent']==parent])) for key in keys} for parent in parents}
                    stats={key:float(np.mean([v[key] for v in src.values()])) for key in keys}
                    stats.update(sources=len(parents),cells=len(rr),source_harm_v=sum(v['delta_v']<-.05 for v in src.values()),source_harm_s=sum(v['delta_s']<-.05 for v in src.values()),cell_harm_v=sum(r['delta_v']<-.05 for r in rr),cell_harm_s=sum(r['delta_s']<-.05 for r in rr))
                    result[arm][order][group][subset]=stats
                    if group in ['clean','corruption']:
                        effects.extend(dict(arm=arm,order=order,group=group,subset=subset,parent=k,**v) for k,v in src.items())
    across={arm:{group:{sub:{key:describe([result[arm][o][group][sub][key] for o in result[arm]]) for key in result[arm][next(iter(result[arm]))][group][sub]} for sub in ['all','nonexpert']} for group in result[arm][next(iter(result[arm]))]} for arm in result}
    return result,across,effects

def run():
    import ijson
    torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==960
    for f,h in {**bar['files'],**bar['final_states']}.items():assert sha(OUT/f)==h
    keys={r['key'] for r in p['rows']};lp=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json'
    with lp.open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
    assert set(gt)==keys;write_exposure(OUT/'GT_EXPOSURE.json',dict(time=time.time(),queries=16,old_exposed_only=True,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),GT_container_sha256=sha(lp)))
    base=read(J01/'ROWS.json');ref={(r['order'],r['condition'],r['parent']):r for r in base};initial=load(NATIVE/'PARAMETER_SUPPORT.pt')['center']
    rows=[];diagnostics=[];count=coords=KL=PL=teacherchecks=0;maxparam=maxloss=maxdistance=0.;rewards_checked=0
    for arm in p['arms']:
      for order in p['orders']:
       for cond in p['conditions']:
        previous=initial
        for arrival,parent in enumerate(p['orders'][order]):
            r=next(v for v in p['rows'] if v['ordinal']==parent);name=f'{parent:03}.pt';x=load(OUT/'online'/arm/order/cond/name);z=ref[(order,cond,parent)]
            assert x['expert_scheduled']==(arrival in p['expert_indices']) and x['arrival']==arrival
            assert state_hash(x['pre_state'])==x['pre_state_sha256'] and state_hash(x['post_state'])==x['post_state_sha256'];assert all(torch.equal(v,previous[n]) for n,v in x['pre_state'].items());previous=x['post_state'];count+=1
            assert torch.equal(x['output_prediction']['boxes'],x['prediction']['boxes'])
            diag=dict(arm=arm,order=order,condition=cond,parent=parent,arrival=arrival,updated=x['updated'],step_norm=x['parameter_displacement'],inherited_norm=x['displacement_from_source'],pre_sha256=x['pre_state_sha256'],post_sha256=x['post_state_sha256'])
            if x['expert_scheduled']:
                assert x['expert_reads']==['temporal','spatial'];teacher=load(TEMP/'c2'/cond/name)
                for field in ['temporal','fast_control']:
                    d=x[field];scores=independent_scores(d['candidates'],teacher);np.testing.assert_allclose(scores,d['scores'],atol=1e-14,rtol=0);assert int(np.argmax(scores))==d['selected'];teacherchecks+=len(scores)
                e=load(S0/'expert'/cond/name)
                if arm!='direct_pl':
                    from vg_tta.box_stability_diagnostics_v1 import overlap
                    if e['valid'].any():
                        rewards=np.array([overlap(c['prediction']['boxes'].numpy()[e['valid']],e['boxes'][e['valid']]).mean() for c in x['candidates']]);np.testing.assert_allclose(rewards,x['rewards'],atol=1e-12,rtol=0);rewards_checked+=9
                    assert x['support_center_sha256']==state_hash(initial if arm=='off_policy' else x['pre_state'])
                if not e['valid'].any():assert not x['updated']
            else:assert x['expert_reads']==[] and 'candidates' not in x and 'update' not in x
            if x['updated']:
                u=x['update'];diag['loss_before']=u['loss_before'];diag['loss_after']=u['loss_after'];diag['gradient_norm']=u['global_gradient_norm']
                if arm=='direct_pl':
                    v=e['valid'];expected=float(loss_dist(x['prediction']['boxes'].numpy()[v],e['boxes'][v],[1,1]));PL+=1
                else:
                    re=np.array(x['rewards']);indices=np.argsort(-re,kind='stable');rank=np.empty(9);i=0
                    while i<9:
                        j=i+1
                        while j<9 and re[indices[i]]-re[indices[j]]<=1e-12:j+=1
                        rank[indices[i:j]]=(i+j-1)/2;i=j
                    perm=p['permutations'][order][str(parent)] if arm=='random_rank' else list(range(9));np.testing.assert_array_equal(rank,x['correct_ranks']);assigned=-re if arm=='raw_rkl' else rank[perm];np.testing.assert_array_equal(assigned,x['assigned_ranks']);assert perm==x['permutation']
                    qq=np.exp(-assigned);qq/=qq.sum();np.testing.assert_allclose(qq,u['q'],atol=1e-7,rtol=0)
                    d=loss_dist(x['prediction']['boxes'].numpy()[None],np.stack([c['prediction']['boxes'].numpy() for c in x['candidates']]),u['coefficients']);err=float(np.max(np.abs(d-np.array(u['distances']))));assert err<2e-6;maxdistance=max(maxdistance,err)
                    # Match the saved FP32 distance inputs; geometry audited independently above.
                    d=np.asarray(u['distances'],dtype=float);pp=np.exp(-d+d.min());pp/=pp.sum();np.testing.assert_allclose(pp,u['p'],atol=1e-7,rtol=0);expected=float(np.logaddexp(0,(d[:,None]-d[None,:])[assigned[:,None]<assigned[None,:]]).mean()) if arm=='pairwise_rank' and np.any(assigned[:,None]<assigned[None,:]) else (0. if arm=='pairwise_rank' else float(np.sum(pp*np.log(pp/qq))));KL+=int(arm=='raw_rkl')
                    diag.update(distances=d.tolist(),rewards=x['rewards'],ranks=assigned.tolist(),correct_ranks=rank.tolist(),permutation=perm,q=qq.tolist(),p=pp.tolist(),q_entropy=float(-(qq*np.log(qq)).sum()))
                err=abs(expected-u['loss_before']);assert err<2e-6,(arm,err);maxloss=max(maxloss,err)
                for n,b in x['pre_state'].items():
                    g=u['gradients'][n];expected=(b.double().numpy()-.005*g.double().numpy()).astype(np.float32);err=float(np.max(np.abs(expected-x['post_state'][n].numpy())));assert err<=1.5e-7;maxparam=max(maxparam,err);coords+=g.numel()
            else:assert all(torch.equal(v,x['pre_state'][n]) for n,v in x['post_state'].items())
            pred=x['output_prediction'];m,_=checked_score(pred['boxes'],gt[r['key']],r['frame_ids'],pred['indices'])
            rec=dict(arm=arm,order=order,condition=cond,parent=parent,arrival=arrival,expert_scheduled=x['expert_scheduled'])
            for code,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:rec.update({code:m[key],'delta_'+code:m[key]-z['frozen_'+code],'minus_final_'+code:m[key]-z['final_'+code]})
            if not x['expert_scheduled']:
                for code in ['s','t','v']:assert z['fast_'+code]==z['frozen_'+code]
            rows.append(rec);diagnostics.append(diag)
        assert all(torch.equal(v,previous[n]) for n,v in load(OUT/'final_states'/arm/order/f'{cond}.pt').items())
    for x in base:
        for arm,prefix in [('Frozen','frozen'),('Fast-only','fast'),('Slow-only','slow'),('Final','final')]:
            rec={k:x[k] for k in ['order','condition','parent','arrival','expert_scheduled']};rec['arm']=arm
            for code in ['s','t','v']:rec.update({code:x[prefix+'_'+code],'delta_'+code:x[prefix+'_'+code]-x['frozen_'+code],'minus_final_'+code:x[prefix+'_'+code]-x['final_'+code]})
            rows.append(rec)
    summary,across,effects=aggregate(rows)
    for name,value in [('ROWS.json',rows),('DIAGNOSTICS.json',diagnostics),('SUMMARY.json',summary),('ACROSS_ORDERS.json',across),('SOURCE_EFFECTS.json',effects)]:write(OUT/name,value)
    write(OUT/'AUDIT.json',dict(status='pass',new_cells=count,reused_cells=len(base),state_resets=60,state_links=count,SGD_coordinates=coords,KL_updates=KL,pairwise_updates=sum(d["updated"] and d["arm"]=="pairwise_rank" for d in diagnostics),PL_updates=PL,reward_values=rewards_checked,independent_temporal_scores=teacherchecks,dual_metric_calls=count,maximum_parameter_error=maxparam,maximum_loss_error=maxloss,maximum_distance_error=maxdistance,all_predictions_sealed_before_GT=True,frozen_method_unchanged=True))
    for arm in across:print(arm,{sub:{k:across[arm]['corruption'][sub][k] for k in ['delta_s','delta_v','minus_final_v','source_harm_v','cell_harm_v']} for sub in ['all','nonexpert']},flush=True)

if __name__=='__main__':run()
