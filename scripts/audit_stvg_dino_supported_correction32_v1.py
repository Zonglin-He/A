"""Independent conditional-assignment, source-choice, metric and factorial audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_supported_correction32_v1 import D,P,OLD,OLDP,CRITIC,BASES,GRID,verify


def chart(b):
    b=np.asarray(b,float);return np.concatenate((b[:,:2],np.log(b[:,2:])),axis=1)


def close(a,b,tol=2e-10):
    if a is None or b is None:assert a is b;return
    assert np.max(abs(np.asarray(a)-np.asarray(b)),initial=0)<tol,(a,b)


def energy_minimum(pools,scores,scale,cutoff,beta):
    # Minimize conditional LS + explicit candidate costs + explicit outlier
    # constants. Does not call the estimator's truncated proposal-rescoring path.
    penalty=[]
    for ss in scores:
        s=np.maximum(ss,1e-12);penalty.append(beta*cutoff*np.log(s.max()/s))
    minimum=np.inf;count=0
    for assign in itertools.product(*[range(-1,len(v))for v in pools]):
        active=np.array([v[k]for v,k in zip(pools,assign)if k>=0]).reshape(-1,4)
        delta=active.mean(0)if len(active)else np.zeros(4)
        cost=float(np.square((active-delta)/scale).sum()+(len(pools)-len(active))*cutoff)
        cost+=float(sum(c[k]for c,k in zip(penalty,assign)if k>=0))
        minimum=min(minimum,cost);count+=1
    return minimum,count


def decoded(b,delta):
    if np.array_equal(delta,np.zeros(4)):return np.asarray(b,float).copy()
    z=chart(b)+delta;return np.column_stack((z[:,:2],np.exp(z[:,2:])))


def iou(a,b):
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
    return float(inter/union)if union>0 else 0.


def unobserved_metric(b,ids,planned,inp,truth,ds):
    scale=np.array([inp['width'],inp['height']]*2);xy=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*scale
    if ds=='hc2':xy=np.maximum(xy,0)
    vv=[iou(xy[i],np.asarray(truth[int(fid)],float))for i,fid in enumerate(ids)if int(fid)in truth and i not in planned]
    return float(np.mean(vv))if vv else None


def main():
    import torch
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube
    from scripts.score_stvg_opd_p1_v1 import truths
    torch.set_num_threads(4);p=verify(inputs=True);bc=read(D/'BETA_CALIBRATION.json');cal=read(OLD/'SOURCE_CALIBRATION.json')
    assert sha(OLD/'SOURCE_CALIBRATION.json')==bc['old_scale_cutoff_sha256'];assert bc['one_beta_both_domains']and not bc['target_GT_used']
    source_rows=[];solverchecks=0;assignments=0;sourcechecks=0
    for r in p['source']:
        if r['split']!='source_validation':continue
        out=load(D/'source_validation_predictions'/(r['alias']+'.pt'));orig=load(CRITIC/'source'/(r['alias']+'.pt'))
        y=load(ROOT/'artifacts/stvg_joint_posterior_v1/source_supervision'/(r['alias']+'.pt'));b=out['Base'];pp=[];scores=[]
        for anchor in orig['expert']['anchors']['single4']:
            probe=orig['expert']['observations'][('original',anchor['position'])]['probe']
            pp.append(chart(probe['boxes'])-chart(b)[anchor['position']]);scores.append(np.asarray(probe['target_scores'],float))
        setting=cal['source_domains'][r['dataset']];u0=unobserved_metric(b,out['frame_ids'],out['planned_positions'],out['input'],y['dense_source_GT'],r['dataset'])
        row=dict(frame_ids=out['frame_ids'],input=out['input']);v0=DenseTube(b,row,y['dense_source_GT'],y['span'],clip=r['dataset']=='hc2').score(out['Native_interval'])['v'];vals={}
        for beta in GRID:
            est=out['outputs'][str(beta)]['estimate'];bb=out['outputs'][str(beta)]['boxes'];opt,n=energy_minimum(pp,scores,np.asarray(setting['scale']),setting['truncation'],beta)
            close(opt,est['objective'],1e-7);solverchecks+=1;assignments+=n;close(decoded(b,est['delta']),bb,1e-13)
            uu=unobserved_metric(bb,out['frame_ids'],out['planned_positions'],out['input'],y['dense_source_GT'],r['dataset'])
            vv=DenseTube(bb,row,y['dense_source_GT'],y['span'],clip=r['dataset']=='hc2').score(out['Native_interval'])['v']
            vals[str(beta)]=dict(unobserved_gain_pp=100*(uu-u0),full_v_gain_pp=100*(vv-v0))
            saved=next(s for s in bc['source_rows']if s['alias']==r['alias'])['values'][str(beta)]
            for key,value in vals[str(beta)].items():close(value,saved[key]);sourcechecks+=1
        source_rows.append(dict(dataset=r['dataset'],values=vals))
    pooled={str(beta):{ep:float(np.mean([np.mean([r['values'][str(beta)][ep]for r in source_rows if r['dataset']==ds])for ds in ['hc2','vidstg']]))for ep in ['unobserved_gain_pp','full_v_gain_pp']}for beta in GRID}
    for beta in GRID:
        for ep in pooled[str(beta)]:close(pooled[str(beta)][ep],bc['pooled'][str(beta)][ep]);sourcechecks+=1
    selected=max(GRID[1:],key=lambda b:(pooled[str(b)]['unobserved_gain_pp'],pooled[str(b)]['full_v_gain_pp'],-b));assert selected==bc['selected_beta'];sourcechecks+=1
    bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_E_and_diagnostics_before_target_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    old=read(OLDP/'RESULTS.json');assert sha(OLDP/'RESULTS.json')==read(P/'RESULTS.json')['old_results_sha256']
    prior={r['alias']:r for r in old['rows']};result=read(P/'RESULTS.json');rr={r['alias']:r for r in result['rows']}
    target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in target}
    outputs=0;metrics=0;attrs=0;oldchecks=0;aggregates=0;maxerr=0.
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));oldz=load(OLD/'predictions'/(r['alias']+'.pt'));s=rr[r['alias']]
        setting=cal['source_domains'][cal['target_to_source'][r['dataset']]];scale=np.array(setting['scale']);cutoff=setting['truncation']
        row=plans[r['dataset']][r['query_ordinal']];truth,span=target[r['dataset']][0][r['query_ordinal']],target[r['dataset']][1][r['query_ordinal']]
        assert len(z['frames'])==len(oldz['frames'])
        for f,g in zip(z['frames'],oldz['frames']):
            assert np.array_equal(f['boxes'],g['boxes'])and np.array_equal(f['scores'],g['scores'])and f['position']==g['position'];oldchecks+=1
        for base in BASES:
            data=z['bases'][base];b=oldz['bases'][base]['Base'];pp=[chart(f['boxes'])-chart(b)[f['position']]for f in z['frames']];scores=[np.array(f['scores'])for f in z['frames']]
            est=data['estimate'];opt,n=energy_minimum(pp,scores,scale,cutoff,selected);close(opt,est['objective'],1e-7);solverchecks+=1;assignments+=n
            picked=[v[k]for v,k in zip(pp,est['selected'])if k>=0];delta=np.mean(picked,axis=0)if picked else np.zeros(4)
            close(delta,est['delta']);bb=decoded(b,delta);maxerr=max(maxerr,float(np.max(abs(bb-data['E']),initial=0)));close(bb,data['E'],1e-13);outputs+=1
            for candidate in range(2):
                for mask in range(2):
                    choice=[data['old_latent'],est['latent_best']][candidate];accept=np.array([data['old_selected'],est['selected']][mask])>=0
                    use=[v[k]for v,k,yes in zip(pp,choice,accept)if yes];d=np.mean(use,axis=0)if use else np.zeros(4)
                    close(decoded(b,d),data['candidate_acceptance_diagnostics'][str(candidate*2+mask)],1e-13);outputs+=1
            for arm in ['Base','A_mean','B_robust_top1','C_joint','D_C_center']:
                assert s['values'][base][arm]==prior[r['alias']]['values'][base][arm];oldchecks+=1
            for label,boxes in {**data['candidate_acceptance_diagnostics'],'E':data['E']}.items():
                expected=s['diagnostic_factorial_values'][base][label];tube=DenseTube(boxes,row,truth,span,clip=r['dataset']=='hc2');met=tube.score(z['interval'])
                for key in ['v','s','t']:close(met[key],expected['metrics'][key]);metrics+=1
                u=unobserved_metric(boxes,z['frame_ids'],z['planned_positions'],row['input'],truth,r['dataset']);close(u,expected['supports']['unobserved']['s']);metrics+=1
            for ep in ['v','unobserved']:
                four=[s['diagnostic_factorial_values'][base][str(i)]['metrics']['v']if ep=='v'else s['diagnostic_factorial_values'][base][str(i)]['supports']['unobserved']['s']for i in range(4)]
                factor=s['attribution'][base][ep];close(factor['candidate_choices'],50*((four[2]-four[0])+(four[3]-four[1])))
                close(factor['accepted_frame_mask'],50*((four[1]-four[0])+(four[3]-four[2])))
                close(factor['candidate_choices']+factor['accepted_frame_mask'],100*(four[3]-four[0]));attrs+=3
    for ds,gg in result['groups'].items():
        rows=[r for r in result['rows']if r['dataset']==ds]
        for base,g in gg.items():
            for control,endpoints in g['E_minus'].items():
                for ep,pair in endpoints.items():
                    vals=[]
                    for r in rows:
                        e=r['values'][base]['E_DINO_support'];a=r['values'][base][control]
                        v=e['metrics'][ep]if ep in ['v','s']else e['supports'][ep]['s'];w=a['metrics'][ep]if ep in ['v','s']else a['supports'][ep]['s']
                        if v is not None and w is not None:vals.append(100*(v-w))
                    v=np.array(vals);close(v.mean(),pair['mean']);close(v,pair['per_parent_pp'])
                    idx=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));close(np.quantile(v[idx].mean(1),[.025,.975]),pair['CI95']);aggregates+=3
            for ep,axes in g['E_minus_C_factorial_attribution'].items():
                for axis,pair in axes.items():close(np.mean([r['attribution'][base][ep][axis]for r in rows]),pair['mean']);aggregates+=1
    verify(inputs=True);assert not torch.cuda.is_initialized()
    receipt=dict(status='pass',count=32,independent_conditional_assignment_games=solverchecks,assignments_enumerated=assignments,
        source_choice_and_source_metric_checks=sourcechecks,source_beta_shared_and_exact=True,
        output_reconstruction_checks=outputs,maximum_output_error=maxerr,independent_metric_checks=metrics,
        factorial_attribution_checks=attrs,old_inputs_and_ABCD_metric_exact_checks=oldchecks,
        aggregate_and_CI_checks=aggregates,all_prediction_code_and_registry_pins_unchanged=True,
        no_new_model_expert_training_or_Student_updates=True,independent_math_not_new_human_identity_labels=True)
    put(P/'ROOT_INDEPENDENT_AUDIT.json',receipt);put(D/'ROOT_COMPLETION.json',receipt);print(receipt,flush=True)


if __name__=='__main__':main()
