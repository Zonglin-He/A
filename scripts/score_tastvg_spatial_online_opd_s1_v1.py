"""Post-seal prequential task scores and independent persistent SGD reconstruction."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_spatial_online_opd_s1_v1 import OUT,S0,NATIVE,CRITIC,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats
from methods.decota_final_simplified_v1.tensors import state_hash

def write_exposure(path,value):
    if path.exists():
        prior=read(path);value['time']=prior['time'];assert value==prior
    else:write(path,value)

def macro(rows,key):
    by=collections.defaultdict(list)
    for r in rows:
        if r[key] is not None:by[r['parent']].append(r[key])
    return stats([np.mean(by[p]) for p in sorted(by)])

def run():
    import ijson
    torch.set_num_threads(4);lock=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==96
    for f,h in {**bar['files'],**bar['final_states']}.items():assert sha(OUT/f)==h
    keys={r['key'] for r in lock['rows']};labelpath=ROOT/'artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json'
    with labelpath.open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
    assert set(gt)==keys
    write_exposure(OUT/'GT_EXPOSURE.json',dict(old_exposed_keys_only=True,retained_queries=16,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),container_sha256=sha(labelpath),time=time.time()))
    baseline={(r['parent'],r['condition']):r for r in read(NATIVE/'ROWS.json')};critic={(r['parent'],r['condition']):r for r in read(CRITIC/'REWARDS.json')};initial=load(NATIVE/'PARAMETER_SUPPORT.pt')['center'];rows=[];calls=0;updatechecks=0;maxerror=0.;gradient_params=0;maxklerror=0.
    for cond in lock['conditions']:
        previous=initial
        for arrival,r in enumerate(lock['rows']):
            x=load(OUT/'online'/cond/f"{r['ordinal']:03}.pt");assert state_hash(x['pre_state'])==x['pre_state_sha256'] and state_hash(x['post_state'])==x['post_state_sha256'];assert all(torch.equal(v,previous[n]) for n,v in x['pre_state'].items());assert x['expert_scheduled']==(arrival in lock['expert_indices'])
            if x['updated']:
                u=x['update'];d=u['distances'].double().numpy();re=np.asarray(x['rewards'],float);pp=np.exp(-d+d.min());pp/=pp.sum();qq=np.exp(re-re.max());qq/=qq.sum();kl=float(np.sum(pp*(np.log(pp)-np.log(qq))));klerror=abs(kl-u['loss_before']);assert klerror<1e-6;maxklerror=max(maxklerror,klerror)
                np.testing.assert_allclose(pp,u['p'].numpy(),atol=1e-7,rtol=0);np.testing.assert_allclose(qq,u['q'].numpy(),atol=1e-7,rtol=0)
                for n,b in x['pre_state'].items():
                    expected=(b.double().numpy()-lock['lr']*u['gradients'][n].double().numpy()).astype(np.float32);err=float(np.max(abs(expected-x['post_state'][n].numpy())));assert err<=1.5e-7;maxerror=max(maxerror,err);gradient_params+=b.numel()
                assert u['candidate_targets_detached'] and u['reward_detached'];updatechecks+=1
            else:assert all(torch.equal(v,x['pre_state'][n]) for n,v in x['post_state'].items())
            previous=x['post_state'];label=gt[r['key']];metrics=[]
            for pred in [x['prediction'],x['post_prediction']]:
                m,_=checked_score(pred['boxes'],label,r['frame_ids'],pred['indices']);metrics.append(m);calls+=1
            b=baseline[r['ordinal'],cond];selected=critic[r['ordinal'],cond]['selected'] if x['expert_scheduled'] else 0
            bm=b['candidate_metrics'][selected];native=b['candidate_metrics'][0];record=dict(parent=r['ordinal'],condition=cond,arrival=arrival,expert_scheduled=x['expert_scheduled'],updated=x['updated'],state_displacement=x['displacement_from_source'],step_displacement=x['parameter_displacement'],pre_state_sha256=x['pre_state_sha256'],post_state_sha256=x['post_state_sha256'],changed_boxes=not torch.equal(x['prediction']['boxes'],load(S0/'capture'/cond/f"{r['ordinal']:03}.pt")['prediction']['boxes']),update_diagnostics={k:v for k,v in x.get('update',{}).items() if k not in ['gradients','p','q','distances']})
            if x['updated']:record['update_diagnostics'].update(p=x['update']['p'].tolist(),q=x['update']['q'].tolist(),distances=x['update']['distances'].tolist(),reward=x['rewards'])
            for short,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:
                vals=dict(frozen=native[key],budgeted=bm[key],online=metrics[0][key],post=metrics[1][key])
                for arm,val in vals.items():record[f'{arm}_{short}']=val
                for name,a,z in [('online_minus_frozen','online','frozen'),('online_minus_budgeted','online','budgeted'),('post_minus_pre','post','online')]:record[f'{name}_{short}']=vals[a]-vals[z] if vals[a] is not None and vals[z] is not None else None
            rows.append(record)
        assert all(torch.equal(v,previous[n]) for n,v in load(OUT/'final_states'/f'{cond}.pt').items())
    summary={};keys=[k for k in rows[0] if k.endswith(('_s','_t','_v'))]
    for group in ['corruption']+lock['conditions']:
        rr=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)];summary[group]={}
        for subset in ['all','nonexpert','expert']:
            seq=[r for r in rr if subset=='all' or r['expert_scheduled']==(subset=='expert')]
            summary[group][subset]=dict(cells=len(seq),sources=len({r['parent'] for r in seq}),metrics={k:macro(seq,k) for k in keys},changed_box_cells=sum(r['changed_boxes'] for r in seq))
    write(OUT/'ROWS.json',rows);write(OUT/'SUMMARY.json',summary);write(OUT/'AUDIT.json',dict(status='pass',state_chain_cells=96,stream_resets=6,updates=updatechecks,independent_parameter_updates=gradient_params,maximum_SGD_rounding_error=maxerror,maximum_KL_reconstruction_error=maxklerror,KL_absolute_tolerance=1e-6,dual_metric_calls=calls,KL_reconstructions=updatechecks,full_learned_state_reinsertions=len(read(OUT/'REINSERTION_AUDIT.json')['cells']),GT_post_seal=True))
    print({g:{sub:{k:summary[g][sub]['metrics'][k] for k in ['online_minus_frozen_s','online_minus_frozen_v']} for sub in ['nonexpert','all']} for g in ['corruption','clean']})

if __name__=='__main__':run()
