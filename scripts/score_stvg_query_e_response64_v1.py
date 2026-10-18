"""Post-seal CPU evaluation of local interventions and complete control paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,collections,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_query_e_response64_v1 import D,P,OLD,verify
from scripts.analyze_stvg_query_e_path64_v1 import overlap,mean,cosine


def paired(x):
    a=np.array([v for v in x if v is not None],float)
    if not len(a):return dict(n=0,mean_pp=None,CI95_pp=None)
    rng=np.random.default_rng(20261018);b=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(n=len(a),mean_pp=float(a.mean()),CI95_pp=np.quantile(b,[.025,.975]).tolist(),positive=int((a>1e-8).sum()),negative=int((a< -1e-8).sum()),neutral=int((abs(a)<=1e-8).sum()),severe_below_minus5=int((a< -5).sum()))


def main():
    lock=read(D/'ANALYSIS_IMPLEMENTATION_LOCK.json')
    for f,h in lock['pins'].items():assert sha(ROOT/f)==h,f
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and not bar['GT_read']
    start=time.perf_counter();targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    oldres=read(ROOT/'results/stvg_error_state/2026-10-18/query_e64/RESULTS.json');oldby={r['alias']:r for r in oldres['rows']}
    rows=[];counts=collections.Counter()
    for rec in p['records']:
        ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q]
        z=load(D/'predictions'/(rec['alias']+'.pt'));old=load(OLD/'predictions'/(rec['alias']+'.pt'))
        ids=np.array(z['frame_ids']);mask=np.array([int(v)in gt for v in ids])&~np.isin(np.arange(len(ids)),z['planned_positions'])
        teacher=old['E_training_teacher_float32'].double().numpy();target=np.array([gt[int(t)]for t in ids[mask]])
        def metrics(box):
            b=box.double().numpy()if torch.is_tensor(box)else np.asarray(box)
            met=official(b,row,gt,span,z['interval'],ds);ref=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval'])
            assert all(abs(met[k]-ref[k])<2e-10 for k in ['v','t','s'])
            xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            counts['metric_states']+=1
            return dict(**met,unobserved_actual_s=mean(box_iou(xy[mask],target))if mask.any()else None,E_IoU_all=float(overlap(b,teacher).mean()))
        local=[]
        for loc in z['local']:
            b=metrics(loc['before']['boxes']);variants={}
            for name,v in loc['variants'].items():
                mv=metrics(v['boxes']);assert mv['t']==b['t'];counts['local_time_invariants']+=1
                variants[name]=dict(values=mv,surrogate=v['loss'],surrogate_change=None if v['loss']is None else v['loss']-loc['before']['loss'],
                    QR_step_norm=float(v['delta'].norm()),output_L2=float((v['boxes']-loc['before']['boxes']).norm()),
                    gradient_dot_step=v['gradient_dot_step'],native_gradient_dot_step=v['same_native_gradient_dot_step'],chain_gradient_dot_step=v['full_chain_gradient_dot_step'])
            local.append(dict(round=loc['round'],arm=loc['arm'],primary=loc['primary'],updated=loc['updated'],before=b,before_surrogate=loc['before']['loss'],variants=variants,
                FD=loc['FD'],gradient_cosine=cosine(loc['gradients'].get(False,torch.zeros(256)),loc['gradients'].get(True,torch.zeros(256))),compute=loc['compute']))
        fits={};values=oldby[rec['alias']]['values'].copy()
        for arm,f in z['fits'].items():
            path=[]
            for k,st in enumerate(f['path']):
                m=metrics(st['boxes']);assert m['t']==values['Source_MBR']['t'];counts['control_time_invariants']+=1
                path.append(dict(step=k,**m,QR_norm=float(st['state']['spatial.query_residual'].norm()),output_L2_from_Source=float((st['boxes']-old['fits']['Dense_OPD']['before']).norm())))
            rounds=[]
            for k,rr in enumerate(f['rounds']):
                trials=rr['trials'];rounds.append(dict(round=k,updated=rr['updated'],no_expert_support=rr.get('no_expert_support',False),
                    informative=rr.get('informative',False),accepted_scale=rr.get('accepted_scale'),loss_before=rr.get('loss_before'),trials=trials,
                    QR_step_norm=float((f['path'][k+1]['state']['spatial.query_residual']-f['path'][k]['state']['spatial.query_residual']).norm()),
                    output_L2=float((f['path'][k+1]['boxes']-f['path'][k]['boxes']).norm()),rejected_state_restored=rr.get('rejected_state_restored',False)))
            values[arm]={k:path[-1][k]for k in ['v','t','s','unobserved_actual_s']}
            fits[arm]=dict(states=path,rounds=rounds,accepted=sum(r['updated']for r in rounds),
                restored=sum(r['rejected_state_restored']for r in rounds),flat_rewards=sum(not r['informative']and not r['no_expert_support']for r in rounds),
                scales=dict(collections.Counter(str(r['accepted_scale'])for r in rounds if r['updated'])),
                final_output_L2=path[-1]['output_L2_from_Source'],final_QR_norm=path[-1]['QR_norm'],compute=f['compute'])
        rows.append(dict(alias=rec['alias'],dataset=ds,active=z['active'],values=values,local=local,fits=fits,compute=z['compute']))
    summary={};effects={}
    comparisons=[(a,b)for a,b in [('Dense_control','Dense_OPD'),('Dense_control','Source_MBR'),('Dense_control','E_MBR'),('Dense_control','Dense_Direct'),('Dense_control','Fixed_rollout'),('Sparse_control','Sparse_OPD'),('Sparse_control','Source_MBR')]]
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];summary[ds]={}
        summary[ds]['means']={a:{ep:mean([r['values'][a][ep]for r in rr])for ep in ['v','t','s','unobserved_actual_s']}for a in rr[0]['values']}
        summary[ds]['paired']={a+'-'+b:{ep:paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])for ep in ['v','unobserved_actual_s']}for a,b in comparisons}
        summary[ds]['control']={a:dict(accepted=sum(r['fits'][a]['accepted']for r in rr),restored=sum(r['fits'][a]['restored']for r in rr),
            flat_rewards=sum(r['fits'][a]['flat_rewards']for r in rr),scales=dict(sum((collections.Counter(r['fits'][a]['scales'])for r in rr),collections.Counter())),
            mean_final_output_L2=mean([r['fits'][a]['final_output_L2']for r in rr]),mean_final_QR_norm=mean([r['fits'][a]['final_QR_norm']for r in rr]))for a in ['Sparse_control','Dense_control']}
        # Each query contributes one mean over its three prespecified states.
        effects[ds]={}
        for factor in ['half_minus_full','current_minus_history','chain_minus_detach']:
            queryeffects=[]
            for r in rr:
                vals=[]
                for loc in [x for x in r['local']if x['primary']]:
                    for chain,hist,amp in itertools.product(['detach','chain'],['history','current'],['full','half']):
                        if factor=='half_minus_full'and amp!='full':continue
                        if factor=='current_minus_history'and hist!='history':continue
                        if factor=='chain_minus_detach'and chain!='detach':continue
                        n0=f'{chain}_{hist}_{amp}';n1=f'{chain}_{hist}_half'if factor=='half_minus_full'else f'{chain}_current_{amp}'if factor=='current_minus_history'else f'chain_{hist}_{amp}'
                        a,b=loc['variants'][n1],loc['variants'][n0]
                        vals.append(dict(v=100*(a['values']['v']-b['values']['v']),E=100*(a['values']['E_IoU_all']-b['values']['E_IoU_all']),
                            loss=None if a['surrogate']is None else a['surrogate']-b['surrogate']))
                queryeffects.append({k:mean([v[k]for v in vals])for k in ['v','E','loss']})
            effects[ds][factor]={k:paired([v[k]for v in queryeffects])for k in ['v','E','loss']}
            loss_summary=effects[ds][factor]['loss']
            loss_summary['mean']=loss_summary.pop('mean_pp');loss_summary['CI95']=loss_summary.pop('CI95_pp')
            loss_summary.pop('severe_below_minus5');loss_summary['units']='original surrogate units, not percentage points'
        fds=[fd for r in rr for l in r['local']if l['primary']for fd in l['FD']]
        effects[ds]['FD']=dict(count=len(fds),native_abs_error_median=mean([]),
            rows=[dict(alias=r['alias'],round=l['round'],**fd)for r in rr for l in r['local']if l['primary']for fd in l['FD']])
        if fds:
            effects[ds]['FD'].update(native_abs_error_median=float(np.median([abs(x['finite_difference']-x['native_autograd'])for x in fds])),
                chain_abs_error_median=float(np.median([abs(x['finite_difference']-x['chain_autograd'])for x in fds])),
                native_wrong_sign=sum(x['finite_difference']*x['native_autograd']<0 for x in fds),chain_wrong_sign=sum(x['finite_difference']*x['chain_autograd']<0 for x in fds))
    result=dict(status='complete',count=64,summary=summary,local_factor_effects=effects,rows=rows,counts=dict(counts),
        GT_only_after_global_seal=True,no_GT_selected_steps=True,local_not_method_performance=True,independent_confirmation=False)
    put(P/'RESULTS.json',result);put(D/'CPU_COMPLETION.json',dict(status='complete',count=64,CPU_seconds=time.perf_counter()-start,result_sha256=sha(P/'RESULTS.json')))
    print({ds:summary[ds]['means']for ds in summary})


if __name__=='__main__':main()
