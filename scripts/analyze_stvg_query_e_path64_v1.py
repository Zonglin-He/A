"""CPU-only retrospective diagnosis of immutable, already sealed QR256 paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,json,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load
OLD=ROOT/'artifacts/stvg_query_e_opd64_v1'
D=ROOT/'artifacts/stvg_query_e_path64_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/query_e_path64'
ARMS=['Sparse_OPD','Dense_OPD','Fixed_rollout','Dense_Direct']
CASES=['H008','H128','V038','H119','V016'];EPS=1e-10


def mean(x):
    x=[float(v)for v in x if v is not None];return float(np.mean(x))if x else None


def overlap(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    a=np.concatenate((x[...,:2]-x[...,2:]/2,x[...,:2]+x[...,2:]/2),-1);b=np.concatenate((y[...,:2]-y[...,2:]/2,y[...,:2]+y[...,2:]/2),-1)
    it=np.maximum(0,np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2])).prod(-1)
    un=np.maximum(0,a[...,2:]-a[...,:2]).prod(-1)+np.maximum(0,b[...,2:]-b[...,:2]).prod(-1)-it
    return np.divide(it,un,out=np.zeros_like(it),where=un>0)


def direct_loss(x,y):
    a=np.c_[x[:,:2]-x[:,2:]/2,x[:,:2]+x[:,2:]/2];b=np.c_[y[:,:2]-y[:,2:]/2,y[:,:2]+y[:,2:]/2]
    inter=np.maximum(0,np.minimum(a[:,2:],b[:,2:])-np.maximum(a[:,:2],b[:,:2])).prod(1)
    union=np.maximum(0,a[:,2:]-a[:,:2]).prod(1)+np.maximum(0,b[:,2:]-b[:,:2]).prod(1)-inter;union=np.maximum(union,1e-7)
    en=np.maximum(np.maximum(a[:,2:],b[:,2:])-np.minimum(a[:,:2],b[:,:2]),0).prod(1);en=np.maximum(en,1e-7)
    return float((5*abs(x-y).sum(1)+2*(1-inter/union+(en-union)/en)).mean())


def sign(v):return 'up'if v>EPS else'down'if v< -EPS else'flat'


def cosine(a,b):
    a=np.ravel(a);b=np.ravel(b);norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm)if norm>0 else None


def surrogate(mu,rr,sigma):
    old=rr['mean'].double().numpy();samples=rr['samples'].double().numpy();weights=rr['weights'].double().numpy()
    kl=((mu-old)**2).sum(-1)/(2*sigma**2)
    logp=-((samples-mu[:,None])**2).sum(-1)/(2*sigma**2)-2*np.log(2*np.pi*sigma**2)
    return float((kl-((weights-1/32)*logp).sum(-1)).mean())


def prepare():
    assert not(D/'PLAN_LOCK.json').exists();old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==64 and bar['all_outputs_before_GT']and read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
    inputs={v['path']:v['sha256']for v in bar['files']}
    inputs['results/stvg_error_state/2026-10-18/query_e64/RESULTS.json']=read(OLD/'CPU_COMPLETION.json')['result_sha256']
    own=['scripts/analyze_stvg_query_e_path64_v1.py','protocols/stvg_query_e_path64_v1.md']
    pins={**read(OLD/'RUNTIME_LOCK.json')['pins'],**inputs,**{f:sha(ROOT/f)for f in own}}
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    put(D/'PLAN_LOCK.json',dict(status='locked_before_path_GT_readback',count=64,records=old['records'],arms=ARMS,fixed_cases=CASES,
        previous_commit='288f2876732b714c3c9d6d1f1312df1102e84da9',original_selected_steps_unchanged=True,
        new_models=0,new_fits=0,new_DINO_calls=0,sign_epsilon=EPS,CPU_budget_seconds=3600,deadline_epoch=time.time()+7200))
    put(D/'RUNTIME_LOCK.json',dict(pins=pins,registries=read(OLD/'RUNTIME_LOCK.json')['registries'],plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'STATUS.json',dict(status='prepared',count=64,GT_for_new_paths_read=False))


def verify():
    from main_method.runtime import PinVerifier,resource_guard
    p=read(D/'PLAN_LOCK.json');lock=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==lock['plan_sha256'];assert time.time()<p['deadline_epoch']
    PinVerifier({**lock['pins'],**lock['registries']}).verify(full=True);resource_guard();return p


def classify(f):
    a,b=f['states'][0],f['states'][-1];e=b['E_IoU_all']-a['E_IoU_all'];v=100*(b['v']-a['v']);ev=100*(b['E_IoU_all']-a['E_IoU_all'])
    tr=[v for v in f['transitions']if v['updated']]
    return dict(E_sign=sign(e),v_sign=sign(v),E_change_pp=ev,v_change_pp=v,
        unobserved_change_pp=None if a['unobserved_actual_s']is None else 100*(b['unobserved_actual_s']-a['unobserved_actual_s']),
        actual_updates=len(tr),E_decreasing_updates=sum(v['E_all_change_pp']< -1e-8 for v in tr),
        E_increase_GT_decrease_updates=sum(v['E_all_change_pp']>1e-8 and v['v_change_pp']< -1e-8 for v in tr),
        same_rollout_surrogate_rise_updates=sum(v.get('surrogate_rise_above_tolerance',False)for v in tr),
        output_descent_cos_negative_updates=sum(v.get('negative_mean_gradient_actual_mu_cosine')is not None and v['negative_mean_gradient_actual_mu_cosine']<0 for v in tr),
        final_qr_norm=b['query_norm'],output_L2_from_Source=b['output_L2_from_Source'],
        E_total_negative_variation_pp=-sum(min(v['E_all_change_pp'],0)for v in tr),
        final_reference_loss_change=b['E_direct_objective']-a['E_direct_objective'])


def summarize(rows):
    groups={}
    for arm in ARMS:
        groups[arm]={}
        for scope in ['all','active']:
            rr=rows if scope=='all'else[r for r in rows if r['active']];fs=[r['fits'][arm]for r in rr];ds=[f['diagnosis']for f in fs];tr=[t for f in fs for t in f['transitions']if t['updated']]
            cells={a+'_'+b:sum(d['E_sign']==a and d['v_sign']==b for d in ds)for a in ['up','down','flat']for b in ['up','down','flat']}
            groups[arm][scope]=dict(queries=len(rr),final_E_v_sign_counts=cells,
                mean_final_E_change_pp=mean([d['E_change_pp']for d in ds]),mean_final_v_change_pp=mean([d['v_change_pp']for d in ds]),
                mean_final_unobserved_change_pp=mean([d['unobserved_change_pp']for d in ds]),
                queries_with_any_E_decrease=sum(d['E_decreasing_updates']>0 for d in ds),queries_with_net_E_loss=sum(d['E_sign']=='down'for d in ds),
                actual_updates=len(tr),E_decreasing_updates=sum(d['E_decreasing_updates']for d in ds),
                E_increase_GT_decrease_updates=sum(d['E_increase_GT_decrease_updates']for d in ds),
                same_rollout_surrogate_rise_updates=sum(d['same_rollout_surrogate_rise_updates']for d in ds),
                first_order_param_loss_descent_updates=sum(t.get('predicted_first_order_loss_change',0)< -1e-12 for t in tr),
                negative_mean_gradient_actual_mu_cosine=mean([t.get('negative_mean_gradient_actual_mu_cosine')for t in tr]),
                output_descent_cos_negative_updates=sum(d['output_descent_cos_negative_updates']for d in ds),
                E_total_negative_variation_mean_pp=mean([d['E_total_negative_variation_pp']for d in ds]),
                mean_query_norm=mean([d['final_qr_norm']for d in ds]),mean_output_L2_from_Source=mean([d['output_L2_from_Source']for d in ds]))
    return groups


def analyze():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    torch.set_num_threads(4);p=verify();start=time.perf_counter();old=read(ROOT/'results/stvg_error_state/2026-10-18/query_e64/RESULTS.json');oldby={r['alias']:r for r in old['rows']}
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets};rows=[];count=collections.Counter();errs=collections.defaultdict(float)
    for ii,r in enumerate(p['records']):
        assert time.perf_counter()-start<p['CPU_budget_seconds'];ds=r['dataset'];z=load(OLD/'predictions'/(r['alias']+'.pt'));q=r['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q]
        ids=np.array(z['frame_ids']);source=z['outputs']['Source_MBR'];teacher=z['E_training_teacher_float32'].double().numpy();planned=z['planned_positions'];unobs=np.setdiff1d(np.arange(len(ids)),planned)
        known=np.array([int(v)in gt for v in ids]);umask=known&~np.isin(np.arange(len(ids)),planned);gtbox=np.array([gt[int(t)]for t in ids[umask]])
        fits={}
        for arm,f in z['fits'].items():
            states=[];transitions=[];pos=f['feedback_positions'];rounds=f['rounds'];cfg=f['config']
            for k,path in enumerate(f['path']):
                b=path['boxes'].double().numpy();v=official(b,row,gt,span,z['interval'],ds);vv=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval']);assert max(abs(v[ep]-vv[ep])for ep in ['v','t','s'])<2e-10;count['independent_official_metrics']+=3
                xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
                if ds=='hc2':xy=np.maximum(xy,0)
                iou=overlap(b,teacher);u=path['state']['spatial.query_residual'].double().numpy()
                st=dict(step=k,**v,unobserved_actual_s=mean(box_iou(xy[umask],gtbox))if umask.any()else None,
                    E_IoU_all=float(iou.mean()),E_IoU_feedback=float(iou[pos].mean()),E_IoU_planned=float(iou[planned].mean()),
                    E_IoU_unobserved=mean(iou[unobs]),E_direct_objective=direct_loss(b,teacher),
                    output_L2_from_Source=float(np.linalg.norm(b-source)),output_coordinate_abs_mean_from_Source=float(abs(b-source).mean()),query_norm=float(np.linalg.norm(u)))
                states.append(st);count['states']+=1
                if k==0 or k==len(f['path'])-1:
                    expected=oldby[r['alias']]['values']['Source_MBR'if k==0 else arm]
                    for ep in ['v','t','s','unobserved_actual_s']:
                        assert expected[ep]is None and st[ep]is None or expected[ep]is not None and st[ep]is not None and abs(expected[ep]-st[ep])<2e-10;count['old_endpoint_exact']+=1
                if not k:continue
                prev=f['path'][k-1];a=states[-2];saved=rounds[k-1];b0=prev['boxes'].double().numpy();du=u-prev['state']['spatial.query_residual'].double().numpy()
                tr=dict(step_from=k-1,step_to=k,updated=saved['updated'],E_all_change_pp=100*(st['E_IoU_all']-a['E_IoU_all']),
                    E_feedback_change_pp=100*(st['E_IoU_feedback']-a['E_IoU_feedback']),v_change_pp=100*(st['v']-a['v']),
                    output_L2=float(np.linalg.norm(b-b0)),output_coordinate_abs_mean=float(abs(b-b0).mean()),query_step_norm=float(np.linalg.norm(du)),
                    recorded_reward_variance=saved.get('reward_variance'),recorded_ESS=saved.get('ESS'),
                    unobserved_change_pp=None if st['unobserved_actual_s']is None else 100*(st['unobserved_actual_s']-a['unobserved_actual_s']))
                for field,actual in [('reference_IoU_before',a['E_IoU_all']),('reference_IoU_after',st['E_IoU_all'])]:
                    err=abs(actual-saved[field]);assert err<3e-6;errs[field]=max(errs[field],err);count['saved_E_overlap_checks']+=1
                if saved['updated']:
                    grad=prev['update']['gradient'].double().numpy();tr['predicted_first_order_loss_change']=float(grad@du);tr['parameter_gradient_actual_step_cosine']=cosine(-grad,du)
                rr=saved.get('rollout')
                if rr is not None:
                    mu0=prev['chart']['mean'].double().numpy();before=surrogate(mu0,rr,cfg['sigma']);count['old_rollout_objectives']+=1
                    if 'chart'in path:mu1=path['chart']['mean'].double().numpy();unknown=False
                    elif np.all((b[pos]>0)&(b[pos]<1)):mu1=np.log(b[pos]/(1-b[pos]));unknown=False
                    else:mu1=None;unknown=True
                    tr.update(same_rollout_loss_before=before,final_logit_chart_unknown=unknown,
                        mean_sample_reward=float(rr['rewards'].double().mean()),weighted_sample_reward=float((rr['rewards'].double()*rr['weights'].double()).sum(1).mean()),
                        exact_flat_reward=not rr['informative'])
                    if mu1 is not None:
                        after=surrogate(mu1,rr,cfg['sigma']);change=after-before;tol=max(1e-5,abs(before)*1e-6)
                        gm=prev['chart']['autograd_gradient'].double().numpy();tr.update(same_rollout_loss_after=after,same_rollout_loss_change=change,
                            surrogate_rise_above_tolerance=change>tol,negative_mean_gradient_actual_mu_cosine=cosine(-gm,mu1-mu0),actual_logit_mean_step_norm=float(np.linalg.norm(mu1-mu0)))
                transitions.append(tr);count['transitions']+=1
            assert all(st['t']==states[0]['t']for st in states);count['time_invariant_paths']+=1
            out=dict(states=states,transitions=transitions,active=not f['empty'],config=cfg,feedback_positions_count=len(pos),selected_step_unchanged=f['selected_step'],GT_never_selects_step=True)
            out['diagnosis']=classify(out);fits[arm]=out
        rows.append(dict(alias=r['alias'],dataset=ds,active=z['active'],E_vs_Source_v_pp=100*(oldby[r['alias']]['values']['E_MBR']['v']-oldby[r['alias']]['values']['Source_MBR']['v']),
            original_final_values=oldby[r['alias']]['values'],fits=fits))
        put(D/'STATUS.json',dict(status='running',stage='CPU_cached_path_readback',done=ii+1,total=64,new_models=0,new_fits=0),True)
    groups={ds:summarize([r for r in rows if r['dataset']==ds])for ds in targets}
    put(P/'RESULTS.json',dict(status='complete',count=64,rows=rows,summary=groups,fixed_cases=CASES,
        configurations_unchanged=True,original_steps_and_predictions_unchanged=True,all_GT_is_posthoc_diagnosis=True,not_new_method_performance=True,
        independent_counts=dict(count),max_readback_errors=dict(errs),new_models=0,new_fits=0,new_expert_calls=0))
    verify();assert not torch.cuda.is_initialized()
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=64,new_models=0,new_fits=0,CPU_seconds=time.perf_counter()-start,
        source_predictions_and_implementation_SHA_unchanged=True,result_sha256=sha(P/'RESULTS.json')))
    print(json.dumps(dict(status='complete',count=64,checks=dict(count),seconds=time.perf_counter()-start)))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','analyze']);args=ap.parse_args();globals()[args.stage]()
