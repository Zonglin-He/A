import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import torch,numpy as np,json,hashlib
from scipy.stats import spearmanr
R=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();T=R/'outputs/layout_reward_dev';O=T/'research';torch.set_num_threads(1)
metrics=json.loads((T/'scope_hc2/private_case_mapping.json').read_text())
def cosine(a,b):
 a=a.double().flatten();b=b.double().flatten();d=a.norm()*b.norm();return float((a@b)/d)if d>0 else None

def stats(a):
 a=np.asarray([v for v in a if v is not None],dtype=float)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))

rows=[];global_rounds={a:{k:[]for k in ['gradient_norm','step_norm','reward_step_delta']}for a in ['LayoutOPD','LayoutDirect']};ess=[];wmax=[];rstd=[];rspan=[];round_adv=[]
for m in metrics:
 key=m['private_id'];p=T/'traces'/f'{key}.pt';z=torch.load(p,map_location='cpu',weights_only=False)
 with np.load(T/'predictions'/f'{key}.npz',allow_pickle=False)as q:positions=q['positions8'].tolist()
 observed=np.zeros(len(z['LayoutOPD']['final']),bool);observed[positions]=True
 row=dict(id=key,public_id=m['public_id'],trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arms={})
 for arm,a in z.items():
  rew=torch.stack([b['layout_reward']for b in a['path']]).double();boxes0=a['path'][0]['boxes'].double();boxes1=a['path'][-1]['boxes'].double()
  # Dimensionless displacement in centre/target-size, log size; not GT quality.
  disp=torch.cat(((boxes1[:,:2]-boxes0[:,:2])/boxes0[:,2:],(boxes1[:,2:]/boxes0[:,2:]).log()),-1).norm(dim=-1)
  gradients=[float(b['gradient'].norm())for b in a['rounds']];steps=[float(b['state_step'].norm())for b in a['rounds']]
  rg=rew[-1]-rew[0];state=torch.stack([b['state_step']for b in a['rounds']]).sum(0)
  x=dict(reward_initial=float(rew[0].mean()),reward_final=float(rew[-1].mean()),reward_delta=float(rg.mean()),anchors_reward_up=int((rg>1e-9).sum()),anchors_reward_down=int((rg< -1e-9).sum()),gradient_norm_mean=float(np.mean(gradients)),gradient_norm_step0=gradients[0],gradient_norm_last=gradients[-1],step_norm_mean=float(np.mean(steps)),final_state_norm=float(state.norm()),anchor_displacement=float(disp[observed].mean()),unobserved_displacement=float(disp[~observed].mean()),full_displacement=float(disp.mean()),reward_curve=rew.mean(1).tolist(),updates=a['actual_updates'],v_delta_pp=(m['values'][arm]['v']-m['values']['ERefStrong']['v'])*100,unobs_s_delta_pp=(m['values'][arm]['unobserved_actual_sIoU']-m['values']['ERefStrong']['unobserved_actual_sIoU'])*100)
  row['arms'][arm]=x
  for k,v in [('gradient_norm',gradients),('step_norm',steps),('reward_step_delta',(rew[1:].mean(1)-rew[:-1].mean(1)).tolist())]:global_rounds[arm][k].extend(v)
  if arm=='LayoutOPD':
   eq=[];sq=[];wq=[];spq=[];adq=[]
   for step,b in enumerate(a['rounds']):
    rr=b['rollout'];w=rr['weights'].double();r=rr['rewards'].double()
    eq.extend(rr['ESS'].tolist());sq.extend(r.std(-1).tolist());wq.extend(w.max(-1).values.tolist());spq.extend((r.max(-1).values-r.min(-1).values).tolist())
    adq.extend(((r*w).sum(-1)-rew[step]).tolist())
   x['ESS_mean']=float(np.mean(eq));x['weight_max_mean']=float(np.mean(wq));x['reward_candidate_std_mean']=float(np.mean(sq));x['candidate_weighted_advantage_mean']=float(np.mean(adq))
   ess.extend(eq);rstd.extend(sq);wmax.extend(wq);rspan.extend(spq);round_adv.extend(adq)
 row['initial_gradient_cosine']=cosine(z['LayoutOPD']['rounds'][0]['gradient'],z['LayoutDirect']['rounds'][0]['gradient'])
 row['final_state_cosine']=cosine(torch.stack([r['state_step']for r in z['LayoutOPD']['rounds']]).sum(0),torch.stack([r['state_step']for r in z['LayoutDirect']['rounds']]).sum(0))
 row['final_output_delta_cosine']=cosine(z['LayoutOPD']['path'][-1]['boxes']-z['LayoutOPD']['path'][0]['boxes'],z['LayoutDirect']['path'][-1]['boxes']-z['LayoutDirect']['path'][0]['boxes'])
 rows.append(row)

summary={}
for arm in ['LayoutOPD','LayoutDirect']:
 aa=[r['arms'][arm]for r in rows];out={k:stats([a[k]for a in aa])for k in ['reward_initial','reward_final','reward_delta','gradient_norm_mean','gradient_norm_step0','gradient_norm_last','step_norm_mean','final_state_norm','anchor_displacement','unobserved_displacement','full_displacement']}
 out['queries_reward_up']=sum(a['reward_delta']>1e-9 for a in aa);out['queries_reward_down']=sum(a['reward_delta']< -1e-9 for a in aa)
 out['queries_reward_up_but_v_down']=sum(a['reward_delta']>1e-9 and a['v_delta_pp']< -1e-8 for a in aa)
 out['queries_reward_up_but_unobserved_down']=sum(a['reward_delta']>1e-9 and a['unobs_s_delta_pp']< -1e-8 for a in aa)
 out['anchor_reward_up_total']=sum(a['anchors_reward_up']for a in aa);out['anchor_reward_down_total']=sum(a['anchors_reward_down']for a in aa)
 out['reward_task_spearman']={k:dict(rho=float(spearmanr([a['reward_delta']for a in aa],[a[k]for a in aa]).statistic),p=float(spearmanr([a['reward_delta']for a in aa],[a[k]for a in aa]).pvalue))for k in ['v_delta_pp','unobs_s_delta_pp']}
 out['rounds']={k:stats(v)for k,v in global_rounds[arm].items()};out['mean_reward_curve']=np.mean([a['reward_curve']for a in aa],0).tolist();summary[arm]=out
summary['candidate_sampling']=dict(ESS=stats(ess),maximum_weight=stats(wmax),candidate_reward_std=stats(rstd),candidate_reward_range=stats(rspan),candidate_weighted_advantage_over_current=stats(round_adv),ESS_over30_fraction=float(np.mean(np.array(ess)>30)),ESS_below8_fraction=float(np.mean(np.array(ess)<8)))
summary['arm_alignment']={k:stats([r[k]for r in rows])for k in ['initial_gradient_cosine','final_state_cosine','final_output_delta_cosine']}
summary['warning']='Post hoc CPU-only mechanism audit. Reward improvement is observed training objective, not proof of localization correctness. No trajectory-level candidate feedback existed. No formal VidGT read.'
result=dict(summary=summary,queries=rows)
(O/'algorithm_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
