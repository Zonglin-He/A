"""Self-contained reconstruction from published anonymous P0 scalars only."""
import sys,json,collections
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def summary(rows,fields):
 if not rows:return dict(sources=0,cells=0,metrics={})
 groups=collections.defaultdict(list)
 for r in rows:groups[r['source_id'],r['order'],r['condition']].append([r[k] for k in fields])
 so=collections.defaultdict(list)
 for (s,o,c),v in groups.items():so[s,o].append(np.mean(v,axis=0))
 sources=collections.defaultdict(list);orders=collections.defaultdict(list)
 for (s,o),v in so.items():z=np.mean(v,axis=0);sources[s].append(z);orders[o].append(z)
 matrix=np.array([np.mean(sources[s],axis=0) for s in sorted(sources)]);rng=np.random.default_rng(20261001);boots=[]
 for i in range(0,10000,100):boots.append(matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1))
 ci=np.quantile(np.concatenate(boots),[.025,.975],axis=0);ov=np.array([np.mean(orders[o],axis=0) for o in sorted(orders)])
 return dict(sources=len(matrix),cells=len(rows),metrics={k:dict(mean=float(matrix[:,j].mean()),ci95=ci[:,j].tolist(),query_macro=float(np.mean([r[k] for r in rows])),order_values=ov[:,j].tolist(),order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,harm_gt5pp_sources=int((matrix[:,j]<-.05).sum()) if k.startswith('delta_') else None) for j,k in enumerate(fields)})
def equal(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return sum(equal(a[k],b[k]) for k in a)
 if isinstance(a,list):assert len(a)==len(b);return sum(equal(x,y) for x,y in zip(a,b))
 if a is None or isinstance(a,(str,bool)):assert a==b
 else:assert np.isclose(a,b,atol=1e-12,rtol=1e-10),(a,b)
 return 1
def ranks(values):
 a=np.array(values,float);order=np.argsort(-a,kind='stable');r=np.empty(len(a));i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[i]]-a[order[j]]<=1e-12:j+=1
  r[order[i:j]]=(i+j-1)/2;i=j
 return r
def filtered(rows,g,s):return [r for r in rows if (r['condition']=='clean')==(g=='clean') and (s=='all' or r['expert_scheduled']==(s=='expert'))]
def dir_stats(rows):
 valid=[r for r in rows if r['rewards'] is not None];first=[r for r in valid if r['step']==0]
 def quant(values):
  return dict(mean=float(np.mean(values)),quantiles=np.quantile(values,[0,.01,.05,.5,.95,.99,1]).tolist()) if len(values) else None
 cos=[r['output_space_cosine'] for r in valid if r['output_space_cosine'] is not None]
 return dict(steps=len(rows),eligible_steps=len(valid),first_eligible_steps=len(first),
  no_op_reasons={reason:sum(r['no_op_reason']==reason for r in valid) for reason in ['flat_rewards','central_selected','zero_selected_gradient','zero_rkl_magnitude']},
  matched_steps=sum(r['magnitude_match_applicable'] for r in valid),
  actual_step=quant([r['actual_step_norm'] for r in valid]),counterfactual_rkl_step=quant([r['counterfactual_rkl_step_norm'] for r in valid]),
  functional_box_movement=quant([r['box_movement_mean_abs'] for r in valid]),
  output_cosine=quant(cos),cosine_positive=sum(v>0 for v in cos),cosine_negative=sum(v<0 for v in cos),cosine_undefined=len(valid)-len(cos),
  selected_distance_decreases=sum(r['selected_distance_decreased'] for r in valid),
  selected_distance_increases=sum(r['selected_loss_after']>r['selected_loss_before']+1e-12 for r in valid),
  selected_central=sum(r['selected_index']==0 for r in valid),flat_rewards=sum(r['selected_flat_noop'] for r in valid),
  unique_useful_top_harm_first=sum(r['unique_useful_top_harm'] for r in first),
  unique_useful_top_severe_harm_first=sum(r['unique_useful_top_harm'] and r['delta_update']<-.05 for r in first),
  selected_useful_harm=sum(r['selected_useful_harm'] for r in valid),
  selected_useful_harm_first=sum(r['selected_useful_harm'] for r in first),
  selected_distance_down_GT_harm=sum(r['selected_distance_decreased'] and r['delta_update']< -1e-12 for r in valid),
  loss_down_GT_harm=sum(r['loss_decreased'] and r['delta_update']< -1e-12 for r in valid),
  GT_harm_steps=sum(r['delta_update']< -1e-12 for r in valid),GT_gain_steps=sum(r['delta_update']>1e-12 for r in valid),
  severe_GT_harm_steps=sum(r['delta_update']<-.05 for r in valid),
  local_gross_gain=float(np.mean([max(0,r['delta_update']) for r in valid])) if valid else None,
  local_gross_loss=float(np.mean([max(0,-r['delta_update']) for r in valid])) if valid else None,
  post_fixed_time_delta=quant([r['delta_update'] for r in valid]))
def run(base):
 base=Path(base);checks=0;total=0
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]+['delta_inherited_boxes','delta_inherited_interval','delta_temporal_rerank','delta_post_fixed_time']
 for ds in ['vidstg','hc2']:
  folder=base/ds;cfg=read(folder/'CONFIG.json');assert cfg['completed_arms']==['A','G'] and cfg['sources']==cfg['queries']==32;assert cfg['arrivals_per_arm']==384 and cfg['expert_fraction']==.25;data={};scores={};diagnosis={}
  for arm in ['A','G']:
   out=folder/arm;rows=read(out/'ROWS.json');data[arm]=rows;assert len(rows)==384 and len({r['source_id'] for r in rows})==32
   assert len({(r['condition'],r['order'],r['arrival']) for r in rows})==384
   for r in rows:
    assert cfg['orders'][r['order']][r['arrival']]==r['parent'] and r['expert_scheduled']==(r['arrival']%4==0)
    for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:checks+=equal(r['delta_'+k],r['Ours_'+k]-r['Frozen_'+k])
    checks+=equal(r['final_v'],r['Ours_m_vIoU'])
   ss={g:{sub:summary(filtered(rows,g,sub),fields) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']};checks+=equal(ss,read(out/'SUMMARY.json'));scores[arm]=ss['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean'];total+=len(rows)
   st=read(out/'SPATIAL_STEP_ROWS.json');assert 96<=len(st)<=96*cfg['params']['steps'];trace=collections.defaultdict(list)
   for s in st:trace[s['condition'],s['order'],s['arrival']].append(s)
   assert len(trace)==96
   for seq in trace.values():
    assert [s['step'] for s in seq]==list(range(len(seq))) and len(seq)<=cfg['params']['steps']
    if len(seq)<cfg['params']['steps']:assert seq[-1]['rewards'] is None
   request=read(out/'REQUEST.json');assert request['params']==cfg['params'] and request['arm']==arm;method=request['method'];mode={'A':'rkl','G':'selected_rollout'}[arm];assert method['actuation']==mode
   for s in st:
    checks+=equal(s['delta_update'],s['post_v']-s['pre_v']);assert cfg['orders'][s['order']][s['arrival']]==s['parent'] and s['arrival']%4==0
    if s['rewards'] is None:assert not s['updated'];continue
    rank=ranks(s['rewards']);checks+=equal(s['teacher_ranks'],rank.tolist())
    reward_values=np.asarray(s['rewards']);selected=int(np.argmax(reward_values));flat=bool(np.ptp(reward_values)==0.)
    checks+=equal(selected,s['selected_index']);checks+=equal(flat,s['selected_flat_noop'])
    checks+=equal(s['reward_spread'],float(np.ptp(reward_values)));checks+=equal(s['strength'],1.)
    desired=cfg['params']['lr']*s['global_gradient_norm'];assert abs(desired-s['counterfactual_rkl_step_norm'])<1e-10;checks+=1
    if arm=='G':
     sn=s['selected_gradient_norm'];reason='flat_rewards' if flat else 'central_selected' if selected==0 else 'zero_selected_gradient' if sn==0. else 'zero_rkl_magnitude' if desired==0. else None
     checks+=equal(reason,s['no_op_reason']);checks+=equal(s['srd_backward_calls'],int(selected!=0 and not flat))
     if reason is not None:assert s['actual_step_norm']==0 and not s['updated'] and not s['magnitude_match_applicable'];checks+=3
     else:assert abs(desired-s['actual_step_norm'])<2e-6+1e-5*desired and s['magnitude_match_applicable'];checks+=2
    else:assert s['srd_backward_calls']==0 and abs(desired-s['actual_step_norm'])<2e-6+1e-5*desired;checks+=2
    assert s['rkl_backward_calls']==1 and s['projection_factor']==1.;checks+=equal(s['executed_to_probe_radius'],s['actual_step_norm']/cfg['probe_radius'])
    assert s['box_movement_mean_abs']>=0 and s['box_movement_l2']>=0 and s['selected_target_movement_l2']>=0;checks+=3
    defined=s['box_movement_l2']>0 and s['selected_target_movement_l2']>0
    checks+=equal(defined,s['output_space_cosine_defined'])
    if defined:assert s['output_space_cosine'] is not None and -1.00000000001<=s['output_space_cosine']<=1.00000000001;checks+=1
    else:assert s['output_space_cosine'] is None;checks+=1
    pv=s['candidate_v'];checks+=equal(s['selected_target_GT_delta'],pv[selected]-pv[0]);checks+=equal(s['selected_target_GT_v'],pv[selected]);checks+=equal(s['candidate_oracle_v'],max(pv))
    checks+=equal(s['selected_useful_harm'],pv[selected]>pv[0]+1e-12 and s['post_v']<pv[0]-1e-12)
    checks+=equal(s['selected_distance_decreased'],s['selected_loss_after']<s['selected_loss_before']-1e-12)
    checks+=equal(s['unique_useful_top_harm'],len(s['teacher_top_indices'])==1 and pv[s['teacher_top_indices'][0]]>pv[0]+1e-12 and s['post_v']<pv[0]-1e-12)
   diagnosis[arm]={g:dir_stats([r for r in st if (r['condition']=='clean')==(g=='clean')]) for g in ['clean','corruption']}
   audit=read(out/'AUDIT.json');assert audit['status']=='pass' and audit['state_links']==384 and audit['GT_after_global_barrier'] and not audit['GPU_initialized'];assert audit['max_dense_metric_error']<1e-10;checks+=5
  contrasts={};gross={};tails={}
  for arm in ['A','G']:
   paired=[]
   for a,b in zip(data[arm],data['A']):
    assert all(a[k]==b[k] for k in ['parent','source_id','condition','order','arrival','expert_scheduled'])
    for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:checks+=equal(a['Frozen_'+k],b['Frozen_'+k])
    d=a['Ours_m_vIoU']-b['Ours_m_vIoU'];paired.append({**a,**{'delta_'+k:a['Ours_'+k]-b['Ours_'+k] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']},'delta_vs_A':d,'gross_gain_vs_A':max(0,d),'gross_loss_vs_A':max(0,-d)})
   if arm!='A':contrasts[arm]={g:{sub:summary(filtered(paired,g,sub),['delta_'+k for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
   future=filtered(paired,'corruption','nonexpert');gross[arm]=summary([{**r,'gross_gain_vs_Frozen':max(0,r['Ours_m_vIoU']-r['Frozen_m_vIoU']),'gross_loss_vs_Frozen':max(0,r['Frozen_m_vIoU']-r['Ours_m_vIoU'])} for r in future],['delta_vs_A','gross_gain_vs_A','gross_loss_vs_A','gross_gain_vs_Frozen','gross_loss_vs_Frozen']);v=np.array([r['delta_vs_A'] for r in future]);tails[arm]=dict(arrivals=len(future),harm_over5pp=int((v<-.05).sum()),gain_over5pp=int((v>.05).sum()),delta_quantiles=np.quantile(v,[0,.01,.05,.5,.95,.99,1]).tolist())
  checks+=equal(contrasts,read(folder/'PAIRED.json'));checks+=equal(gross,read(folder/'GROSS_FUTURE_VS_A.json'));checks+=equal(tails,read(folder/'NEGATIVE_TAILS.json'));checks+=equal(diagnosis,read(folder/'DIRECTION_DIAGNOSIS.json'))
  winner='A'
  for arm in ['G']:
   if scores[arm]>scores[winner]+1e-12:winner=arm
  decision=read(folder/'DEVELOPMENT_DECISION.json');assert decision['arm']==winner;checks+=equal(scores,decision['objectives'])
 assert total==1536
 print(json.dumps(dict(status='pass',scalar_checks=checks,arrivals=total,GT_read=False,model_execution=False,scope='coverage, anonymous selection/no-op/target/norm/cosine, source-macro paired bootstrap, gross/tails/direction diagnosis and sealed development decisions; full functional direction and target-distance anonymous statistics'),indent=2))
if __name__=='__main__':run(sys.argv[1])
