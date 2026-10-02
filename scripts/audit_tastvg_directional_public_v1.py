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
 valid=[r for r in rows if r['rewards'] is not None];first=[r for r in valid if r['step']==0];cos=[r['rkl_preference_cosine'] for r in valid if r['rkl_preference_cosine'] is not None]
 def quant(v):return dict(mean=float(np.mean(v)),quantiles=np.quantile(v,[0,.01,.05,.5,.95,.99,1]).tolist()) if v else None
 return dict(steps=len(rows),eligible_steps=len(valid),first_eligible_steps=len(first),zero_direction_steps=sum(r['preferred_direction_norm']==0 for r in valid),no_op_reasons={reason:sum(r['no_op_reason']==reason for r in valid) for reason in ['no_rank_direction','zero_rkl_magnitude']},matched_steps=sum(r['magnitude_match_applicable'] for r in valid),actual_step=quant([r['actual_step_norm'] for r in valid]),counterfactual_rkl_step=quant([r['counterfactual_rkl_step_norm'] for r in valid]),cosine=quant(cos),cosine_negative=sum(v<0 for v in cos),probe_ratio=quant([r['executed_to_probe_radius'] for r in valid]),chosen_pair_GT_agrees=sum(r['critic_chosen_pair_GT_agrees'] for r in valid),chosen_pair_GT_neutral=sum(r['rank_contrasts'][r['top_pair']]*r['critic_chosen_pair_GT_delta']==0 for r in valid),chosen_pair_GT_disagrees=sum(r['rank_contrasts'][r['top_pair']]*r['critic_chosen_pair_GT_delta']<0 for r in valid),unique_useful_top_harm_first=sum(r['unique_useful_top_harm'] for r in first),unique_useful_top_severe_harm_first=sum(r['unique_useful_top_harm'] and r['delta_update']<-.05 for r in first),loss_down_GT_harm=sum(r['loss_decreased'] and r['delta_update']< -1e-12 for r in valid),GT_harm_steps=sum(r['delta_update']< -1e-12 for r in valid),GT_gain_steps=sum(r['delta_update']>1e-12 for r in valid),severe_GT_harm_steps=sum(r['delta_update']<-.05 for r in valid),post_fixed_time_delta=quant([r['delta_update'] for r in valid]))
def run(base):
 base=Path(base);checks=0;total=0
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]+['delta_inherited_boxes','delta_inherited_interval','delta_temporal_rerank','delta_post_fixed_time']
 for ds in ['vidstg','hc2']:
  folder=base/ds;cfg=read(folder/'CONFIG.json');assert cfg['completed_arms']==['A','E','F'] and cfg['sources']==cfg['queries']==32;assert cfg['arrivals_per_arm']==384 and cfg['expert_fraction']==.25;data={};scores={};diagnosis={}
  for arm in ['A','E','F']:
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
   request=read(out/'REQUEST.json');assert request['params']==cfg['params'] and request['arm']==arm;method=request['method'];mode={'A':'rkl','E':'rank_directional','F':'top_directional'}[arm];assert method['actuation']==mode
   for s in st:
    checks+=equal(s['delta_update'],s['post_v']-s['pre_v']);assert cfg['orders'][s['order']][s['arrival']]==s['parent'] and s['arrival']%4==0
    if s['rewards'] is None:assert not s['updated'];continue
    rank=ranks(s['rewards']);c=rank[2::2]-rank[1::2];top=int(np.argmax(np.abs(c)));checks+=equal(s['teacher_ranks'],rank.tolist());checks+=equal(s['rank_contrasts'],c.tolist());checks+=equal(s['top_pair'],top)
    checks+=equal(s['reward_spread'],float(np.ptp(s['rewards'])));checks+=equal(s['strength'],1.)
    vn=float(np.linalg.norm(c)) if arm!='F' else float(abs(np.sign(c[top])));assert abs(vn-s['preferred_direction_norm'])<1e-10;checks+=1
    desired=cfg['params']['lr']*s['global_gradient_norm'];assert abs(desired-s['counterfactual_rkl_step_norm'])<1e-10;checks+=1
    if arm!='A' and vn==0:assert s['actual_step_norm']==0 and not s['updated'] and s['no_op_reason']=='no_rank_direction' and not s['magnitude_match_applicable'];checks+=4
    else:assert abs(desired-s['actual_step_norm'])<2e-6+1e-5*desired and s['magnitude_match_applicable'];checks+=2
    assert s['projection_factor']==1.;checks+=equal(s['executed_to_probe_radius'],s['actual_step_norm']/cfg['probe_radius'])
    if s['rkl_preference_cosine'] is not None:
     signed=np.array(s['rkl_probe_axis_components']);weights=c if arm!='F' else np.eye(4)[top]*np.sign(c[top]);expected=float(np.dot(signed,weights)/(s['global_gradient_norm']*vn));assert abs(expected-s['rkl_preference_cosine'])<1e-10 and -1.00000000001<=expected<=1.00000000001;checks+=2
    pv=s['candidate_v'];checks+=equal(s['critic_chosen_pair_GT_delta'],pv[1+2*top]-pv[2+2*top]);checks+=equal(s['critic_chosen_pair_GT_agrees'],bool(c[top]*s['critic_chosen_pair_GT_delta']>0))
    checks+=equal(s['rank_weighted_pair_GT_contrast'],float(np.dot(c,np.array(pv[1::2])-np.array(pv[2::2]))))
    checks+=equal(s['unique_useful_top_harm'],len(s['teacher_top_indices'])==1 and pv[s['teacher_top_indices'][0]]>pv[0]+1e-12 and s['post_v']<pv[0]-1e-12)
   diagnosis[arm]={g:dir_stats([r for r in st if (r['condition']=='clean')==(g=='clean')]) for g in ['clean','corruption']}
   audit=read(out/'AUDIT.json');assert audit['status']=='pass' and audit['state_links']==384 and audit['GT_after_global_barrier'] and not audit['GPU_initialized'];assert audit['max_dense_metric_error']<1e-10;checks+=5
  contrasts={};gross={};tails={}
  for arm in ['A','E','F']:
   paired=[]
   for a,b in zip(data[arm],data['A']):
    assert all(a[k]==b[k] for k in ['parent','source_id','condition','order','arrival','expert_scheduled'])
    for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:checks+=equal(a['Frozen_'+k],b['Frozen_'+k])
    d=a['Ours_m_vIoU']-b['Ours_m_vIoU'];paired.append({**a,**{'delta_'+k:a['Ours_'+k]-b['Ours_'+k] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']},'delta_vs_A':d,'gross_gain_vs_A':max(0,d),'gross_loss_vs_A':max(0,-d)})
   if arm!='A':contrasts[arm]={g:{sub:summary(filtered(paired,g,sub),['delta_'+k for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
   future=filtered(paired,'corruption','nonexpert');gross[arm]=summary(future,['delta_vs_A','gross_gain_vs_A','gross_loss_vs_A']);v=np.array([r['delta_vs_A'] for r in future]);tails[arm]=dict(arrivals=len(future),harm_over5pp=int((v<-.05).sum()),gain_over5pp=int((v>.05).sum()),delta_quantiles=np.quantile(v,[0,.01,.05,.5,.95,.99,1]).tolist())
  checks+=equal(contrasts,read(folder/'PAIRED.json'));checks+=equal(gross,read(folder/'GROSS_FUTURE_VS_A.json'));checks+=equal(tails,read(folder/'NEGATIVE_TAILS.json'));checks+=equal(diagnosis,read(folder/'DIRECTION_DIAGNOSIS.json'))
  winner='A'
  for arm in ['E','F']:
   if scores[arm]>scores[winner]+1e-12:winner=arm
  decision=read(folder/'DEVELOPMENT_DECISION.json');assert decision['arm']==winner;checks+=equal(scores,decision['objectives'])
 movement=read(base/'FUNCTIONAL_MOVEMENT_AUDITED.json');functional=read(base/'FUNCTIONAL_STEP_ROWS.json')
 assert movement['GT_read'] is False and movement['model_execution'] is False and movement['GPU_initialized'] is False
 for ds in ['vidstg','hc2']:
  for arm in ['A','E','F']:
   rows=functional[ds][arm];steps=read(base/ds/arm/'SPATIAL_STEP_ROWS.json');lookup={(s['condition'],s['order'],s['arrival'],s['step']):s for s in steps if s['condition']!='clean' and s['rewards'] is not None}
   assert len(rows)==len(lookup) and len({(r['condition'],r['order'],r['arrival'],r['step']) for r in rows})==len(rows)
   for r in rows:
    st=lookup[r['condition'],r['order'],r['arrival'],r['step']];checks+=equal(r['actual_parameter_step'],st['actual_step_norm']);assert r['box_mean_absolute_coordinate_change']>=0
    checks+=equal(r['box_change_per_parameter_norm'],r['box_mean_absolute_coordinate_change']/r['actual_parameter_step'] if r['actual_parameter_step']>0 else None)
    checks+=equal(r['RKL_norm_fraction_in_four_axes'],st['rkl_probe_subspace_norm']/st['global_gradient_norm'] if st['global_gradient_norm']>0 else None)
   statistics={k:dict(mean=float(np.mean([r[k] for r in rows if r[k] is not None])),median=float(np.median([r[k] for r in rows if r[k] is not None]))) for k in ['box_mean_absolute_coordinate_change','actual_parameter_step','box_change_per_parameter_norm','RKL_norm_fraction_in_four_axes']}
   checks+=equal(dict(eligible_corrupt_steps=len(rows),statistics=statistics),movement['datasets'][ds][arm])
 assert total==2304
 print(json.dumps(dict(status='pass',scalar_checks=checks,arrivals=total,GT_read=False,model_execution=False,scope='coverage, anonymous arithmetic/rank/sign/norm/cosine, source-macro paired bootstrap, gross/tails/direction diagnosis and sealed development decisions; functional movement anonymous means/medians'),indent=2))
if __name__=='__main__':run(sys.argv[1])
