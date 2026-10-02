"""Independent anonymous readback: rankings, matched differences and bootstrap."""
import sys,json,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def read(p):return json.loads(Path(p).read_text())
def audit(folder):
 folder=Path(folder);checks=0
 def close(a,b,tol=1e-12):
  nonlocal checks
  assert abs(a-b)<tol,(a,b);checks+=1
 def aggregate(rows,f):
  grouped=collections.defaultdict(list)
  for r in rows:grouped[r['source_id'],r['order'],r['condition']].append(r[f])
  byorder=collections.defaultdict(list)
  for (s,o,c),v in grouped.items():byorder[s,o].append(np.mean(v))
  sources=collections.defaultdict(list);orders=collections.defaultdict(list)
  for (s,o),v in byorder.items():x=np.mean(v);sources[s].append(x);orders[o].append(x)
  vector=np.array([np.mean(sources[s]) for s in sorted(sources)]);rng=np.random.default_rng(20261001)
  boot=np.concatenate([vector[rng.integers(0,len(vector),(100,len(vector)))].mean(1) for _ in range(100)])
  return vector.mean(),np.quantile(boot,[.025,.975])
 def summary(rows,z):
  assert z['cells']==len(rows) and z['sources']==len({r['source_id'] for r in rows})
  for f,m in z['metrics'].items():
   take=[r for r in rows if r.get(f) is not None and (not f.startswith('T_') or r.get('T_available',True))];v,ci=aggregate(take,f);close(v,m['mean']);close(ci[0],m['ci95'][0]);close(ci[1],m['ci95'][1]);close(np.mean([r[f] for r in take]),m['query_macro'])
   if 'available_cells' in m:assert m['available_cells']==len(take) and m['available_sources']==len({r['source_id'] for r in take})
 def rank(vals):
  order=sorted(range(len(vals)),key=lambda i:-vals[i]);out=[0.]*len(vals);j=0
  while j<len(order):
   k=j+1
   while k<len(order) and vals[order[j]]-vals[order[k]]<=1e-12:k+=1
   for i in order[j:k]:out[i]=(j+k-1)/2
   j=k
  return out
 for ds in ['hc2','vidstg']:
  rows=read(folder/ds/'ONLINE_ROWS.json');z=read(folder/ds/'ONLINE_SUMMARY.json');assert len(rows)==384 and len({r['source_id'] for r in rows})==32
  for r in rows:
   assert r['expert_scheduled']==(r['arrival']%4==0);checks+=1
   for f in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:
    for a,b,name in [('R','A','delta_R_A'),('R','Frozen','delta_R_Frozen'),('A','Frozen','delta_A_Frozen')]:close(r[name+'_'+f],r[a+'_'+f]-r[b+'_'+f])
   assert r['future_both']==(not r['expert_scheduled'] and r['A_has_prior_write'] and r['R_has_prior_write']);checks+=1
  for group,zz in z.items():
   rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')]
   for sub,zzs in zz.items():
    take=[r for r in rr if sub=='all' or (r['future_both'] if sub=='future_nonexpert' else r['expert_scheduled']==(sub=='expert'))];summary(take,zzs)
    dv=np.array([r['delta_R_A_m_vIoU'] for r in take]);close(np.maximum(dv,0).mean()*100,zzs['gross_gain_pp']);close(-np.minimum(dv,0).mean()*100,zzs['gross_loss_pp'])
  for arm in ['A','R']:
   srows=read(folder/ds/arm/'SPATIAL_STEP_ROWS.json')
   for s in srows:
    u=s['candidate_v'];rewards=s['rewards'];close(s['delta_update'],s['post_v']-s['pre_v']);close(s['pre_v'],u[0]);close(s['oracle_regret'],max(u)-(s.get('selected_target_GT_v',u[0])))
    if rewards is not None:
     selected=max(range(9),key=lambda i:rewards[i]);assert s['selected_index']==selected;checks+=1;close(s['selected_target_GT_v'],u[selected]);close(s['selected_target_GT_delta'],u[selected]-u[0])
     assert s['teacher_ranks']==rank(rewards);checks+=1
     top=[i for i,v in enumerate(rewards) if v>=max(rewards)-1e-12];assert s['teacher_top_indices']==top;checks+=1
     assert s['unique_useful_top_harm']==(len(top)==1 and u[top[0]]>u[0]+1e-12 and s['post_v']<u[0]-1e-12);checks+=1
    rr=[0.]*9 if rewards is None else rewards;correct=0.;strict=decisive=0
    for i in range(9):
     for j in range(i+1,9):
      d=u[i]-u[j];q=rr[i]-rr[j];sign=lambda x:int(x>1e-12)-int(x< -1e-12)
      if sign(d):strict+=1;correct+=.5 if sign(q)==0 else float(sign(q)==sign(d));decisive+=sign(q)!=0
    if strict:close(correct/strict,s['pairwise']);close(decisive/strict,s['decisive_coverage'])
    else:assert s['pairwise'] is None
  diag=read(folder/ds/'PIPELINE_DIAGNOSIS.json')
  for group in ['corruption','clean']:
   for arm in ['A','R']:
    ss=[s for s in read(folder/ds/arm/'SPATIAL_STEP_ROWS.json') if (s['condition']!='clean')==(group=='corruption')];first=[s for s in ss if s['step']==0];d=diag[group][arm]
    assert d['steps']==len(ss) and d['expert_arrivals']==len(first)
    assert d['updated_steps']==sum(s['updated'] for s in ss) and d['empty_arrivals']==sum(s['rewards'] is None for s in first)
    assert d['no_scored_valid']==sum(s['raw_reference_GT_frames']==0 for s in first)
    assert d['unique_useful_top_harm']==sum(s['unique_useful_top_harm'] for s in ss) and d['selected_useful_harm']==sum(s.get('selected_useful_harm',False) for s in ss)
    assert d['loss_down_GT_down']==sum(s['loss_decreased'] and s['delta_update']< -1e-12 for s in ss)
    val=np.array([s['delta_update'] for s in ss]);assert d['local_positive_steps']==int((val>1e-12).sum()) and d['local_negative_steps']==int((val< -1e-12).sum())
    close(np.maximum(val,0).mean()*100,d['local_step_gross_gain_pp']);close(-np.minimum(val,0).mean()*100,d['local_step_gross_loss_pp'])
    summary([s for s in first if s['pairwise'] is not None],d['first_step_ranking'])
    rr=[r for r in read(folder/ds/arm/'ROWS.json') if (r['condition']!='clean')==(group=='corruption') and r['expert_scheduled']];summary(rr,d['net_arrival_update'])
  cases=read(folder/ds/'ONLINE_CASES.json');ordered=sorted([r for r in rows if r['condition']!='clean'],key=lambda r:r['delta_R_A_m_vIoU']);assert cases['negative']==ordered[:5] and cases['positive']==ordered[-5:]
  if (folder/ds/'LOCAL_UPDATE_PAIRED_ROWS.json').exists():
   ar={a:read(folder/ds/a/'ROWS.json') for a in ['A','R']};lookup={a:{(r['source_id'],r['condition'],r['order'],r['arrival']):r for r in ar[a]} for a in ar};pr=read(folder/ds/'LOCAL_UPDATE_PAIRED_ROWS.json');pz=read(folder/ds/'LOCAL_UPDATE_PAIRED_SUMMARY.json')
   assert len(pr)==96
   for r in pr:
    key=tuple(r[k] for k in ['source_id','condition','order','arrival']);a,z=lookup['A'][key],lookup['R'][key];assert a['expert_scheduled'] and z['expert_scheduled'];close(r['delta_R_A_local_update'],z['delta_post_fixed_time']-a['delta_post_fixed_time'])
   for group,z in pz.items():summary([r for r in pr if (r['condition']!='clean')==(group=='corruption')],z)
  tr=read(folder/ds/'TOKEN_ROWS.json');tz=read(folder/ds/'TOKEN_SUMMARY.json');assert len(tr)==30 and len({r['source_id'] for r in tr})==10
  for r in tr:
   u=r['utilities'];assert len(u)==9 and r['T_available']==all(v is not None for v in r['T_rewards']);checks+=1
   for arm in ['S','T']:
    scores=r[arm+'_rewards'] if arm=='S' or r['T_available'] else None;vals=[0.]*9 if scores is None else scores;sel=0 if scores is None else max(range(9),key=lambda i:vals[i]);assert r[arm+'_selected']==sel;checks+=1
    close(r[arm+'_v'],u[sel]);close(r[arm+'_gain'],u[sel]-u[0]);close(r[arm+'_regret'],max(u)-u[sel]);strict=decisive=0;correct=0.
    for p in r[arm+'_pairs']:
     i,j=p['i'],p['j'];close(p['reward_difference'],vals[i]-vals[j]);close(p['utility_difference'],u[i]-u[j]);sign=lambda d:int(d>1e-12)-int(d< -1e-12);q,g=sign(vals[i]-vals[j]),sign(u[i]-u[j]);assert p['expert_sign']==q and p['GT_sign']==g;checks+=2
     if g:strict+=1;decisive+=q!=0;correct+=.5 if q==0 else float(q==g)
    if strict and (arm=='S' or r['T_available']):close(r[arm+'_pairwise'],correct/strict);close(r[arm+'_decisive'],decisive/strict)
    else:assert r[arm+'_pairwise'] is None
   if r['T_available']:close(r['delta_T_S_v'],r['T_v']-r['S_v'])
  for group,zs in tz.items():
   rr=[r for r in tr if (r['condition']!='clean')==(group=='corruption')];summary(rr,zs)
   totals=collections.Counter()
   for r in rr:
    computed=collections.Counter({k:0 for k in ['strict_pairs','S_wrong_T_right','S_right_T_wrong','both_right','both_wrong','either_tie']})
    for a,b in zip(r['S_pairs'],r['T_pairs']):
     if not a['GT_sign'] or not r['T_available']:continue
     computed['strict_pairs']+=1;s,t,g=a['expert_sign'],b['expert_sign'],a['GT_sign'];key='either_tie' if s==0 or t==0 else 'both_right' if s==t==g else 'both_wrong' if s!=g and t!=g else 'S_wrong_T_right' if t==g else 'S_right_T_wrong';computed[key]+=1
    assert dict(computed)==r['complementarity'];totals.update(computed)
    a=r['event_object_diagnostic'];assert a['qualified_same_object_inside_outside']==bool(a['labelled_accurate_inside'] and a['labelled_accurate_outside'])
    if not a['qualified_same_object_inside_outside']:assert a['pair_accuracy'] is None and a['AUROC'] is None
   assert dict(totals)==zs['complementarity'] and zs['token_all9_available']==sum(r['T_available'] for r in rr) and zs['token_unavailable_cells']==sum(not r['T_available'] for r in rr)
   assert zs['outside_unavailable_candidates']==sum(r['outside_unavailable_candidates'] for r in rr)
  eligible=read(folder/'ST_ELIGIBILITY.json')[ds]['eligible'];m=tz['corruption']['metrics'];assert eligible==('T_pairwise' in m and m['T_pairwise']['mean']>.5 and m['T_gain']['mean']>0);checks+=1
  if (folder/ds/'TOKEN_CASES.json').exists():
   tc=read(folder/ds/'TOKEN_CASES.json');ordered=sorted([r for r in tr if r['condition']!='clean' and r['delta_T_S_v'] is not None],key=lambda r:r['delta_T_S_v']);assert tc['negative']==ordered[:5] and tc['positive']==[r for r in ordered if r['delta_T_S_v']>0][-5:]
  if eligible:
   fused=read(folder/ds/'ST_ROWS.json');fz=read(folder/ds/'ST_SUMMARY.json');index={r['cell']:r for r in tr}
   for r in fused:
    t=index[r['cell']];sr=rank([0.]*9 if t['S_rewards'] is None else t['S_rewards']);tt=rank(t['T_rewards']);ranks=[a+b for a,b in zip(sr,tt)];assert r['S_ranks']==sr and r['T_ranks']==tt and r['ST_ranks']==ranks;checks+=3
    selected=min(range(9),key=lambda i:ranks[i]);assert selected==r['ST_selected'];checks+=1;close(r['ST_v'],r['utilities'][selected]);close(r['delta_ST_S_v'],r['ST_v']-t['S_v']);close(r['delta_ST_T_v'],r['ST_v']-t['T_v'])
   for group,zs in fz.items():summary([r for r in fused if (r['condition']!='clean')==(group=='corruption')],zs)
 result=dict(status='pass',scalar_checks=checks,raw_private_assets_required=False);print(json.dumps(result));return result
if __name__=='__main__':audit(Path(sys.argv[1]))
