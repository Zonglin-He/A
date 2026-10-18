"""Independent finite-game, geometry, scope and anonymous aggregation audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,itertools,json,math,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
D=ROOT/'artifacts/stvg_error_state_hc_diagnosis32_v1';OLD=ROOT/'artifacts/stvg_error_state32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/hc_diagnosis32';REVIEW=ROOT/'artifacts/stvg_dino_error_attribution32_v1'
GROUPS=['WI','CP','MX','CI','UN'];checks=0
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 assert sha(p)==read(Path(p).with_suffix('.receipt.json'))['sha256']
 return torch.load(p,map_location='cpu',weights_only=False)
def same(a,b,tol=2e-10):
 global checks
 checks+=1
 if a is None or b is None:assert a is b,(a,b)
 else:assert abs(float(a)-float(b))<tol,(a,b)
def paired(v,saved):
 v=np.array([x for x in v if x is not None]);same(len(v),saved['parents'])
 if not len(v):assert saved['mean']is None and saved['CI95']is None;return
 same(np.mean(v),saved['mean']);assert np.array_equal(v,np.asarray(saved['per_parent']))
 for k,n in [('positive',np.sum(v>1e-10)),('negative',np.sum(v< -1e-10)),('neutral',np.sum(abs(v)<=1e-10))]:same(n,saved[k])
 ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
 for a,b in zip(np.quantile(v[ix].mean(1),[.025,.975]),saved['CI95']):same(a,b)
def chart(b):return np.column_stack((b[:,:2],np.log(b[:,2:])))
def correction(b,pos,target):
 if not len(pos):return b.copy(),np.zeros(4)
 residual=chart(target)-chart(b)[pos]
 delta=np.linalg.lstsq(np.ones((len(pos),1)),residual,rcond=None)[0][0]
 return np.column_stack((b[:,:2]+delta[:2],b[:,2:]*np.exp(delta[2:]))),delta
def permutation_shapley(game,n):
 if not n:return []
 total=np.zeros(n);perms=list(itertools.permutations(range(n)))
 for perm in perms:
  selected=0
  for j in perm:
   new=selected|(1<<j);total[j]+=game[new]-game[selected];selected=new
 return total/len(perms)
def endpoint(row,base,name,ep):
 x=row['values'][base][name];return x[ep]if ep in ['v','s']else x['supports'][ep]['s']

def main():
 torch.set_num_threads(1);complete=read(D/'CPU_COMPLETION.json');assert complete['count']==32
 assert sha(P/'RESULTS.json')==complete['result_sha256']
 plan=read(D/'PLAN_LOCK.json');lock=read(D/'RUNTIME_LOCK.json')
 for p,h in {**lock['pins'],**lock['registries'],**lock['inputs']}.items():assert sha(ROOT/p)==h
 projection=lambda rows:[(r['alias'],r['dataset'],r['query_ordinal'],r['parent'])for r in rows]
 roster=projection(plan['records'])
 assert roster==projection(read(OLD/'PLAN_LOCK.json')['records'])
 assert roster==projection(read(ROOT/'artifacts/stvg_joint_event_critic_v1/PLAN_LOCK.json')['target'])
 assert roster==projection(read(ROOT/'artifacts/stvg_dino_rollback32_v1/PLAN_LOCK.json')['records'])
 reviewlock=read(REVIEW/'PROTOCOL_LOCK.json')
 for p,h in reviewlock['inputs'].items():assert sha(ROOT/p)==h
 review=read(REVIEW/'VISUAL_REVIEW.json');rv={r['source_alias']:r for r in review['rows']}
 assert review['model_assisted']and review['independent_human_labels']==0 and len(rv)==32
 panels={r['source_alias']:r for r in read(REVIEW/'PIXEL_PANEL_MANIFEST.json')['panels']}
 for alias,r in rv.items():assert sha(ROOT/panels[alias]['path'])==panels[alias]['sha256']==r['actual_panel_sha256']
 assert set(plan['uncertain_role_queries'])=={a for a,r in rv.items()if r['semantic_role_unknown']}
 assert len(plan['uncertain_role_queries'])==6
 x=read(P/'RESULTS.json');rows=x['rows'];old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/cached32/RESULTS.json')['rows']}
 bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and not bar['new_dense_GT_read']
 assert (D/'GLOBAL_PREDICTION_BARRIER.json').stat().st_mtime_ns<=(D/'CPU_COMPLETION.json').stat().st_mtime_ns
 synthetic=0
 for n in range(5):
  game={i:float(sum((j+1)for j in range(n)if i&(1<<j))+.7*(i.bit_count()>=2))for i in range(1<<n)}
  perm=permutation_shapley(game,n)
  for j,v in enumerate(perm):
   formula=sum(math.factorial(m.bit_count())*math.factorial(n-m.bit_count()-1)/math.factorial(n)*(game[m|(1<<j)]-game[m])for m in game if not(m&(1<<j)))
   same(v,formula);synthetic+=1
 matched=games=outputs=0;max_output=max_partition=0.
 for r in rows:
  alias=r['alias'];z=load(D/'predictions'/(alias+'.pt'));prior=load(OLD/'dino_full'/(alias+'.pt'));oracle=load(OLD/'oracle'/(alias+'.pt'))
  pos=oracle['references']['GT_common']['positions'];assert pos==oracle['references']['DINO_common']['positions']==z['common_positions']
  gt=oracle['references']['GT_common']['targets'];dn=oracle['references']['DINO_common']['targets']
  original=[rv[alias]['slots'][prior['planned_positions'].index(p)]for p in pos]
  cats=['UN'if rv[alias]['semantic_role_unknown']else c for c in original]
  assert cats==z['common_categories']
  assert z['full_categories']==['UN'if rv[alias]['semantic_role_unknown']else rv[alias]['slots'][prior['planned_positions'].index(p)]for p in prior['admitted_positions']]
  present=[g for g in GROUPS if g in cats];assert present==z['groups_present']
  for j,rec in enumerate(z['anchor_geometry']):
   assert rec['original_code']==original[j]and rec['category']==cats[j]
   err=chart(dn[[j]])[0]-chart(gt[[j]])[0]
   same(rec['log_width_ratio'],err[2]);same(rec['log_height_ratio'],err[3]);same(rec['area_ratio'],dn[j,2:].prod()/gt[j,2:].prod());same(rec['center_error_in_target_sizes'],np.linalg.norm(err[:2]/gt[j,2:]))
   aa=np.r_[dn[j,:2]-dn[j,2:]/2,dn[j,:2]+dn[j,2:]/2];bb=np.r_[gt[j,:2]-gt[j,2:]/2,gt[j,:2]+gt[j,2:]/2]
   inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();union=dn[j,2:].prod()+gt[j,2:].prod()-inter
   same(rec['reference_IoU'],inter/union if union>0 else 0.)
  for g in GROUPS:
   same(cats.count(g),r['common_reference_counts'][g]);same(z['full_categories'].count(g),r['full_reference_counts'][g])
  for base,b in prior['bases'].items():
   assert np.array_equal(z['outputs'][base]['Base'],b)
   _,dino_delta=correction(b,pos,dn);_,gt_delta=correction(b,pos,gt)
   category_delta={g:np.sum((chart(dn)-chart(gt))[np.asarray(cats)==g],axis=0)/len(pos)if pos else np.zeros(4)for g in GROUPS}
   for g in GROUPS:assert np.max(abs(category_delta[g]-z['delta_diagnostics'][base]['category_delta_error'][g]))<1e-12
   assert np.max(abs(sum(category_delta.values())-(dino_delta-gt_delta)))<1e-12
   assert np.max(abs(z['delta_diagnostics'][base]['DINO_minus_GT_delta']-(dino_delta-gt_delta)))<1e-12
   for mask in range(1<<len(present)):
    target=dn.copy()
    for j,c in enumerate(cats):
     if mask&(1<<present.index(c)):target[j]=gt[j]
    expected,_=correction(b,pos,target);err=float(abs(expected-z['outputs'][base]['groups_'+str(mask)]).max());assert err<2e-12;max_output=max(max_output,err);outputs+=1
   for mask in range(4):
    target=dn.copy()
    if mask&1:target[:,:2]=gt[:,:2]
    if mask&2:target[:,2:]=gt[:,2:]
    expected,_=correction(b,pos,target);err=float(abs(expected-z['outputs'][base]['geometry_'+str(mask)]).max());assert err<2e-12;max_output=max(max_output,err);outputs+=1
   for ref,pp,target in [('DINO_full',prior['reference_positions'],prior['reference_targets']),('DINO_common',pos,dn),('GT_common',pos,gt)]:
    transformed,delta=correction(b,pp,target)
    for scope in ['global','planned_hull','admitted_nodes']:
     mask=np.ones(len(b),bool)if scope=='global'else np.isin(np.arange(len(b)),range(min(prior['planned_positions']),max(prior['planned_positions'])+1))if scope=='planned_hull'else np.isin(np.arange(len(b)),prior['admitted_positions'])
     expected=b.copy();expected[mask]=transformed[mask];actual=z['outputs'][base]['scope_'+ref+'_'+scope]
     err=float(abs(expected-actual).max());assert err<2e-12;max_output=max(max_output,err);outputs+=1
     if not pp:assert np.array_equal(actual,b)
   for ep in ['v','s','unobserved','unobserved_inside_MBR']:
    if endpoint(r,base,'Base',ep)is None:
     assert r['category_repair_shapley_pp'][base][ep]is None and r['center_size_repair_shapley_pp'][base][ep]is None;continue
    game={m:100*endpoint(r,base,'groups_'+str(m),ep)for m in range(1<<len(present))}
    sp=dict(zip(present,permutation_shapley(game,len(present))))
    for g in GROUPS:same(sp.get(g,0.),r['category_repair_shapley_pp'][base][ep][g])
    geom=permutation_shapley({m:100*endpoint(r,base,'geometry_'+str(m),ep)for m in range(4)},2)
    for key,v in zip(['center','size'],geom):same(v,r['center_size_repair_shapley_pp'][base][ep][key])
    same(sum(sp.values()),sum(geom));same(sum(geom),100*(endpoint(r,base,'geometry_3',ep)-endpoint(r,base,'geometry_0',ep)));games+=2
   for name,ref in [('geometry_0','DINO_common'),('geometry_3','GT_common'),('scope_DINO_full_global','DINO_full')]:
    v=old[alias]['values'][base][ref]['GlobalError']
    for ep in ['v','t','s']:same(r['values'][base][name][ep],v['clocks']['final_MBR'][ep]);matched+=1
    same(endpoint(r,base,name,'unobserved'),v['supports']['unobserved_decoder']['s']);matched+=1
   for name,value in r['values'][base].items():
    for ep in ['v','t','s']:assert np.isfinite(value[ep])
    inside=value['supports']['unobserved_inside_hull'];outside=value['supports']['unobserved_outside_hull'];allv=value['supports']['unobserved'];same(inside['frames']+outside['frames'],allv['frames'])
    weighted=sum(v['s']*v['frames']for v in [inside,outside]if v['frames'])/allv['frames']if allv['frames']else None;same(weighted,allv['s'])
   for ref in ['DINO_full','DINO_common','GT_common']:
    for scope,parts in r['scope_signed_unobserved_contributions_pp'][base][ref].items():
     gain=100*(endpoint(r,base,'scope_'+ref+'_'+scope,'unobserved')-endpoint(r,base,'Base','unobserved'))
     err=abs(sum(parts.values())-gain);max_partition=max(max_partition,err);assert err<1e-10
     total=r['values'][base]['Base']['supports']['unobserved']['frames']
     for side,ep in [('inside_hull','unobserved_inside_hull'),('outside_hull','unobserved_outside_hull')]:
      support=r['values'][base]['Base']['supports'][ep];n=support['frames']
      expected=0. if not n else 100*(endpoint(r,base,'scope_'+ref+'_'+scope,ep)-support['s'])*n/total
      same(parts[side],expected)
  del z,prior,oracle
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==16
  for base,g in x['groups'][ds].items():
   for ep,categories in g['category_oracle_repair_shapley'].items():
    for category,c in categories.items():
     paired([r['category_repair_shapley_pp'][base][ep][category]if r['category_repair_shapley_pp'][base][ep]is not None else None for r in rr],c['cohort_contribution'])
     same(sum(r['common_reference_counts'][category]for r in rr),c['reference_count']);same(sum(r['common_reference_counts'][category]>0 for r in rr),c['present_parents'])
   for ep,parts in g['center_size_oracle_repair_shapley'].items():
    for part,saved in parts.items():paired([r['center_size_repair_shapley_pp'][base][ep][part]if r['center_size_repair_shapley_pp'][base][ep]is not None else None for r in rr],saved)
   for ref,scopes in g['scope'].items():
    for scope,saved in scopes.items():
     for ep,pair in saved.items():
      if ep=='signed_unobserved_parts':
       for side,pair2 in pair.items():paired([r['scope_signed_unobserved_contributions_pp'][base][ref][scope][side]for r in rr],pair2)
      else:paired([None if endpoint(r,base,'Base',ep)is None else 100*(endpoint(r,base,'scope_'+ref+'_'+scope,ep)-endpoint(r,base,'Base',ep))for r in rr],pair)
   for category,metrics in g['reference_geometry'].items():
    for metric,value in metrics.items():
     data=[a[metric]for r in rr for a in r['reference_geometry_by_category'][category]];same(np.mean(data)if data else None,value)
 result=dict(status='pass',count=32,old_query_parent_rosters_exact=True,old_review_pixel_panel_SHA_only_bound=True,private_media_not_decoded_or_viewed=True,semantic_role_unknown_queries=6,all_unknown_roles_overridden_UN_original_codes_retained=True,independent_candidate_output_checks=outputs,max_candidate_error=max_output,permutation_Shapley_games=games,synthetic_permutation_formula_checks=synthetic,old_matched_metric_checks=matched,scalar_aggregate_CI_quality_count_partition_checks=checks,max_inside_outside_sum_error=max_partition,all_cohort_denominators_preserved=True,sparse_cached_reference_targets_only=True,dense_GT_files_read=False,new_labels_models_experts_training_updates=0,overlapping_estimands_not_added=True,result_sha256=sha(P/'RESULTS.json'),script_sha256=sha(__file__))
 target=P/'ROOT_INDEPENDENT_AUDIT.json';assert not target.exists();target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
