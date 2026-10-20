"""Independent selected-five-row metrics, bootstrap and selection audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,P,read,put,verify

def main():
 p=verify();r=read(P/'SELECTED_FIVE_ROW_RESULTS_PUBLIC.json');selection=read(P/'SELECTION_PUBLIC.json');errors=[];checks=0
 for ds,g in r['directions'].items():
  from scripts.finalize_stvg_single_referdino_parameter_search256_v4 import result_sources
  candidates=[]
  for ns,ph,dp,pp,z in result_sources(ds):
   for arm,c0 in z['definitions'].items():
    c=dict(c0);c.setdefault('initialization_gamma',1.);tail=z['paired'][arm+'_minus_'+('E_Strong'if ns=='main'else'E_G'+str(int(c['initialization_gamma']*100)).zfill(3))]['tail']['below_minus5'];ref='E_Strong'if ns=='main'else'E_G'+str(int(c['initialization_gamma']*100)).zfill(3);d=np.array([100*(r['values'][arm]['v']-r['values'][ref]['v'])for r in z['rows']]);candidates.append((z['means'][arm]['v'],ns,ph,arm,c,tail,int((d>=.1).sum()),int((d<=-.1).sum())))
  candidates.sort(key=lambda x:(-x[0],abs(x[4]['alpha']-1),x[4]['steps'],x[4]['lr'],-x[4]['tau'],abs(x[4]['initialization_gamma']-1),x[1],x[2]));u=candidates[0];cap=selection['datasets'][ds]['severe_loss_limit'];baseline=read(P/ds/'coarse/RESULTS_PUBLIC.json');ba=p['baseline']+f"_S{p['baseline_steps'][ds]:02}_A100";assert cap==baseline['paired'][ba+'_minus_E_Strong']['tail']['below_minus5'];eligible=[x for x in candidates if x[5]<=cap];top=eligible[0];near=[x for x in eligible if x[0]>=top[0]-.001-1e-14];near.sort(key=lambda x:(-x[6],x[7],-x[0],abs(x[4]['alpha']-1),x[4]['steps'],x[4]['lr'],-x[4]['tau'],x[1],x[2]));chosen=near[0];assert chosen[1:4]==(selection['datasets'][ds]['namespace'],selection['datasets'][ds]['phase'],selection['datasets'][ds]['arm']);assert chosen[5]==selection['datasets'][ds]['selected_severe_loss_count'];assert chosen[6]==selection['datasets'][ds]['meaningful_positive_parents']and chosen[7]==selection['datasets'][ds]['meaningful_negative_parents'];assert abs(100*(top[0]-chosen[0])-selection['datasets'][ds]['mean_concession_pp'])<1e-10;assert top[0]==selection['datasets'][ds]['protected_mean_winner']['v'];w=selection['datasets'][ds]['unconstrained_mean_winner'];assert (w['namespace'],w['phase'],w['arm'],w['v'],w['severe_loss_count'])==(u[1],u[2],u[3],u[0],u[5])
  for row in g['rows']:
   z=read(D/f"selected_private_receipts/{ds}/{row['alias']}.json");u,v=z['GT_span'];observed=set(z['observed_GT_ids'])
   for arm,m in row['values'].items():
    q=z['per_arm'][arm];a,b=q['interval'];ids=np.array(q['dense_GT_ids']);iou=np.array(q['dense_GT_IoU']);mask=(ids>=max(a,u))&(ids<min(b,v));overlap=max(0,min(b,v)-max(a,u));union=b-a+v-u-overlap;actual=q['actual_IoU'];expected=dict(v=float(iou[mask].sum()/max(max(b,v)-min(a,u),1)),t=overlap/union,s=float(iou.mean()),fixed_actual_sIoU=float(np.mean(list(actual.values()))),unobserved_actual_sIoU=float(np.mean([x for f,x in actual.items()if int(f)not in observed])),boundary_MAE=(abs(a-u)+abs(b-v))/2)
    if observed:expected['observed_actual_sIoU']=float(np.mean([x for f,x in actual.items()if int(f)in observed]))
    for k,x in expected.items():errors.append(abs(x-m[k]));checks+=1
  for arm,m in g['means'].items():
   for k,x in m.items():values=[row['values'][arm][k]for row in g['rows']if row['values'][arm][k]is not None];errors.append(abs(sum(values)/len(values)-x));checks+=1
  rng=np.random.default_rng(20261020);idx=rng.integers(0,128,(10000,128))
  for name,m in g['paired'].items():
   a,b=name.split('_minus_');get=lambda row:row['historical'][b]if b in ['original_Full','original_ERef_Strong']else row['values'][b];x=np.array([100*(row['values'][a]['v']-get(row)['v'])for row in g['rows']]);errors.extend(abs(np.quantile(x[idx].mean(1),[.025,.975])-m['v']['CI95']).tolist());checks+=1
   for k,z in m.items():
    if k=='tail':continue
    errors.append(abs(np.mean([100*(row['values'][a][k]-get(row)[k])for row in g['rows']])-z['mean']));checks+=1
   t=m['tail'];assert [int((x>1e-10).sum()),int((x< -1e-10).sum()),int((abs(x)<=1e-10).sum()),int((x< -5).sum())]==[t[k]for k in ['positive','negative','unchanged','below_minus5']]
 assert max(errors)<1e-10
 put(P/'SELECTED_ROOT_AUDIT_PUBLIC.json',dict(status='pass',checks=checks,max_abs=max(errors),whole_parent_configuration_ranking_verified=True,severe_loss_bound_and_unconstrained_winner_verified=True,coverage_priority_and_mean_concession_verified=True,all256_retained=True,no_new_inference=True));print('SELECTED_ROOT_AUDIT_PASS',checks,max(errors),flush=True)
if __name__=='__main__':main()
