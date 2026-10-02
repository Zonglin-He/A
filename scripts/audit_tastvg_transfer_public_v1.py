"""Rebuild public derived scores, selections, summary/bootstrap, coverage."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write
from scripts.score_tastvg_single_write_v1 import summarize
from scripts.score_tastvg_aligned_token_v1 import summary
from vg_tta.tastvg_reference_selection_v1 import pairwise
def equal(a,b):
 global checks
 if isinstance(a,dict):assert set(a)==set(b);[equal(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
 elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-11;checks+=1
 else:assert a==b;checks+=1
def run(folder):
 global checks
 checks=0;rows=read(folder/'hc2/TRANSFER_ROWS.json');assert len(rows)==768;assert len({(r['arm'],r['condition'],r['order'],r['donor_arrival'],r['role']) for r in rows})==768
 for r in rows:
  for k in ['v','s']:equal(r['delta_'+k],r['post_'+k]-r['pre_'+k])
  assert r['lag']==r['target_arrival']-r['donor_arrival'] and ((r['role']=='self')==(r['lag']==0));assert r['donor_arrival']%4==0;assert r['target_scheduled']==(r['target_arrival']%4==0)
 equal(summarize(rows),read(folder/'hc2/TRANSFER_SUMMARY.json'))
 target=read(folder/'TARGET_SELECTION.json');sim=np.asarray(target['cosine_matrix']);assert sim.shape==(32,32);np.testing.assert_allclose(sim,sim.T,atol=1e-14,rtol=0)
 for t in target['table']:
  i=t['donor_arrival'];seq=target['orders'][t['order']];cand=list(range(i+1,32));d=seq[i];expected={'self':i,'next':next(j for j in cand if j%4!=0),'near':min(cand,key=lambda j:(-sim[d,seq[j]],j)),'far':min(cand,key=lambda j:(sim[d,seq[j]],j))}
  for role,j in expected.items():assert t['roles'][role]['arrival']==j and t['roles'][role]['source_id']==seq[j];equal(t['roles'][role]['cosine'],float(sim[d,seq[j]]) if role!='self' else 1.)
 for ds in ['hc2','vidstg']:
  rr=read(folder/ds/'TOKEN_ROWS.json');assert len(rr)==30
  for r in rr:
   u=r['utilities'];S=r['S_rewards'];T=r['T_rewards'];valid=all(z is not None for z in T);assert valid==r['T_available']
   for k,v in enumerate(T):
    av=r['availability'][k];n=r['object_tokens'];count=r['event_tokens'];a=av['token_in_max'];b=av['token_out_max']
    if count and a is not None and b is not None:equal(v,float(np.mean(np.asarray(a[n:])-np.asarray(b[n:]))))
    else:assert v is None
    if n and a is not None:equal(r['object_scores'][k],float(np.mean(a[:n])))
    else:assert r['object_scores'][k] is None
   comp=dict(strict_pairs=0,S_wrong_T_right=0,S_right_T_wrong=0,both_right=0,both_wrong=0,either_tie=0,unavailable_pairs=0)
   pp={}
   for arm,reward in [('S',S),('T',T if valid else None)]:
    pp[arm]=pairwise(reward,u);selected=0 if reward is None else int(np.argmax(reward));equal(r[arm+'_selected'],selected);equal(r[arm+'_v'],u[selected]);equal(r[arm+'_gain'],u[selected]-u[0]);equal(r[arm+'_regret'],max(u)-u[selected]);equal(r[arm+'_pairs'],pp[arm]['pairs']);equal(r[arm+'_pairwise'],pp[arm]['pairwise_accuracy'] if arm=='S' or valid else None)
   for s,t in zip(pp['S']['pairs'],pp['T']['pairs']):
    g=s['GT_sign'];sa=s['expert_sign'];ta=t['expert_sign']
    if not g:continue
    if not valid:comp['unavailable_pairs']+=1;continue
    comp['strict_pairs']+=1;key='either_tie' if not sa or not ta else 'both_right' if sa==ta==g else 'both_wrong' if sa!=g and ta!=g else 'S_wrong_T_right' if ta==g else 'S_right_T_wrong';comp[key]+=1
   equal(comp,r['complementarity']);equal(r['delta_T_S_v'],r['T_v']-r['S_v']);equal(r['T_pairwise_full_fallback'],pp['T']['pairwise_accuracy']);equal(r['unique_binding_scores'],len(set(x for x in T if x is not None)))
  equal(summary(rr),read(folder/ds/'TOKEN_SUMMARY.json'))
 out=dict(status='pass',checks=checks,transfer_rows=768,token_cells=60,source_bootstraps=10000,GT_read=False,raw_assets_read=False);print(json.dumps(out));return out
if __name__=='__main__':
 f=Path(sys.argv[1]);o=run(f)
 if len(sys.argv)>2:write(Path(sys.argv[2]),o)
