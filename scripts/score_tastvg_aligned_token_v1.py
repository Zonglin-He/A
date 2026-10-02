"""CPU fixed-candidate qualification; independent patch-coordinate readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_transfer_aligned_common_v1 import *
def independent(patches,qo,qe,tubes,interval):
 p=np.asarray(patches,dtype=np.float64);p/=np.sqrt(np.sum(p*p,axis=-1,keepdims=True));qs=np.concatenate([qo,qe]).astype(np.float64);qs/=np.sqrt(np.sum(qs*qs,axis=-1,keepdims=True)) if len(qs) else 1;side=int(np.sqrt(p.shape[1]));rows=[]
 for tube in tubes:
  pools=[[],[]]
  for fid,bb in enumerate(tube):
   bb=np.asarray(bb,dtype=np.float64);x0,y0=bb[:2]-bb[2:]/2;x1,y1=bb[:2]+bb[2:]/2
   for y in range(side):
    if min(y1,1.,(y+1)/side)<=max(y0,0.,y/side):continue
    for x in range(side):
     if min(x1,1.,(x+1)/side)>max(x0,0.,x/side):pools[0 if interval[0]<=fid<=interval[1] else 1].append(p[fid,y*side+x])
  maxima=[(np.stack(pool)@qs.T).max(0) if pool and len(qs) else None for pool in pools]
  obj=float(maxima[0][:len(qo)].mean()) if maxima[0] is not None and len(qo) else None;bind=float(np.mean(maxima[0][len(qo):]-maxima[1][len(qo):])) if len(qe) and all(z is not None for z in maxima) else None
  rows.append(dict(object=obj,binding=bind,inside_patches=len(pools[0]),outside_patches=len(pools[1]),in_max=maxima[0].tolist() if maxima[0] is not None else None,out_max=maxima[1].tolist() if maxima[1] is not None else None))
 return rows
def summary(rows):
 from scripts.score_tastvg_best_quick_v1 import source_summary
 result={}
 fields=['S_v','S_gain','S_regret','S_pairwise','T_v','T_gain','T_regret','T_pairwise','T_pairwise_full_fallback','delta_T_S_v','delta_T_S_pairwise','query_cosine','unique_binding_scores','unique_ROI_signatures']
 for group in ['corruption','clean']:
  rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')];z=dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),metrics={})
  for f in fields:
   take=[r for r in rr if r[f] is not None];z['metrics'][f]={**source_summary(take,[f])['metrics'].get(f,{}),'available_cells':len(take),'available_sources':len({r['source_id'] for r in take})}
  matched=[r for r in rr if r['T_available']];z['matched_available_S_T']=source_summary(matched,['S_v','S_gain','S_regret','T_v','T_gain','T_regret','delta_T_S_v'])
  for arm in ['S','T']:
   t=[r for r in matched if r[arm+'_pairwise'] is not None];z['matched_available_S_T'][arm+'_ranking']=source_summary(t,[arm+'_pairwise'])
  z['coverage']=dict(all9_available=sum(r['T_available'] for r in rr),unavailable_cells=sum(not r['T_available'] for r in rr),empty_event_cells=sum(r['event_tokens']==0 for r in rr),empty_object_cells=sum(r['object_tokens']==0 for r in rr),one_unique_score=sum(r['unique_binding_scores']==1 for r in rr),one_unique_ROI=sum(r['unique_ROI_signatures']==1 for r in rr),same_pool_size_bias_not_corrected=True)
  z['complementarity']={k:sum(r['complementarity'][k] for r in rr) for k in ['strict_pairs','S_wrong_T_right','S_right_T_wrong','both_right','both_wrong','either_tie','unavailable_pairs']};result[group]=z
 return result
def run():
 import torch
 torch.set_num_threads(2)
 from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
 from scripts.score_tastvg_reference_selection_v1 import independent_rewards
 from vg_tta.tastvg_reference_selection_v1 import pairwise
 verify('TOKEN');bar=read(BASE/'token/PREDICTION_BARRIER.json');assert bar['cells']==60 and len(bar['files'])==60 and not bar['GT_read']
 for f,h in bar['files'].items():assert sha(BASE/f)==h;receipt((BASE/f).with_suffix('.pt'))
 write(BASE/'token/SCORING_LOCK.json',dict(script_sha256=sha(Path(__file__)),labels={ds:sha(POOL/ds/'GT_LABELS_search.json') for ds in ['hc2','vidstg']},barrier_sha256=sha(BASE/'token/PREDICTION_BARRIER.json'),GT_input_to_model=False,time=time.time()))
 checks=collections.Counter();maxerr=0.;allrows={}
 for ds in ['hc2','vidstg']:
  write(BASE/'token'/ds/'GT_EXPOSURE.json',dict(after_60_score_seal=True,historically_exposed=True,GT_not_used_for_phrases_or_tokens=True,time=time.time()));gt=read(POOL/ds/'GT_LABELS_search.json');plan=read(QUAL/ds/'PLAN.json');p=read(OLD/ds/'PLAN.json');prev={r['cell']:r for r in read(ROOT/'results/tastvg_routed_online_token/2026-10-02'/ds/'TOKEN_ROWS.json')};rows=[]
  for c in plan['cells']:
   x=load(BASE/'token'/ds/'scores'/f'{c["cell"]:03}.pt');old=load(ROOT/c['payload']);assert x['pixel_sha256']==c['pixel_sha256'] and x['pre_state_sha256']==c['pre_state_sha256'] and x['interval']==old['slow']['indices'];assert not x['GT_read'] and not x['parameter_updates'] and x['new_experts']==0 and x['vision_projection_bitwise'];checks['prediction_input_bindings']+=1
   tubes=np.stack([a['prediction']['boxes'].numpy() for a in old['update_steps'][0]['candidates']]);ind=independent(x['patches'],x['q_obj'],x['q_evt'],tubes,x['interval']);s=x['scores']
   for k,rr in enumerate(ind):
    av=s['availability'][k];assert rr['inside_patches']==av['inside_patches'] and rr['outside_patches']==av['outside_patches'];checks['independent_ROI_pools']+=2
    for field,val in [('object_scores',rr['object']),('binding_scores',rr['binding'])]:
     b0=s[field][k]
     if val is None:assert b0 is None
     else:err=abs(val-b0);assert err<1e-12;maxerr=max(maxerr,err)
     checks['independent_token_score']+=1
    for name,vals in [('token_in_max',rr['in_max']),('token_out_max',rr['out_max'])]:
     if vals is None:assert av[name] is None
     else:err=float(np.max(np.abs(np.asarray(vals)-av[name])));assert err<1e-12;maxerr=max(maxerr,err)
     checks['independent_token_maxima']+=1
   row=p['rows'][c['parent']];g=gt[str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()};utility=[evaluator(a,row,truth,g['span'],ds=='hc2')(x['interval'])['v'] for a in tubes];np.testing.assert_allclose(utility,prev[c['cell']]['utilities'],atol=1e-12,rtol=0);checks['previous_fixed_utility_parity']+=9
   er=read(QUAL/ds/'receipts'/f'{c["cell"]:03}_student_routed.json');ev=load(QUAL/er['cache']);assert sha(QUAL/er['cache'])==er['cache_sha256'];S=independent_rewards(tubes,ev);assert S==prev[c['cell']]['S_rewards'];T=s['binding_scores'];available=all(z is not None for z in T);pairsS=pairwise(S,utility);pairsT=pairwise(T if available else None,utility);selectedS=0 if S is None else int(np.argmax(S));selectedT=int(np.argmax(T)) if available else 0
   qo=x['q_obj'].astype(float);qe=x['q_evt'].astype(float);cos=None
   if len(qo) and len(qe):a=qo.mean(0);b0=qe.mean(0);cos=float(a@b0/(np.linalg.norm(a)*np.linalg.norm(b0)))
   r={k:c[k] for k in ['cell','source_id','condition','order','arrival']};r.update(utilities=utility,S_rewards=S,T_rewards=T,T_available=available,object_scores=s['object_scores'],object_tokens=len(qo),event_tokens=len(qe),phrase_counts=x['phrase_counts'],referent_rule=x['referent_rule'],query_cosine=cos,availability=s['availability'],unique_ROI_signatures=s['unique_ROI_signatures'],unique_binding_scores=s['unique_binding_scores'])
   for arm,pr,sel in [('S',pairsS,selectedS),('T',pairsT,selectedT)]:
    r.update({arm+'_'+k:v for k,v in dict(selected=sel,v=utility[sel],gain=utility[sel]-utility[0],regret=max(utility)-utility[sel],pairwise=pr['pairwise_accuracy'] if arm=='S' or available else None,decisive=pr['decisive_coverage'] if arm=='S' or available else None,pairs=pr['pairs']).items()})
   r.update(T_pairwise_full_fallback=pairsT['pairwise_accuracy'],delta_T_S_v=r['T_v']-r['S_v'],delta_T_S_pairwise=r['T_pairwise']-r['S_pairwise'] if available and r['T_pairwise'] is not None and r['S_pairwise'] is not None else None)
   comp=dict(strict_pairs=0,S_wrong_T_right=0,S_right_T_wrong=0,both_right=0,both_wrong=0,either_tie=0,unavailable_pairs=0)
   for a,b0 in zip(pairsS['pairs'],pairsT['pairs']):
    g0=a['GT_sign']
    if not g0:continue
    if not available:comp['unavailable_pairs']+=1;continue
    comp['strict_pairs']+=1;sa=a['expert_sign'];ta=b0['expert_sign'];key='either_tie' if sa==0 or ta==0 else 'both_right' if sa==ta==g0 else 'both_wrong' if sa!=g0 and ta!=g0 else 'S_wrong_T_right' if ta==g0 else 'S_right_T_wrong';comp[key]+=1
   r['complementarity']=comp;rows.append(r);print('TOKEN CPU AUDIT',ds,c['cell'],flush=True)
  allrows[ds]=rows;write(PUBLIC/ds/'TOKEN_ROWS.json',rows);sm=summary(rows);write(PUBLIC/ds/'TOKEN_SUMMARY.json',sm);write(PUBLIC/ds/'TOKEN_CASES.json',dict(positive=sorted([r for r in rows if r['condition']!='clean' and r['T_available']],key=lambda r:-r['delta_T_S_v'])[:5],negative=sorted([r for r in rows if r['condition']!='clean' and r['T_available']],key=lambda r:r['delta_T_S_v'])[:5],unavailable=[{k:r[k] for k in ['cell','source_id','condition','referent_rule','phrase_counts','event_tokens']} for r in rows if not r['T_available']]))
 audit=dict(status='pass',checks=dict(checks),max_error=maxerr,GT_after_token_barrier=True,GPU_initialized=torch.cuda.is_initialized(),time=time.time());assert not torch.cuda.is_initialized();write(BASE/'token/ROOT_READBACK.json',audit);write(PUBLIC/'TOKEN_ROOT_READBACK.json',audit);write(PUBLIC/'TOKEN_RESOURCES.json',read(BASE/'token/RESOURCES.json'));write(PUBLIC/'TOKEN_INTERFACE.json',dict(checkpoint=read(BASE/'TOKEN_RUNTIME_LOCK.json')['checkpoint'],environment=read(BASE/'TOKEN_RUNTIME_LOCK.json')['environment'],parser_summary=read(BASE/'TOKEN_RUNTIME_LOCK.json')['parser_summary'],representation='CLIP pretrained common projection; local token alignment empirically tested, not FILIP trained',fusion='not_run'))
 status(BASE/'token/STATUS.json',dict(status='scored_pending_publication',done=60,total=60,time=time.time()));print('TOKEN root readback pass',audit)
if __name__=='__main__':run()
