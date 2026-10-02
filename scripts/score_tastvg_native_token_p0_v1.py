"""Post-token-seal CPU qualification and conditional fixed rank-sum readout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_routed_common_v1 import *
import numpy as np

def summarize(rows):
 from scripts.score_tastvg_best_quick_v1 import source_summary
 fields=['T_v','T_gain','T_regret','T_pairwise','T_decisive','S_v','S_gain','S_regret','S_pairwise','S_decisive','delta_T_S_v','delta_T_S_pairwise','query_cosine','text_weights_mean_L1']
 result={}
 for group in ['corruption','clean']:
  rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')];z=dict(cells=len(rr),sources=len({r['source_id'] for r in rr}),metrics={})
  for f in fields:
   take=[r for r in rr if r.get(f) is not None and (not f.startswith('T_') or r['T_available'])]
   if take:z['metrics'][f]={**source_summary(take,[f])['metrics'][f], 'available_cells':len(take),'available_sources':len({r['source_id'] for r in take})}
  z['complementarity']={k:sum(r['complementarity'][k] for r in rr) for k in ['strict_pairs','S_wrong_T_right','S_right_T_wrong','both_right','both_wrong','either_tie']}
  z.update(token_all9_available=sum(r['T_available'] for r in rr),token_unavailable_cells=sum(not r['T_available'] for r in rr),outside_unavailable_candidates=sum(r['outside_unavailable_candidates'] for r in rr))
  result[group]=z
 return result

def independent_query(cache,attention,branch):
 frame=[]
 for off,v in enumerate(cache['views']):
  H=np.asarray(v['H'],float);h,w=v['info']['fea_map_size'];n=h*w;l=H.shape[0]-2*n;a=np.asarray(attention[branch][off],float);mask=np.asarray(v['info']['encoded_mask'],bool)
  qs=[]
  for t in range(H.shape[1]):
   weights=np.array([0. if mask[t,n+j] else a[t,0,n+j if branch=='spatial' else j] for j in range(l)])
   qs.append(np.sum(H[n:n+l,t]*weights[:,None],axis=0)/sum(weights))
  frame.append(qs)
 return np.mean([frame[i%2][i//2] for i in range(len(cache['frame_ids']))],axis=0)

def independent_vectors(cache,tube,interval):
 n=len(tube);inside=[];outside=[];appearance=[];availability=[]
 for i,bb in enumerate(tube):
  bb=np.asarray(bb,dtype=np.float64)
  v=cache['views'][i%2];h,w=v['info']['fea_map_size'];c=h*w;H=np.asarray(v['H'],float);padding=np.asarray(v['info']['encoded_mask'],bool)[i//2,:c]
  x0,y0=np.maximum(bb[:2]-bb[2:]/2,0);x1,y1=np.minimum(bb[:2]+bb[2:]/2,1);areas=[]
  for yy in range(h):
   for xx in range(w):areas.append(0. if padding[yy*w+xx] else max(0,min((xx+1)/w,x1)-max(xx/w,x0))*max(0,min((yy+1)/h,y1)-max(yy/h,y0)))
  if sum(areas)<=0:availability.append(False);continue
  weights=np.array(areas)/sum(areas);appearance_vector=np.sum(H[:c,i//2]*weights[:,None],axis=0);motion=np.sum(H[-c:,i//2]*weights[:,None],axis=0);availability.append(True)
  if interval[0]<=i<=interval[1]:inside.append(motion);appearance.append(appearance_vector)
  else:outside.append(motion)
 return dict(appearance_inside=np.mean(appearance,axis=0) if appearance else None,motion_inside=np.mean(inside,axis=0) if inside else None,motion_outside=np.mean(outside,axis=0) if outside else None)

def run():
 import torch
 torch.set_num_threads(2)
 from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
 from scripts.score_tastvg_reference_selection_v1 import independent_rewards
 from scripts.score_tastvg_best_quick_v1 import source_summary
 from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
 from vg_tta.tastvg_native_token_binding_v1 import cosine
 from vg_tta.tastvg_reference_selection_v1 import pairwise
 lock=verify();bar=read(BASE/'token/PREDICTION_BARRIER.json');assert bar['cells']==60 and not bar['GT_read'] and bar['parameter_updates']==0
 for f,h in bar['files'].items():
  assert sha(BASE/f)==h
  if Path(f).parent.name in DATASETS:assert sha((BASE/f).with_suffix('.pt'))==read(BASE/f)['sha256']
  else:assert f.startswith('token/revisions/'),'Unexpected metadata in token seal'
 pins=dict(read(BASE/'TOKEN_RUNTIME_LOCK.json')['pins'])
 for rev in sorted((BASE/'token/revisions').glob('*.json')):pins.update(read(rev)['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h
 checks=collections.Counter();maxerr=0.;all_rows={};conditional={}
 for ds in DATASETS:
  plan=read(QUAL/ds/'PLAN.json');p=read(PRIOR/ds/'PLAN.json');lab=read(BASE/'SCORING_RUNTIME_LOCK.json')['diagnostic_labels'][ds];assert sha(ROOT/lab['path'])==lab['sha256'];gt=read(ROOT/lab['path']);rows=[]
  write(BASE/'token'/ds/'GT_EXPOSURE.json',dict(after_token_barrier=True,barrier_sha256=sha(BASE/'token/PREDICTION_BARRIER.json'),historically_exposed=True,GT_used_for_online=False,time=time.time()))
  for c in plan['cells']:
   x=load(BASE/'token'/ds/f'{c["cell"]:03}.pt');s=x['scores'];old=load(ROOT/c['payload']);row=p['rows'][c['parent']];cr=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json');cache=load(POOL/ds/cr['cache']);tubes=np.stack([z['prediction']['boxes'].numpy() for z in old['update_steps'][0]['candidates']]);idx=old['slow']['indices'];g=gt[str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()};span=g['span']
   assert x['prediction_bitwise_parity'] and not x['parameter_update'] and not x['GT_read'] and x['pre_state_sha256']==c['pre_state_sha256'];checks['native_prediction_parity']+=1
   for branch,name in [('spatial','q_obj'),('temporal','q_evt')]:
    q=independent_query(cache,x['attention'],branch);err=float(np.max(np.abs(q-s[name])));assert err<1e-12;maxerr=max(maxerr,err);checks['independent_text_query']+=1
   for k,tube in enumerate(tubes):
    vv=independent_vectors(cache,tube,idx)
    for name,a in vv.items():
     b=s['vectors'][k][name]
     if a is None:assert b is None
     else:err=float(np.max(np.abs(a-b)));assert err<1e-12;maxerr=max(maxerr,err)
     checks['independent_ROI_vector']+=1
    obj=cosine(vv['appearance_inside'],s['q_obj']) if vv['appearance_inside'] is not None else None
    bind=cosine(vv['motion_inside'],s['q_evt'])-cosine(vv['motion_outside'],s['q_evt']) if vv['motion_inside'] is not None and vv['motion_outside'] is not None else None
    for a,b in [(obj,s['object_scores'][k]),(bind,s['binding_scores'][k])]:
     if a is None:assert b is None
     else:assert abs(a-b)<1e-12
     checks['independent_cosine']+=1
   utilities=[evaluator(t,row,truth,span,ds=='hc2')(idx)['v'] for t in tubes]
   rr=read(QUAL/ds/'receipts'/f'{c["cell"]:03}_student_routed.json');ev=load(QUAL/rr['cache']);S=independent_rewards(tubes,ev);T=s['binding_scores'];valid=all(z is not None for z in T)
   r={k:c[k] for k in ['cell','source_id','condition','order','arrival']};r.update(utilities=utilities,S_rewards=S,T_rewards=T,object_scores=s['object_scores'],T_available=valid,query_cosine=s['query_cosine'],text_weights_mean_L1=s['text_weights_mean_L1'],outside_unavailable_candidates=sum(z['outside_frames']==0 for z in s['availability']),availability=s['availability'])
   for arm,scores in [('S',S),('T',T if valid else None)]:
    selected=0 if scores is None else int(np.argmax(scores));pairs=pairwise(scores,utilities);u=utilities[selected]
    r.update({arm+'_'+k:v for k,v in dict(selected=selected,v=u,gain=u-utilities[0],regret=max(utilities)-u,pairwise=pairs['pairwise_accuracy'] if arm=='S' or valid else None,decisive=pairs['decisive_coverage'] if arm=='S' or valid else None,pairs=pairs['pairs']).items()})
   r['delta_T_S_v']=r['T_v']-r['S_v'] if valid else None;r['delta_T_S_pairwise']=r['T_pairwise']-r['S_pairwise'] if valid and r['T_pairwise'] is not None and r['S_pairwise'] is not None else None
   comp=dict(strict_pairs=0,S_wrong_T_right=0,S_right_T_wrong=0,both_right=0,both_wrong=0,either_tie=0)
   for a,b in zip(r['S_pairs'],r['T_pairs']):
    if not a['GT_sign'] or not valid:continue
    comp['strict_pairs']+=1
    sa,ta=a['expert_sign'],b['expert_sign'];g=a['GT_sign']
    key='either_tie' if sa==0 or ta==0 else 'both_right' if sa==ta==g else 'both_wrong' if sa!=g and ta!=g else 'S_wrong_T_right' if ta==g else 'S_right_T_wrong';comp[key]+=1
   r['complementarity']=comp
   # Offline object correctness is labelled only on true annotated support.
   labelled_in=[];labelled_out=[];unlabelled_out=0;accurate_in=accurate_out=0
   for pos in ev['positions']:
    if not ev['valid'][pos]:continue
    fid=row['frame_ids'][pos];event=span[0]<=fid<=span[1] if ds=='hc2' else span[0]<=fid<span[1]
    if fid not in truth:
     if not event:unlabelled_out+=1
     continue
    bb=np.asarray(ev['boxes'][pos],float);wh=np.array([row['input']['width'],row['input']['height']]);xy=np.r_[wh*(bb[:2]-bb[2:]/2),wh*(bb[:2]+bb[2:]/2)];gg=np.array(truth[fid],float);inter=np.maximum(np.minimum(xy[2:],gg[2:])-np.maximum(xy[:2],gg[:2]),0).prod();union=np.maximum(xy[2:]-xy[:2],0).prod()+np.maximum(gg[2:]-gg[:2],0).prod()-inter
    if inter/max(union,1e-12)>=.5:
     # This remains a candidate-native frame score; Sa2VA ROI itself is not substituted.
     value=s['frame_event_cosines'][0][pos]
     if value is not None:
      (labelled_in if event else labelled_out).append(value)
      if event:accurate_in+=1
      else:accurate_out+=1
   accuracy=np.mean([float(a>b)+.5*float(a==b) for a in labelled_in for b in labelled_out]) if labelled_in and labelled_out else None
   r['event_object_diagnostic']=dict(labelled_accurate_inside=accurate_in,labelled_accurate_outside=accurate_out,unlabelled_outside_valid_masks=unlabelled_out,pair_accuracy=accuracy,AUROC=accuracy,qualified_same_object_inside_outside=bool(labelled_in and labelled_out))
   rows.append(r)
  z=summarize(rows);all_rows[ds]=rows;write(PUBLIC/ds/'TOKEN_ROWS.json',rows);write(PUBLIC/ds/'TOKEN_SUMMARY.json',z)
  m=z['corruption']['metrics'];passed=bool('T_pairwise' in m and m['T_pairwise']['mean']>.5 and m['T_gain']['mean']>0)
  conditional[ds]=dict(eligible=passed,rule='positive corruption source-macro T pairwise above chance and T top1 gain vs candidate0; development qualification only, no confidence-bound gate',sealed_T_rows_sha256=sha(PUBLIC/ds/'TOKEN_ROWS.json'),time=time.time())
 # Freeze the diagnostic eligibility before looking at any ST task readout.
 write(BASE/'token/ST_ELIGIBILITY.json',conditional)
 for ds,rows in all_rows.items():
  if not conditional[ds]['eligible']:continue
  fused=[]
  for r in rows:
   if not r['T_available']:continue
   sr=average_ranks(np.zeros(9) if r['S_rewards'] is None else r['S_rewards']);tr=average_ranks(r['T_rewards']);ranks=sr+tr;selected=int(np.argmin(ranks));u=r['utilities'];pair=pairwise((-ranks).tolist(),u)
   fused.append({**{k:r[k] for k in ['cell','source_id','condition','order','arrival']},'S_ranks':sr.tolist(),'T_ranks':tr.tolist(),'ST_ranks':ranks.tolist(),'ST_selected':selected,'utilities':u,'ST_v':u[selected],'ST_gain':u[selected]-u[0],'delta_ST_S_v':u[selected]-r['S_v'],'delta_ST_T_v':u[selected]-r['T_v'],'ST_regret':max(u)-u[selected],'ST_pairwise':pair['pairwise_accuracy'],'ST_decisive':pair['decisive_coverage']})
  sm={}
  for group in ['corruption','clean']:
   take=[r for r in fused if (r['condition']!='clean')==(group=='corruption')];fields=['ST_v','ST_gain','delta_ST_S_v','delta_ST_T_v','ST_regret'];sm[group]=source_summary(take,fields)
   pairtake=[r for r in take if r['ST_pairwise'] is not None];sm[group]['ranking']=source_summary(pairtake,['ST_pairwise','ST_decisive'])
  write(PUBLIC/ds/'ST_ROWS.json',fused);write(PUBLIC/ds/'ST_SUMMARY.json',sm)
 write(BASE/'token/ROOT_READBACK.json',dict(status='pass',checks=dict(checks),max_error=maxerr,GT_after_token_seal=True,GPU_initialized=torch.cuda.is_initialized(),time=time.time()));write(PUBLIC/'TOKEN_ROOT_READBACK.json',read(BASE/'token/ROOT_READBACK.json'));write(PUBLIC/'ST_ELIGIBILITY.json',conditional);write(PUBLIC/'TOKEN_RESOURCES.json',read(BASE/'token/RESOURCES.json'))
 assert not torch.cuda.is_initialized();status(BASE/'token/STATUS.json',dict(status='scored_pending_public_audit',done=60,total=60,time=time.time()));print('TOKEN scored and independent ROI readback pass',dict(checks),maxerr)
if __name__=='__main__':run()
