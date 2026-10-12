"""GT only after all192 controlled fits seal; anonymous exhaustive outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_LN_state_intervention_common_v1 import *

def require_seal():
 verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
 assert b['status']=='sealed' and b['actual_complete_GPU_fits']==192 and b['actual_complete_rounds']==4800 and not b['GT_read']
 assert b['runtime_sha256']==sha(BASE/'RUNTIME.json') and len(b['files'])==192
 for rel,h in b['files'].items():assert sha(ROOT/rel)==h
 return b
def rows_write(path,rr):
 assert not path.exists()
 with path.open('xb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
   for r in rr:f.write((json.dumps(r,separators=(',',':'),allow_nan=False)+'\n').encode())
def rows_read(path):
 with gzip.open(path,'rt') as f:return [json.loads(s) for s in f]
def sign(x):return 'unknown' if x is None else 'positive' if x>EPS else 'negative' if x< -EPS else 'neutral'
def estimate(x):
 v=np.array([x for x in x if x is not None],float);n=len(v)
 if not n:return dict(defined_parents=0,mean=None,ci95=None)
 idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n));ci=np.quantile(v[idx].mean(1),[.025,.975])
 return dict(defined_parents=n,mean=float(v.mean()),ci95=ci.tolist())
def arm_stats(rr,field):
 vals=[r[field] for r in rr];v=[v for v in vals if v is not None]
 return dict(total_parents=len(rr),unknown_parents=len(rr)-len(v),positive_parents=sum(x>EPS for x in v),negative_parents=sum(x< -EPS for x in v),
  neutral_parents=sum(abs(x)<=EPS for x in v),harm5pp=sum(x<-.05 for x in v),harm20pp=sum(x<-.2 for x in v),estimate=estimate(vals))
def contrast_stats(rr,fields):
 fa,fb=fields;a=[r[fa] for r in rr];b=[r[fb] for r in rr];pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
 delta=[y-x for x,y in pairs];flips=[(x,y) for x,y in pairs if x*y<0 and abs(x)>EPS and abs(y)>EPS]
 return dict(total_parents=len(rr),defined_pairs=len(pairs),unknown_pairs=len(rr)-len(pairs),
  transition_matrix=dict(collections.Counter(sign(x)+'_'+sign(y) for x,y in zip(a,b))),signed_second_minus_first=estimate(delta),absolute_difference=estimate([abs(v) for v in delta]),
  sign_flips=len(flips),flip_both_at_least1pp=sum(min(abs(x),abs(y))>=.01 for x,y in flips),flip_both_at_least5pp=sum(min(abs(x),abs(y))>=.05 for x,y in flips),
  differences_at_least5pp=sum(abs(v)>=.05 for v in delta),differences_at_least20pp=sum(abs(v)>=.2 for v in delta))
def summary(rr):
 out=dict(parents=64,fits=192,bootstrap=BOOT,seed=SEED,historically_exposed_panel=True,datasets={})
 for ds in ['hc2','vidstg']:
  records=[r for r in rr if r['dataset']==ds];assert len(records)==32;out['datasets'][ds]={}
  for mask in ['own_U','common_U','observed','dense_s','dense_v']:
   out['datasets'][ds][mask]=dict(arms={a:arm_stats(records,a+'_'+mask+'_delta') for a in ARMS},
    contrasts={a+'_to_'+b:contrast_stats(records,[a+'_'+mask+'_delta',b+'_'+mask+'_delta']) for a,b in [('Source','Inherited_order1'),('Source','Inherited_order2'),('Inherited_order1','Inherited_order2')]})
 return out
def real_frame_labels(z,inp,raw,truth,span,independent=False):
 before=z['fit']['before'].numpy().astype(float);after=z['fit']['final'].numpy().astype(float);ids=z['frame_ids']
 W=raw['input']['width'];H=raw['input']['height'];clip=z['dataset']=='hc2'
 if independent:
  from vg_tta.tastvg_paper48_metrics_v1 import xyxy
  from vg_tta.tastvg_oracle_event5_v1 import box_iou
  b=xyxy(before,W,H);a=xyxy(after,W,H)
  if clip:b=np.maximum(b,0);a=np.maximum(a,0)
  overlaps_b={k:float(box_iou(b[k],truth[ids[k]])) for k,f in enumerate(ids) if f in truth and span[0]<=f<span[1]}
  overlaps_a={k:float(box_iou(a[k],truth[ids[k]])) for k in overlaps_b}
 else:
  wh=np.array([W,H]);b=np.concatenate(((before[:,:2]-before[:,2:]/2)*wh,(before[:,:2]+before[:,2:]/2)*wh),1);a=np.concatenate(((after[:,:2]-after[:,2:]/2)*wh,(after[:,:2]+after[:,2:]/2)*wh),1)
  if clip:b=np.maximum(b,0);a=np.maximum(a,0)
  available=[k for k,f in enumerate(ids) if f in truth and span[0]<=f<span[1]]
  overlaps_b={};overlaps_a={}
  for pred,out in [(b,overlaps_b),(a,overlaps_a)]:
   for k in available:
    target=np.asarray(truth[ids[k]],float);extent=np.maximum(np.minimum(pred[k,2:],target[2:])-np.maximum(pred[k,:2],target[:2]),0);area=float(np.prod(extent))
    union=float(np.prod(np.maximum(pred[k,2:]-pred[k,:2],0))+np.prod(np.maximum(target[2:]-target[:2],0))-area);out[k]=area/union if union>0 else 0.
 available=list(overlaps_b);own=set(map(int,inp['visited'][0]));union=own|set(map(int,inp['visited'][1]));masks=dict(own_U=[k for k in available if k not in own],common_U=[k for k in available if k not in union],observed=[k for k in available if k in own])
 out=dict(GT_event_real_frames=len(available),visited_Omega0_positions=len(own),admitted_Omega0_positions=len(z['fit']['positions']),empty_evidence=z['fit']['empty'],actual_backwards=z['fit']['gradient_calls'])
 for name,mask in masks.items():
  bv=math.fsum(overlaps_b[k] for k in mask)/len(mask) if mask else None;av=math.fsum(overlaps_a[k] for k in mask)/len(mask) if mask else None
  dv=math.fsum(overlaps_a[k]-overlaps_b[k] for k in mask)/len(mask) if mask else None
  out.update({name+'_frames':len(mask),name+'_before':bv,name+'_after':av,name+'_delta':dv})
 return out
def assemble(independent=False):
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();bar=require_seal()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 from vg_tta.tastvg_oracle_event5_v1 import official
 from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
 records=read(BASE/'DESIGN_LOCK.json')['records'];targets={}
 for ds in ['hc2','vidstg']:targets[ds]=truths(ds,dict(parents=[r['query_ordinal'] for r in records if r['dataset']==ds]))
 rr=[];provenance={};counts=collections.Counter()
 for r in records:
  truth,spans,binding=targets[r['dataset']];provenance.update(binding);raw=read_row(r['dataset'],r['query_ordinal']);pair=pair_for(r)
  row=dict(parent_id=r['index'],dataset=r['dataset'],historically_exposed=True)
  for arm in ARMS:
   p=BASE/'formal'/f"{r['index']:05}"/(arm+'_repeat0.pt');assert sha(p)==bar['files'][str(p.relative_to(ROOT))];z=load(p)
   assert z['runtime_sha256']==sha(BASE/'RUNTIME.json') and z['evidence_sha256']==pair['evidence_sha256'][0]
   values=real_frame_labels(z,pair,raw,truth[r['query_ordinal']],spans[r['query_ordinal']],independent)
   for k,v in values.items():row[arm+'_'+k]=v
   for name,boxes in [('before',z['fit']['before']),('after',z['fit']['final'])]:
    if independent:metric=dense_metric(boxes.numpy(),raw,truth[r['query_ordinal']],spans[r['query_ordinal']],z['Native_interval'],r['dataset'])[0]
    else:metric=official(boxes.numpy(),raw,truth[r['query_ordinal']],spans[r['query_ordinal']],z['Native_interval'],r['dataset'])
    for m in ['s','v','t']:row[arm+'_dense_'+m+'_'+name]=metric[m]
   for m in ['s','v','t']:row[arm+'_dense_'+m+'_delta']=row[arm+'_dense_'+m+'_after']-row[arm+'_dense_'+m+'_before']
   assert row[arm+'_dense_t_delta']==0;counts['fits']+=1
   assert sha(p)==bar['files'][str(p.relative_to(ROOT))]
  for mask in ['own_U','common_U','observed']:
   assert len({row[a+'_'+mask+'_frames'] for a in ARMS})==1
  rr.append(row)
  status(BASE/('ROOT_SCORING_STATUS.json' if independent else 'CPU_STATUS.json'),dict(status='independent_actual_GT_scalar_readback' if independent else 'actual_postseal_GT_scoring',done=len(rr),total=64,CPU_only=True,GT_read=True,time=time.time()))
 assert len(rr)==64 and counts['fits']==192 and not torch.cuda.is_initialized();verify()
 return rr,dict(counts),provenance
def score():
 require_seal();write(BASE/'GT_EXPOSURE.json',dict(status='actual_GT_after_all192_new_complete_fits_global_sealed',barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
  historical_GT_exposure=True,GT_for_arm_or_state_or_step_selection=False,CPU_only=True,time=time.time()))
 rr,c,provenance=assemble();rows_write(OUT/'ROWS.jsonl.gz',rr);write(OUT/'SUMMARY.json',summary(rr))
 write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',parents=64,fits=192,counts=c,GT_provenance=provenance,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':score()
