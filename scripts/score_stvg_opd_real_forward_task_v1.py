"""Post-global GT only; exhaustive complete-task and real-frame protection/correction outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,math,time
from scripts.stvg_opd_real_forward_task_common_v1 import *
from scripts.score_stvg_opd_LN_state_intervention_v1 import rows_write,rows_read,estimate,arm_stats
FIELDS=['dense_v_delta','dense_s_delta','own_U_delta','observed_delta','correct_U_delta','wrong_U_delta','correct_U_retained','wrong_U_corrected','expert_delta']

def require_seal():
 verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['actual_complete_GPU_fits']==256 and b['actual_complete_rounds']==6400 and not b['GT_read']
 assert len(b['files'])==256 and b['runtime_sha256']==sha(BASE/'RUNTIME.json')
 for n,h in b['files'].items():assert sha(ROOT/n)==h
 return b

def summary(rr,est=estimate):
 s=dict(parents=64,complete_fits=256,control_replays=64,candidate_fits=192,bootstrap=BOOT,seed=SEED,historically_exposed=True,datasets={})
 for ds in ['hc2','vidstg']:
  rows=[r for r in rr if r['dataset']==ds];assert len(rows)==32
  stats={}
  for field in FIELDS:
   arms={a:dict(estimate=est([r[a+'_'+field] for r in rows]),unknown=sum(r[a+'_'+field] is None for r in rows)) for a in ARMS}
   contrasts={}
   for a,b in [('A','SEL'),('HALF','SEL'),('LN','SEL'),('A','HALF'),('A','LN')]:
    vals=[r[b+'_'+field]-r[a+'_'+field] if r[b+'_'+field] is not None and r[a+'_'+field] is not None else None for r in rows];valid=[v for v in vals if v is not None]
    contrasts[b+'_minus_'+a]=dict(estimate=est(vals),unknown=32-len(valid),positive=sum(v>EPS for v in valid),negative=sum(v< -EPS for v in valid),neutral=sum(abs(v)<=EPS for v in valid),harm5pp=sum(v<-.05 for v in valid),harm20pp=sum(v<-.2 for v in valid))
   stats[field]=dict(arms=arms,contrasts=contrasts)
  stats['compute']={a:dict(instrumented_fit_seconds_mean=math.fsum(r[a+'_fit_seconds'] for r in rows)/32,actual_backwards=sum(r[a+'_backwards'] for r in rows),empty_fits=sum(r[a+'_empty'] for r in rows)) for a in ARMS}
  stats['selection_mix']=dict(collections.Counter({k:sum(r['SEL_selection_mix'].get(k,0) for r in rows) for k in ['A','LN','Query','Half']}))
  s['datasets'][ds]=stats
 return s

def frame_labels(z,r,raw,truth,span,independent):
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from vg_tta.tastvg_oracle_event5_v1 import box_iou
 f=z['fit'];before=f['before'].numpy().astype(float);after=f['final'].numpy().astype(float);ids=z['frame_ids'];W=raw['input']['width'];H=raw['input']['height']
 if independent:b=xyxy(before,W,H);a=xyxy(after,W,H)
 else:
  wh=np.array([W,H]);b=np.concatenate(((before[:,:2]-before[:,2:]/2)*wh,(before[:,:2]+before[:,2:]/2)*wh),1);a=np.concatenate(((after[:,:2]-after[:,2:]/2)*wh,(after[:,:2]+after[:,2:]/2)*wh),1)
 if z['dataset']=='hc2':b=np.maximum(b,0);a=np.maximum(a,0)
 def overlap(x,y):
  if independent:return float(box_iou(x,y))
  y=np.asarray(y,float);area=float(np.maximum(0,np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2])).prod());union=float(np.maximum(0,x[2:]-x[:2]).prod()+np.maximum(0,y[2:]-y[:2]).prod()-area);return area/union if union>0 else 0.
 frames=[]
 for k,fr in enumerate(ids):
  if fr not in truth or not span[0]<=fr<span[1]:continue
  bv=overlap(b[k],truth[fr]);av=overlap(a[k],truth[fr]);unvisited=k not in set(r['visited'])
  frames.append(dict(parent_id=r['index'],dataset=r['dataset'],arm=z['arm'],sample_ordinal=k,before=bv,after=av,delta=av-bv,
   visited=not unvisited,initially_correct=bv>=.5,retained=av>=.5,unvisited=unvisited))
 out={}
 for name,pred in [('observed',lambda x: x['visited']),('own_U',lambda x:x['unvisited']),('correct_U',lambda x:x['unvisited'] and x['initially_correct']),('wrong_U',lambda x:x['unvisited'] and not x['initially_correct'])]:
  ff=[x for x in frames if pred(x)];out[name+'_frames']=len(ff)
  for fld in ['before','after','delta']:out[name+'_'+fld]=math.fsum(x[fld] for x in ff)/len(ff) if ff else None
  if name=='correct_U':out[name+'_retained']=sum(x['retained'] for x in ff)/len(ff) if ff else None
  if name=='wrong_U':out[name+'_corrected']=sum(x['retained'] for x in ff)/len(ff) if ff else None
 return out,frames

def assemble(independent=False):
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();bar=require_seal()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.tastvg_oracle_event5_v1 import official
 from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
 from scripts.audit_stvg_opd_forward_candidate_v1 import iou
 from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
 records=read(BASE/'DESIGN_LOCK.json')['records'];targets={ds:truths(ds,dict(parents=[r['query_ordinal'] for r in records if r['dataset']==ds])) for ds in ['hc2','vidstg']}
 rr=[];ff=[];rounds=[];provenance={}
 for r in records:
  tt,sp,binding=targets[r['dataset']];provenance.update(binding);raw=read_row(r['dataset'],r['query_ordinal']);inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert'])
  row=dict(parent_id=r['index'],dataset=r['dataset'],historically_exposed=True)
  for arm in ARMS:
   p=BASE/'formal'/f"{r['index']:05}"/(arm+'_repeat0.pt');assert sha(p)==bar['files'][str(p.relative_to(ROOT))];z=load(p);f=z['fit'];v,frames=frame_labels(z,r,raw,tt[r['query_ordinal']],sp[r['query_ordinal']],independent);ff+=frames
   for k,x in v.items():row[arm+'_'+k]=x
   for name,boxes in [('before',f['before']),('after',f['final'])]:
    metric=(dense_metric(boxes.numpy(),raw,tt[r['query_ordinal']],sp[r['query_ordinal']],z['Native_interval'],r['dataset'])[0] if independent else official(boxes.numpy(),raw,tt[r['query_ordinal']],sp[r['query_ordinal']],z['Native_interval'],r['dataset']))
    for m in ['s','v','t']:row[arm+'_dense_'+m+'_'+name]=metric[m]
   for m in ['s','v','t']:row[arm+'_dense_'+m+'_delta']=row[arm+'_dense_'+m+'_after']-row[arm+'_dense_'+m+'_before']
   assert row[arm+'_dense_t_delta']==0
   en=Energy(ex,f['before'],'top1');pos=[int(x[0]) for x in en.frames];assert pos==f['positions'];target=torch.cat([x[1] for x in en.frames],0).numpy() if pos else np.empty((0,4))
   row[arm+'_expert_delta']=float((iou(f['final'].numpy()[pos],target)-iou(f['before'].numpy()[pos],target)).mean()) if pos else None
   row[arm+'_fit_seconds']=z['compute']['fit_seconds'];row[arm+'_backwards']=f['gradient_calls'];row[arm+'_empty']=f['empty'];mix=collections.Counter()
   for k,h in enumerate(f['path'][:-1]):
    pol=f['geometry']['rounds'][k].get('policy') if f['positions'] else None
    if pol:mix[pol['selected']]+=1
    rounds.append(dict(parent_id=r['index'],dataset=r['dataset'],arm=arm,round=k,updated=f['rounds'][k]['updated'],empty_evidence=f['empty'],
     selected=pol['selected'] if pol else 'A' if arm=='A' else None,reason=pol['reason'] if pol else 'original_A' if arm=='A' and h.get('update') else 'no_information',
     scores=pol['scores'] if pol else {},candidate_seconds=pol['candidate_seconds'] if pol else 0.,
     observed_expert_agreement=float(iou(h['boxes'].numpy()[pos],target).mean()) if pos else None,
     endpoint_drift=float(np.square(h['boxes'].numpy().astype(float)-f['before'].numpy().astype(float)).sum()/len(z['frame_ids']))))
   if arm=='SEL':row['SEL_selection_mix']=dict(mix)
   assert sha(p)==bar['files'][str(p.relative_to(ROOT))]
  assert len({row[a+'_dense_v_before'] for a in ARMS})==1
  rr.append(row);status(BASE/'ROOT_SCORING_STATUS.json' if independent else BASE/'CPU_STATUS.json',dict(status='independent_postseal_GT_readback' if independent else 'actual_postseal_GT_scoring',done=len(rr),total=64,CPU_only=True,GT_read=True,time=time.time()))
 return rr,ff,rounds,provenance

def score():
 require_seal();write(BASE/'GT_EXPOSURE.json',dict(status='actual_GT_only_after_all256_complete_trajectories_sealed',barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),historical_exposure=True,GT_for_selection=False,CPU_only=True,time=time.time()))
 rr,ff,rd,pr=assemble();rows_write(OUT/'ROWS.jsonl.gz',rr);rows_write(OUT/'FRAME_ROWS.jsonl.gz',ff);rows_write(OUT/'ROUND_ROWS.jsonl.gz',rd);write(OUT/'SUMMARY.json',summary(rr))
 write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',parents=64,fits=256,rounds=len(rd),frame_rows=len(ff),GT_provenance=pr,CPU_only=True,GT_read=True,time=time.time()))
if __name__=='__main__':score()
