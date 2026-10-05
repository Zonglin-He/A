"""Full GT diagnosis after global seal; no model loading or CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,gzip,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.decota_fixed_full_common_v1 import *
from vg_tta.decota_fixed_full_audit_v1 import loss,top1_support,vector,audit
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
from scripts.run_decota_fixed_full_v1 import unpack_expert,unvector
from methods.decota_final_simplified_v1.tensors import state_hash
from scripts.score_tastvg_best_full_v1 import source_summary

def truth(ds,p):
 if ds=='vidstg':
  from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth as original
  return original('P1',p)
 ap=ROOT/'data/hcstvg2_official_metadata/val_v2.json';a=read(ap)
 assert sha(ap)==read(BASE/'DESIGN_LOCK.json')['hc_annotation_metadata_sha256'];dense={};spans={}
 for i,r in enumerate(p['rows']):
  v=a[r['annotation_key']];assert v['English'].lower()==r['input']['caption']
  start=int(v['st_frame'])-1;end=start+len(v['bbox'])-1;spans[i]=[start,end]
  dense[i]={start+j:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])] for j,(x,y,w,h) in enumerate(v['bbox'])}
 return dense,spans,{str(ap.relative_to(ROOT)):sha(ap)}

def verify_seal():
 verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['cells']==330840 and not b['GT_read']
 for job,ds,_ in JOBS:
  f=BASE/job/'PREDICTION_BARRIER.json';assert sha(f)==b['jobs'][job];z=read(f)
  assert z['cells']==read(BASE/ds/'PLAN.json')['arrivals_per_checkpoint']
  for rel,h in z['files'].items():assert sha(BASE/rel)==h and read((BASE/rel).with_suffix('.json'))['sha256']==h
 return b

def observations(ex,row,gt,native,before,after):
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 w,h=row['input']['width'],row['input']['height'];out=[]
 for (_,pos),o in sorted(ex['observations'].items()):
  p=o['probe'];fid=row['frame_ids'][pos];b=np.asarray(p['boxes'],float).reshape(-1,4);s=np.asarray(p['target_scores'],float)
  valid=(b[:,2:]>0).all(1);b,s=b[valid],s[valid]
  r=dict(position=pos,GT_known=fid in gt,admitted=bool(p['accepted']),reason=p['reason'],proposals=len(b))
  if fid in gt:
   quality=box_iou(xyxy(b,w,h),gt[fid]) if len(b) else np.array([])
   chosen=float(quality[int(s.argmax())]) if len(s) else None
   r.update(best_iou=float(quality.max()) if len(quality) else None,chosen_iou=chosen,
    support_correct=bool((quality>.5).any()),chosen_correct=chosen is not None and chosen>.5,
    rejected_correct=not p['accepted'] and bool((quality>.5).any()),
    missed_correct=bool((quality>.5).any()) and (chosen is None or chosen<=.5),
    native_iou=float(box_iou(xyxy(native[pos],w,h),gt[fid])),
    before_iou=float(box_iou(xyxy(before[pos],w,h),gt[fid])),after_iou=float(box_iou(xyxy(after[pos],w,h),gt[fid])))
  out.append(r)
 return out

def stage_transitions(rows,a,b):
 changes=np.array([r[b]-r[a] for r in rows]);out=dict(cells=len(rows),improved=int((changes>1e-10).sum()),degraded=int((changes< -1e-10).sum()),
  gross_gain_pp=float(np.maximum(changes,0).mean()*100) if len(rows) else None,gross_loss_pp=float(-np.minimum(changes,0).mean()*100) if len(rows) else None)
 for threshold in [.3,.5]:
  before=np.array([r[a]>threshold for r in rows]);after=np.array([r[b]>threshold for r in rows]);n=int(before.sum());bad=int((before&~after).sum())
  out[str(threshold)]=dict(correct_before=n,correct_to_wrong=bad,wrong_to_correct=int((~before&after).sum()),damage_rate=bad/n if n else None)
 return out

def run():
 torch.set_num_threads(2);verify_seal();PUB.mkdir(parents=True,exist_ok=True)
 if (BASE/'CPU_COMPLETION.json').exists():return
 write(BASE/'GT_EXPOSURE.json',dict(time=time.time(),global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
  scope='Full fixed evaluation and user-authorized pipeline diagnosis only; no reselection or model feedback'))
 all_summaries={};diagnoses={};total=0;checks=0;samples=0;tick=time.time();files={}
 for job,ds,source in JOBS:
  p=read(BASE/ds/'PLAN.json');dense,spans,provenance=truth(ds,p);schema=read(BASE/job/'STATE_SCHEMA.json');names=[x['name'] for x in schema]
  source_ids={s:i for i,s in enumerate(sorted({r['source'] for r in p['rows']}))};rows=[];target=PUB/job;target.mkdir(parents=True,exist_ok=True)
  original=torch.load(BASE/'smoke'/job/'00000.pt',map_location='cpu',weights_only=False)['fit']['initial']
  for cond in p['conditions']:
   for order,seq in p['orders'].items():
    output=target/f'{cond}_{order}.jsonl.gz';assert not output.exists(),output
    with gzip.open(output,'wt',encoding='utf-8',compresslevel=6) as stream:
     prev=original;prevsha=None
     for at,parent in enumerate(seq):
      row=p['rows'][parent];f=BASE/job/'online'/cond/order/f'{at:05}.npz';a,x,rc=load_npz(f)
      assert x['previous_payload_sha256']==prevsha and x['parent']==parent and x['condition']==cond and x['order']==order and not x['GT_read']
      initial,selected,committed=[unvector(a[k],schema) for k in ['initial','selected','committed']]
      assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else prev[n]) for n,v in initial.items());checks+=1792
      for st,k in [(initial,'initial'),(selected,'selected'),(committed,'committed')]:assert state_hash(st)==x[k+'_sha256'];checks+=1
      for n,v in initial.items():assert torch.equal(committed[n],torch.zeros_like(v) if n=='spatial.query_residual' else v+(selected[n]-v)/16);checks+=v.numel()
      inp=BASE/job/'inputs'/cond/f'{parent:05}.npz';ia,ix,ir=load_npz(inp);assert sha(inp.with_suffix('.json'))==x['input_receipt_sha256']
      assert ix['interval']==x['interval'] and ix['pixel_sha256']==x['pixel_sha256'] and ix['subject_sha256']==x['subject_sha256']
      ex=unpack_expert(ix['expert']);support=top1_support(ex);path=a['box_path'];choose=min(range(len(x['losses'])),key=lambda j:x['losses'][j])
      assert x['selected_step']==choose and len(path)==len(x['losses'])==(1 if not support else 11)
      assert x['empty']==(not support) and x['gradient_calls']==len(path)-1
      for b,l in zip(path,x['losses']):assert abs(loss(b,support)-l)<8e-6;checks+=1
      if 'raw_math_sample' in x:
       raw=BASE/x['raw_math_sample']['path'];assert sha(raw)==x['raw_math_sample']['sha256'];z=torch.load(raw,map_location='cpu',weights_only=False)
       assert audit(z['fit'],z['expert'])==x['math_audit'];assert np.array_equal(np.stack([h['boxes'].numpy() for h in z['fit']['path']]),path);samples+=1
      assert x['math_audit']['status']=='pass' and not x['math_audit']['GT_read']
      truth_row,span=dense[parent],spans[parent];iv=x['interval']
      make=lambda b:DenseTube(b,row,truth_row,span,clip=ds=='hc2')
      fs=make(ia['native_boxes']);bf=make(path[0]);af=make(path[choose]);values={k:d.score(iv) for k,d in [('Frozen',fs),('Before',bf),('After',af)]}
      for k,b in [('Frozen',ia['native_boxes']),('Before',path[0]),('After',path[choose])]:
       m=official(b,row,truth_row,span,iv,ds);assert max(abs(m[t]-values[k][t]) for t in m)<2e-12;checks+=3
      r=dict(job=job,dataset=ds,source_checkpoint=source,parent=parent,source_id=source_ids[row['source']],order=order,condition=cond,arrival=at,
       development_source_exposed=row['development_source_exposed'],expert_available=True,empty_evidence=x['empty'],selected_step=choose,
       gradient_calls=x['gradient_calls'],observations=observations(ex,row,truth_row,ia['native_boxes'],path[0],path[choose]),
       loss_path=x['losses'],step_v=[make(b).score(iv)['v'] for b in path],step_s=[make(b).s for b in path],
       gradient_norms=[h['gradient_norm'] for h in x['math_audit']['steps'] if 'gradient_norm' in h],
       step_norms=[h['step_norm'] for h in x['math_audit']['steps'] if 'step_norm' in h],
       compute=x['compute'],native_time_interval=iv,native_time_GT_headroom=af.score(span)['v']-values['After']['v'],
       loss_down_task_down=x['losses'][choose]<x['losses'][0]-1e-12 and values['After']['v']<values['Before']['v']-1e-10)
      for k,m in values.items():
       r.update({k+'_'+t:v for t,v in m.items()});r[k+'_R03']=float(m['v']>.3);r[k+'_R05']=float(m['v']>.5)
      r.update(delta_total_v=r['After_v']-r['Frozen_v'],delta_inherited_v=r['Before_v']-r['Frozen_v'],delta_current_v=r['After_v']-r['Before_v'],
       delta_total_s=r['After_s']-r['Frozen_s'],delta_R03=r['After_R03']-r['Frozen_R03'],delta_R05=r['After_R05']-r['Frozen_R05'])
      seen={o['position'] for o in r['observations']};known=bf.fids;seen_ids={row['frame_ids'][j] for j in seen};mask=np.isin(known,list(seen_ids));di=af.iou-bf.iou
      r.update(observed_GT_frames=int(mask.sum()),unobserved_GT_frames=int((~mask).sum()),observed_frame_iou_delta=float(di[mask].mean()) if mask.any() else None,
       unobserved_frame_iou_delta=float(di[~mask].mean()) if (~mask).any() else None)
      stream.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');rows.append(r);prev=committed;prevsha=rc['sha256'];total+=1
      if total%250==0:
       status(BASE/'CPU_STATUS.json',dict(status='running_full_score_and_diagnosis',job=job,done=total,total=330840,checks=checks,GT_after_global_seal=True,pid=os.getpid(),seconds=time.time()-tick))
       print('FIXED_FULL_CPU',job,total,330840,checks,flush=True)
    files[str(output.relative_to(PUB))]=sha(output)
  fields=[k+'_'+v for k in ['Frozen','Before','After'] for v in ['v','t','s','R03','R05']]+['delta_total_v','delta_inherited_v','delta_current_v','delta_total_s','delta_R03','delta_R05','native_time_GT_headroom']
  summary={};diagnosis={}
  for cohort in ['full','outside_development','development']:
   subset=[r for r in rows if cohort=='full' or r['development_source_exposed']==(cohort=='development')];summary[cohort]={};diagnosis[cohort]={}
   for group in ['corruption','clean',*CONDS[1:],'order1','order2']:
    rr=[r for r in subset if (r['condition']!='clean' if group=='corruption' else r['condition']!='clean' and r['order']==group if group.startswith('order') else r['condition']==group)]
    summary[cohort][group]=source_summary(rr,fields)
    obs=[o for r in rr for o in r['observations']];known=[o for o in obs if o['GT_known']]
    d=dict(cells=len(rr),empty_evidence=sum(r['empty_evidence'] for r in rr),loss_down_task_down=sum(r['loss_down_task_down'] for r in rr),
     transitions={name:stage_transitions(rr,a,b) for name,a,b in [('LN_inheritance','Frozen_v','Before_v'),('current_correction','Before_v','After_v'),('total','Frozen_v','After_v')]},
     tails={f'harm_gt{n}pp':sum(r['delta_total_v']< -n/100 for r in rr) for n in [5,20]},
     observations=dict(total=len(obs),GT_known=len(known),outside_known_event=len(obs)-len(known),admitted=sum(o['admitted'] for o in obs),
      rejected_correct=sum(o['rejected_correct'] for o in known),candidate_support_correct=sum(o['support_correct'] for o in known),
      critic_missed_correct=sum(o['missed_correct'] for o in known),admitted_wrong=sum(o['admitted'] and not o['chosen_correct'] for o in known),
      rejection_reasons=dict(collections.Counter(o['reason'] for o in obs if not o['admitted']))))
    diagnosis[cohort][group]=d
  write(target/'SUMMARY.json',summary);write(target/'PIPELINE_DIAGNOSIS.json',diagnosis)
  write(target/'ROOT_AUDIT.json',dict(status='pass',cells=len(rows),cumulative_checks=checks,all_state_chains_and_step_losses=True,
   all_live_NumPy_Adam_receipts=True,cumulative_raw_math_samples=samples,official_dense_vs_independent_all_readouts=True,
   GT_provenance=provenance,GT_after_global_seal=True,source_cluster_bootstrap=10000,source_macro_then_conditions_then_orders=True,
   decoder_Jacobian_independently_replayed=False,time=time.time()))
  all_summaries[job]=summary;diagnoses[job]=diagnosis
  for name in ['SUMMARY.json','PIPELINE_DIAGNOSIS.json','ROOT_AUDIT.json']:files[str((target/name).relative_to(PUB))]=sha(target/name)
 assert total==330840 and not torch.cuda.is_initialized()
 write(PUB/'SUMMARY.json',all_summaries);write(PUB/'PIPELINE_DIAGNOSIS.json',diagnoses)
 write(PUB/'SCORING_COMPLETION.json',dict(status='completed_pending_root_visual_publication',cells=total,checks=checks,raw_math_samples=samples,
  files=files,GT_after_global_seal=True,GPU_initialized=False,seconds=time.time()-tick,time=time.time()))
 write(BASE/'CPU_COMPLETION.json',dict(status='completed_pending_root_visual_publication',cells=total,time=time.time(),public_dir=str(PUB)))
 archive('全部330840全量结果及本轮GT pipeline诊断CPU实际评分/状态和dense审计已完成；根须回读报告、画图并完成代码/匿名结果GitHub逐远端核验，尚非最终收尾')
if __name__=='__main__':run()
