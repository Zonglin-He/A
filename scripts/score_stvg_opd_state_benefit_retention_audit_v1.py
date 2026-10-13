"""Offline labels for every sealed cached boundary; no state/step selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_benefit_retention_common_v1 import *

def labelled(record,value,raw,truth,span,independent=False,frame_sink=None):
 ids=value['frame_ids'];K=record['config']['steps'];F=len(ids);boxes=value['boxes']
 assert raw['frame_ids']==ids and boxes.shape==(3,K+1,F,4)
 avail=[j for j,fid in enumerate(ids) if fid in truth and span[0]<=fid<span[1]]
 visited=set(map(int,value['visited'][0]));union=visited|set(map(int,value['visited'][1]))
 masks=dict(own_U=[j for j in avail if j not in visited],common_U=[j for j in avail if j not in union],observed=[j for j in avail if j in visited],all_frames=list(range(F)))
 target=np.array([truth[ids[j]] for j in avail],float).reshape(-1,4);quality=np.full((3,K+1,F),np.nan)
 for ai,a in enumerate(ARMS):
  for k in range(K+1):
   pixel=corners(boxes[ai,k],raw['input']['width'],raw['input']['height'],record['dataset']=='hc2',independent)
   quality[ai,k,avail]=iou(pixel[avail],target,independent)
 parent=dict(parent_id=record['index'],dataset=record['dataset'],total_steps=K,historical_exposure=True,
  genuine_output_frames=F,GT_event_real_frames=len(avail),visited_Omega0=len(visited),admitted_Omega0=len(value['positions']),arms={},contrasts={})
 curves=[];early=read(BASE/'DESIGN_LOCK.json')['early_quarter_steps'][record['dataset']]
 last_start=read(BASE/'DESIGN_LOCK.json')['persistent_final_quarter_start'][record['dataset']]
 def q(ai,k,mask):return mean(quality[ai,k,mask])
 for ai,a in enumerate(ARMS):
  v=dict(empty_evidence=value['arm_metrics'][a][0]['empty_evidence'],teacher_before=value['arm_metrics'][a][0]['teacher_agreement'],
   teacher_after=value['arm_metrics'][a][-1]['teacher_agreement'],teacher_gain=difference(value['arm_metrics'][a][-1]['teacher_agreement'],value['arm_metrics'][a][0]['teacher_agreement']))
  for name in MASKS:
   mask=masks[name];b=q(ai,0,mask);af=q(ai,K,mask)
   v.update({name+'_frames':len(mask),name+'_before':b,name+'_after':af,name+'_gain':difference(af,b),name+'_early_gain':difference(q(ai,early,mask),b)})
  for name in ['joint','QR','LN']:
   v[name+'_total_step_norm']=value['arm_metrics'][a][-1][name+'_cumulative_step_norm']
   v[name+'_initial_gradient_norm']=value['arm_metrics'][a][0][name+'_gradient_norm']
   v[name+'_displacement_norm']=value['arm_metrics'][a][-1][name+'_displacement_norm']
  parent['arms'][a]=v
 for a,b in PAIRS:
  key=a+'_to_'+b;ai,bi=ARMS.index(a),ARMS.index(b);v={}
  for name,mask in masks.items():
   r0=value['output_dist'][key][0][0][mask];rK=value['output_dist'][key][K][0][mask]
   d0=mean(r0);dK=mean(rK)
   v[name+'_RMS_before']=d0;v[name+'_RMS_after']=dK;v[name+'_RMS_delta']=difference(dK,d0)
   v[name+'_RMS_ratio']=dK/d0 if d0 is not None and d0>EPS else None
   v[name+'_box_agreement_gain']=difference(mean(value['output_dist'][key][K][1][mask]),mean(value['output_dist'][key][0][1][mask]))
   v[name+'_converged_frame_fraction']=mean((rK<r0-EPS).astype(float))
   if name=='all_frames':continue
   A0=difference(q(bi,0,mask),q(ai,0,mask));AK=difference(q(bi,K,mask),q(ai,K,mask));Ae=difference(q(bi,early,mask),q(ai,early,mask))
   v[name+'_initial_advantage']=A0;v[name+'_final_advantage']=AK;v[name+'_erosion']=difference(AK,A0)
   v[name+'_early_erosion']=difference(Ae,A0);v[name+'_late_erosion']=difference(AK,Ae)
   v[name+'_retention_ratio']=AK/A0 if A0 is not None and A0>=.01 else None
   v[name+'_first_gain']=difference(q(ai,K,mask),q(ai,0,mask));v[name+'_second_gain']=difference(q(bi,K,mask),q(bi,0,mask))
   v[name+'_first_before']=q(ai,0,mask);v[name+'_second_before']=q(bi,0,mask)
   if A0 is not None:assert abs(v[name+'_erosion']-(v[name+'_second_gain']-v[name+'_first_gain']))<3e-15
   stratum=quality[bi,0,mask]>=.5;advantages=quality[bi,0,mask]-quality[ai,0,mask]
   conditions=dict(initial_positive=advantages>EPS,initial_1pp=advantages>=.01,initial_5pp=advantages>=.05,initial_correct=stratum,
    correct_advantaged=(stratum & (advantages>=.01)))
   drops=quality[bi,K,mask]-quality[bi,0,mask];catchup=quality[ai,K,mask]-quality[ai,0,mask]
   er=(quality[bi,K,mask]-quality[ai,K,mask])-advantages
   persistent=np.all(quality[bi,last_start:K+1][:,mask]<quality[bi,0,mask][None,:]-EPS,axis=0) if len(mask) else np.array([],bool)
   for label,keep in conditions.items():
    n=int(keep.sum());prefix=name+'_'+label
    v[prefix+'_frames']=n;v[prefix+'_absolute_loss_frames']=int((keep & (drops< -EPS)).sum())
    v[prefix+'_loss5pp_frames']=int((keep & (drops<-.05)).sum());v[prefix+'_persistent_loss_frames']=int((keep & persistent).sum())
    v[prefix+'_erosion_frames']=int((keep & (er< -EPS)).sum());v[prefix+'_erosion_with_improvement_frames']=int((keep & (er< -EPS) & (drops>=-EPS)).sum())
    v[prefix+'_mean_second_gain']=mean(drops[keep]);v[prefix+'_persistent_loss_fraction']=mean(persistent[keep].astype(float))
    v[prefix+'_mean_erosion']=mean(er[keep]);v[prefix+'_mean_first_gain']=mean(catchup[keep])
  v['second_teacher_gain']=parent['arms'][b]['teacher_gain'];v['first_teacher_gain']=parent['arms'][a]['teacher_gain']
  v['initial_joint_gradient_cosine']=value['pair_metrics'][key][0]['joint_gradient_cosine']
  v['initial_LN_gradient_cosine']=value['pair_metrics'][key][0]['LN_gradient_cosine']
  for name in ['joint','QR','LN']:
   v[name+'_state_RMS_before']=float(value['state_dist'][key][name][0]);v[name+'_state_RMS_after']=float(value['state_dist'][key][name][K])
  for name in MASKS:
   evidence=v['second_teacher_gain'];er=v[name+'_erosion'];n=v[name+'_correct_advantaged_frames']
   v[name+'_joint_agreement_erosion_persistent']=None if evidence is None or er is None or not n else bool(evidence>EPS and er< -EPS and v[name+'_correct_advantaged_persistent_loss_frames']>0)
  parent['contrasts'][key]=v
 for k in range(K+1):
  curve=dict(parent_id=record['index'],dataset=record['dataset'],step=k,total_steps=K,arms={},contrasts={})
  for ai,a in enumerate(ARMS):
   v=dict(value['arm_metrics'][a][k])
   for name in MASKS:v[name+'_Q']=q(ai,k,masks[name]);v[name+'_gain']=difference(q(ai,k,masks[name]),q(ai,0,masks[name]))
   curve['arms'][a]=v
  for a,b in PAIRS:
   key=a+'_to_'+b;ai,bi=ARMS.index(a),ARMS.index(b);v=dict(value['pair_metrics'][key][k])
   for name,mask in masks.items():
    v[name+'_RMS']=mean(value['output_dist'][key][k][0][mask]);v[name+'_box_agreement']=mean(value['output_dist'][key][k][1][mask])
    if name!='all_frames':
     v[name+'_advantage']=difference(q(bi,k,mask),q(ai,k,mask));v[name+'_erosion']=difference(v[name+'_advantage'],parent['contrasts'][key][name+'_initial_advantage'])
   for name in ['joint','QR','LN']:v[name+'_state_RMS']=float(value['state_dist'][key][name][k])
   curve['contrasts'][key]=v
  curves.append(curve)
  if frame_sink is not None:
   for j in range(F):
    row=dict(parent_id=record['index'],dataset=record['dataset'],frame_ordinal=j,step=k,total_steps=K,
     GT_event_available=j in avail,own_U=j in masks['own_U'],common_U=j in masks['common_U'],observed=j in masks['observed'],admitted=j in value['positions'])
    for ai,a in enumerate(ARMS):row[a+'_Q']=None if j not in avail else float(quality[ai,k,j])
    for a,b in PAIRS:
     key=a+'_to_'+b;row[key+'_RMS']=float(value['output_dist'][key][k][0][j]);row[key+'_box_agreement']=float(value['output_dist'][key][k][1][j])
    frame_sink(row)
 return parent,curves

def regression(rr,key):
 valid=[r for r in rr if r['contrasts'][key]['own_U_erosion'] is not None]
 v=[r['contrasts'][key] for r in valid];y=np.array([r['own_U_erosion'] for r in v]);X=np.array([[1,r['own_U_first_before'],r['own_U_second_before']] for r in v])
 if len(y)<4 or np.var(y)<EPS:return dict(n=len(y),LOO_R2=None,correlation_initial_advantage_erosion=None,coefficients=None)
 predicted=[]
 for i in range(len(y)):
  keep=np.arange(len(y))!=i;beta=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0];predicted.append(float(X[i]@beta))
 beta=np.linalg.lstsq(X,y,rcond=None)[0];a=np.array([r['own_U_initial_advantage'] for r in v])
 corr=float(np.corrcoef(a,y)[0,1]) if np.var(a)>EPS else None
 return dict(n=len(y),LOO_R2=float(1-np.square(y-np.array(predicted)).sum()/np.square(y-y.mean()).sum()),
  correlation_initial_advantage_erosion=corr,coefficients=beta.tolist(),causal_headroom_explanation_established=False)
def summarise(parents,curves,independent=False):
 out=dict(parents=64,fits=192,saved_updates=4800,arm_boundaries=4992,bootstrap=BOOT,seed=SEED,historical_exposure=True,datasets={})
 for ds in ['hc2','vidstg']:
  pp=[r for r in parents if r['dataset']==ds];cc=[r for r in curves if r['dataset']==ds];assert len(pp)==32
  domain=dict(arms={},contrasts={},curves=[])
  for a in ARMS:
   keys=list(pp[0]['arms'][a]);domain['arms'][a]={k:distribution([p['arms'][a][k] for p in pp],independent) for k in keys if not isinstance(pp[0]['arms'][a][k],bool)}
   domain['arms'][a]['empty_fits']=sum(p['arms'][a]['empty_evidence'] for p in pp)
  for a,b in PAIRS:
   key=a+'_to_'+b;vv=[p['contrasts'][key] for p in pp];metrics={}
   for k in vv[0]:
    if k.endswith('_frames'):
     metrics[k]=dict(total_frames=sum(v[k] for v in vv),parent_counts=[v[k] for v in vv])
    elif k.endswith('persistent'):
     metrics[k]=dict(unknown_parents=sum(v[k] is None for v in vv),true_parents=sum(v[k] is True for v in vv),false_parents=sum(v[k] is False for v in vv))
    else:metrics[k]=distribution([v[k] for v in vv],independent)
   for name in MASKS:
    vals=[v[name+'_erosion'] for v in vv];defined=[x for x in vals if x is not None];n=len(defined)
    order=sorted([i for i,x in enumerate(vals) if x is not None],key=lambda i:(-abs(vals[i]),pp[i]['parent_id']))
    metrics[name+'_robustness']=dict(all_parent_estimate=estimate(vals,independent),
     leave_one_out_mean_range=[min(mean(defined[:i]+defined[i+1:]) for i in range(n)),max(mean(defined[:i]+defined[i+1:]) for i in range(n))] if n>1 else None,
     largest1_removed=estimate([x for i,x in enumerate(vals) if i not in order[:1]],independent),
     largest3_removed=estimate([x for i,x in enumerate(vals) if i not in order[:3]],independent),
     largest_absolute_erosion_parent_ids=[pp[i]['parent_id'] for i in order[:3]],
     initial_to_final_advantage_signs=dict(collections.Counter(sign(v[name+'_initial_advantage'])+'_'+sign(v[name+'_final_advantage']) for v in vv)))
   metrics['starting_quality_association']=regression(pp,key);domain['contrasts'][key]=metrics
  K=pp[0]['total_steps']
  for k in range(K+1):
   chosen=[r for r in cc if r['step']==k];assert len(chosen)==32;step=dict(step=k,arms={},contrasts={})
   for group,names in [('arms',ARMS),('contrasts',[a+'_to_'+b for a,b in PAIRS])]:
    for name in names:
     step[group][name]={metric:estimate([r[group][name][metric] for r in chosen],independent) for metric in chosen[0][group][name] if not isinstance(chosen[0][group][name][metric],bool)}
   domain['curves'].append(step)
  out['datasets'][ds]=domain
 return out

def assemble(independent=False,frame_sink=None):
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();require_seal()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_decota_paper_main_v1 import read_row
 records=read(BASE/'DESIGN_LOCK.json')['records'];targets={ds:truths(ds,dict(parents=[r['query_ordinal'] for r in records if r['dataset']==ds])) for ds in ['hc2','vidstg']}
 parents=[];curves=[];provenance={};count=collections.Counter()
 for r in records:
  path=BASE/'readouts'/f"{r['index']:05}.pt";value=load(path);assert sha(path)==read(BASE/'GLOBAL_READOUT_BARRIER.json')['files'][str(path.relative_to(ROOT))]
  raw=read_row(r['dataset'],r['query_ordinal']);truth,span,binding=targets[r['dataset']];provenance.update(binding)
  p,cc=labelled(r,value,raw,truth[r['query_ordinal']],span[r['query_ordinal']],independent,frame_sink);parents.append(p);curves.extend(cc)
  count['parents']+=1;count['parent_boundaries']+=len(cc);count['frame_boundaries']+=len(value['frame_ids'])*len(cc)
  status(BASE/('ROOT_LABEL_STATUS.json' if independent else 'CPU_STATUS.json'),dict(status='independent_cached_frame_label_readback' if independent else 'postseal_offline_frame_labels',done=len(parents),total=64,CPU_only=True,GT_read=True,time=time.time()))
  print('RETENTION_LABELS',independent,len(parents),64,flush=True)
 assert len(curves)==1664 and not torch.cuda.is_initialized();verify();return parents,curves,dict(count),provenance
def score():
 require_seal();write(BASE/'GT_EXPOSURE.json',dict(status='offline_diagnostic_labels_after_all_cached_readouts_sealed',
  barrier_sha256=sha(BASE/'GLOBAL_READOUT_BARRIER.json'),historical_GT_exposure=True,GT_for_state_step_parameter_selection=False,CPU_only=True,GT_read=True,time=time.time()))
 with row_writer(OUT/'FRAME_ROUND_ROWS.jsonl.gz') as f:pp,cc,count,provenance=assemble(frame_sink=lambda r:emit(f,r))
 write_rows(OUT/'PARENT_ROWS.jsonl.gz',pp);write_rows(OUT/'ROUND_ROWS.jsonl.gz',cc);write(OUT/'SUMMARY.json',summarise(pp,cc))
 # Keep original full dense and genuine-frame endpoint readouts, with a separate root recomputation.
 from scripts.score_stvg_opd_LN_state_intervention_v1 import assemble as endpoints
 endpoint_rows,_,endpoint_provenance=endpoints();write_rows(OUT/'DENSE_ENDPOINT_ROWS.jsonl.gz',endpoint_rows);provenance.update(endpoint_provenance)
 table=[]
 for p in pp:
  for key,v in p['contrasts'].items():
   table.append(dict(parent_id=p['parent_id'],dataset=p['dataset'],contrast=key,**v))
 with (OUT/'ALL_PARENT_CONTRASTS.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',counts=count,parents=64,fits=192,
  frame_round_rows=count['frame_boundaries'],parent_round_rows=1664,GT_provenance=provenance,new_model_calls=0,new_optimizer_steps=0,CPU_only=True,GT_read=True,time=time.time()))
