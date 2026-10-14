"""All student outputs sealed before independent metrics and GT-gradient diagnosis."""
import sys,time,statistics,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.stvg_temporal_caption_query_v1 import ROOT,BASE,sha,read,write,roster,catalog
from scripts.stvg_temporal_shared_readout_v1 import CFG
from scripts.stvg_sparse_r1_student_v1 import _target_pair
from methods.decota_final_simplified_v1.objectives import project,temporal_loss
from scripts.stvg_opd_p6_existing_common001 import config
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,score_tube_xyxy,_metric_functions

def cosine(a,b):
 na=float(a.norm());nb=float(b.norm())
 return dict(norm_QA=na,norm_GT=nb,cosine=float(torch.dot(a.flatten(),b.flatten())/(a.norm()*b.norm())) if na and nb else None,reliable_cosine=float(torch.dot(a.flatten(),b.flatten())/(a.norm()*b.norm())) if na>1e-12 and nb>1e-12 else None,norm_floor_diagnostic_only=1e-12)
def structures(case,hidden,candidates):
 ev=project(case['zero']['logits'],case['zero']['actions'],case['records'],config(case['dataset']));hs=[h.double().cpu() for h in hidden];features=[];scores=[]
 for c in candidates:
  vals=[];emb=[]
  for h,e,record in zip(hs,ev['offsets'],case['records']):
   s,t,_=_target_pair(record['frame_ids'],c['interval_frames']);ij=e['ij'].cpu();at=int(torch.nonzero((ij[0]==s)&(ij[1]==t))[0]);vals.append(e['score'][at].double().cpu());emb.append(torch.stack([h[s],h[t]],1))
  scores.append(torch.stack(vals).mean());features.append(torch.stack(emb).mean(0))
 return ev,hs,torch.stack(scores),torch.stack(features)
def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 for runtime in [BASE/'RUNTIME.json',BASE/'STUDENT_RUNTIME.json',BASE/'visual_dependency_diagnostic001/RUNTIME.json']:
  rt=read(runtime)
  for n,h in rt['pins'].items():assert sha(ROOT/n)==h,n
 for p in [BASE/'TEACHER_GLOBAL_BARRIER.json',BASE/'STUDENT_GLOBAL_BARRIER.json',BASE/'visual_dependency_diagnostic001/GLOBAL_BARRIER.json']:
  seal=read(p);assert seal['status']=='complete' and seal['GT_read'] is False
  for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 assert read(BASE/'STUDENT_GLOBAL_BARRIER.json')['count']==8
 assert sha(ROOT/'methods/CURRENT_METHOD.json')==read(BASE/'RUNTIME.json')['protected']['methods/CURRENT_METHOD.json']
 cases={r['case_id']:catalog(r)[0] for r in roster()};data={r['case_id']:torch.load(BASE/'student'/f"{r['case_id']}.pt",map_location='cpu',weights_only=False) for r in roster()};jobs=read(BASE/'CAPTION_JOB_LOCK.json')['records'];diagnostic=BASE/'visual_dependency_diagnostic001'
 max_prob_error=0.;max_loss_error=0.;max_tf_error=0.;vision=[];teacher=[]
 for job in jobs:
  key=job['case_id']+'_'+job['candidate']['canonical_id'];a=read(BASE/'teacher'/f'{key}_alignment.json');cap=read(BASE/'teacher'/f'{key}_caption.json');v=read(diagnostic/'outputs'/f'{key}_original_likelihood.json');n=read(diagnostic/'outputs'/f'{key}_no_video_likelihood.json');gain=read(diagnostic/'gain'/f'{key}.json')
  logits=np.asarray(a['label_logits'],float);prob=np.exp(logits-logits.max());prob/=prob.sum();err=float(np.max(abs(prob-a['probabilities'])));max_prob_error=max(max_prob_error,err);assert err<2e-7 and abs(a['reward']-(a['probabilities'][0]-a['probabilities'][1]))<1e-7
  assert a['caption_sha256']==sha(BASE/'teacher'/f'{key}_caption.json') and a['source_case_sha256']==job['source_case_sha256'] and a['query_is_own_case']
  assert v['description_sha256']==n['description_sha256']==hashlib.sha256(cap['raw_caption'].encode()).hexdigest() and v['description_token_ids_sha256']==n['description_token_ids_sha256'] and v['description_token_count']==n['description_token_count']
  for r in [v,n]:
   err=max(abs(np.sum(r['per_token_log_prob'])-r['log_prob_sum']),abs(np.mean(r['per_token_log_prob'])-r['mean_log_prob']));max_tf_error=max(max_tf_error,err);assert err<1e-4
  assert abs(gain['visual_information_gain_per_token']-(v['mean_log_prob']-n['mean_log_prob']))<1e-12
  vision.append(dict(case_id=job['case_id'],dataset=job['dataset'],candidate_role=job['candidate']['source_candidate_id'],gain_per_token=gain['visual_information_gain_per_token'],token_count=v['description_token_count']))
  teacher.append(dict(case_id=job['case_id'],dataset=job['dataset'],candidate_id=job['candidate']['canonical_id'],candidate_role=job['candidate']['source_candidate_id'],reward=a['reward'],probabilities=a['probabilities'],caption_truncated=cap['output_truncated_at_cap']))
 for cid,z in data.items():
  assert z['GT_read'] is False and z['temporal_fit']['active_parameters']==512 and z['spatial_fit']['active_parameters']==1792
  local=[j for j in jobs if j['case_id']==cid];ev,h,b,f=structures(cases[cid],z['temporal_hidden'],[j['candidate'] for j in local]);fit=z['temporal_fit'];inds=fit['valid_candidate_indices'];bb=b[inds];ff=f[inds];r=fit['rewards'];target=(bb+r/CFG['tau_T']).softmax(0)
  if fit['no_information']:target=bb.softmax(0)
  assert torch.allclose(target,fit['target'],atol=1e-15,rtol=0) and torch.allclose(bb.softmax(0),fit['p0'],atol=1e-15,rtol=0)
  for step in fit['path']:
   theta=step['theta'];lp=(bb+(ff*theta).sum((1,2))).log_softmax(0);loss=float((target*(target.clamp_min(1e-300).log()-lp)).sum()+CFG['l2']*theta.square().sum());max_loss_error=max(max_loss_error,abs(loss-step['loss']));assert abs(loss-step['loss'])<2e-12
   intervals=[]
   for hh,e,rec in zip(h,ev['offsets'],cases[cid]['records']):
    ij=e['ij'].cpu();score=e['score'].double().cpu()+hh[ij[0]]@theta[:,0]+hh[ij[1]]@theta[:,1];at=int(score.argmax());s,t=map(int,ij[:,at]);intervals.append([rec['frame_ids'][s],rec['frame_ids'][t]+1])
   assert step['physical_interval']==[min(v[0] for v in intervals),max(v[1] for v in intervals)]
  assert z['arms']['T']['interval']==fit['path'][-1]['physical_interval'] and fit['path'][0]['physical_interval']==cases[cid]['Projection_interval']
  assert np.array_equal(z['arms']['F']['boxes'],z['arms']['T']['boxes']) and np.array_equal(z['arms']['S']['boxes'],z['arms']['TS']['boxes'])
 # All model results above verified before this sole diagnostic label read.
 truths,bindings=_truths_after_seal(cases);official,independent,_=_metric_functions();rows=[];gradients=[];maxmetric=0.;metriccount=0
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 for rec in roster():
  cid=rec['case_id'];case=cases[cid];z=data[cid];gt=truths[cid];local=[j for j in jobs if j['case_id']==cid];ev,hs,b,feat=structures(case,z['temporal_hidden'],[j['candidate'] for j in local]);fit=z['temporal_fit'];valid=fit['valid_candidate_indices'];bb=b[valid];ff=feat[valid];target=fit['target'].detach();theta=torch.zeros(256,2,dtype=torch.float64,requires_grad=True)
  Lqa=(target*(target.clamp_min(1e-300).log()-(bb+(ff*theta).sum((1,2))).log_softmax(0))).sum();gqa=torch.autograd.grad(Lqa,theta)[0]
  analytic=(ff*((bb.softmax(0)-target)[:,None,None])).sum(0);assert torch.allclose(gqa,analytic,atol=1e-14,rtol=0)
  def gt_loss(prior):
   loss=theta.sum()*0
   for h,e,rr,raw in zip(hs,ev['offsets'],case['records'],case['zero']['logits']):
    ij=e['ij'].cpu();s,t,overlap=_target_pair(rr['frame_ids'],gt['span']);assert overlap>0;at=int(torch.nonzero((ij[0]==s)&(ij[1]==t))[0]);offset=h[ij[0]]@theta[:,0]+h[ij[1]]@theta[:,1]
    base=e['score'].double().cpu() if prior=='projection' else raw[0,ij[0],0].double()+raw[0,ij[1],1].double();loss-= (base+offset).log_softmax(0)[at]/2
   return loss
  ggt=torch.autograd.grad(gt_loss('projection'),theta)[0];ggt_native=torch.autograd.grad(gt_loss('native'),theta)[0]
  adapted=[raw.double()+h@theta for raw,h in zip(case['zero']['logits'],hs)];gssl=torch.autograd.grad(temporal_loss(adapted,ev),theta)[0]
  gradient=dict(case_id=cid,dataset=rec['dataset'],QA_vs_GT_shared_S0=cosine(gqa,ggt),SSL_replay_vs_same_GT_shared_S0=cosine(gssl,ggt),QA_vs_GT_native_score=cosine(gqa,ggt_native),SSL_replay_vs_GT_native_score=cosine(gssl,ggt_native),QA_analytic_gradient_max_error=float((gqa-analytic).abs().max()),scope='512_same_frozen_hidden_weights_offline_only_not_historical_66306_gradient',GT_gradient_used_for_update=False)
  gradients.append(gradient)
  def metric(box,iv):
   nonlocal maxmetric,metriccount
   x=score_tube_xyxy(box,z['frame_ids'],case['width'],case['height'],iv,gt['truth'],gt['span'],rec['dataset'],official=official,independent=independent);maxmetric=max(maxmetric,x['official_independent_max_abs_error']);metriccount+=1;return x['metrics']
  arms={k:metric(v['boxes'],v['interval']) for k,v in z['arms'].items()};path=[]
  ssteps=z['spatial_fit']['path']+[dict(step=5,boxes=z['spatial_fit']['final_cached_decoder_boxes'])]
  for k in range(6):
   cx=ssteps[k]['boxes'].double().numpy();box=np.concatenate((cx[:,:2]-cx[:,2:]/2,cx[:,:2]+cx[:,2:]/2),1);iv=fit['path'][k]['physical_interval'];path.append(dict(step=k,temporal_loss=fit['path'][k]['loss'],spatial_loss=z['spatial_fit']['rounds'][k]['loss'] if k<5 else None,temporal_parameter_norm=fit['path'][k]['parameter_norm'],T=metric(z['arms']['F']['boxes'],iv),S=metric(box,case['Native_interval']),TS=metric(box,iv)))
  def cxcywh(box):return np.concatenate(((box[:,:2]+box[:,2:])/2,box[:,2:]-box[:,:2]),1)
  rowmeta={'frame_ids':z['frame_ids'],'input':{'width':case['width'],'height':case['height']}};frozen=DenseTube(cxcywh(z['arms']['F']['boxes']),rowmeta,gt['truth'],gt['span'],clip=rec['dataset']=='hc2');space=DenseTube(cxcywh(z['arms']['S']['boxes']),rowmeta,gt['truth'],gt['span'],clip=rec['dataset']=='hc2');mask=np.isin(sorted(gt['truth']),[z['frame_ids'][p] for p in z['supervised_positions']]);delta=space.iou-frozen.iou
  rows.append(dict(case_id=cid,dataset=rec['dataset'],arms=arms,paths=path,temporal_optimizer_steps=fit['optimizer_steps'],spatial_optimizer_steps=z['spatial_fit']['optimizer_steps'],reward_range=float(fit['rewards'].max()-fit['rewards'].min()),QA_no_information=fit['no_information'],temporal_loss_drop=fit['path'][0]['loss']-fit['path'][-1]['loss'],supervised_GT_frame_count=int(mask.sum()),unobserved_GT_frame_count=int((~mask).sum()),supervised_sIoU_delta_pp=float(delta[mask].mean()*100) if mask.any() else None,unobserved_sIoU_delta_pp=float(delta[~mask].mean()*100) if (~mask).any() else None,spatial_reference_kind='selected_student_pseudoreference_not_independent_teacher',source_forward_wall_seconds=z['wall_seconds'],GPU_peak_bytes=z['GPU_peak_bytes']))
 groups=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds]
  for arm in ['F','Projection','T','S','TS']:
   metrics={m:statistics.mean(r['arms'][arm][m] for r in rr) for m in rr[0]['arms']['F']};deltas={m:statistics.mean((r['arms'][arm][m]-r['arms']['F'][m])*100 for r in rr) for m in metrics}
   groups.append(dict(dataset=ds,arm=arm,queries=len(rr),metrics_percent={m:100*v for m,v in metrics.items()},delta_vs_F_pp=deltas,positive_vIoU_count=sum(r['arms'][arm]['m_vIoU']>r['arms']['F']['m_vIoU']+1e-8 for r in rr),negative_vIoU_count=sum(r['arms'][arm]['m_vIoU']<r['arms']['F']['m_vIoU']-1e-8 for r in rr)))
 cost={}
 for stage,sealpath in [('method_teacher',BASE/'TEACHER_GLOBAL_BARRIER.json'),('auxiliary_diagnostic',diagnostic/'GLOBAL_BARRIER.json')]:
  ps=[read(ROOT/f['path']) for f in read(sealpath)['files']];cost[stage]=dict(calls=len(ps),seconds=sum(p['total_seconds'] for p in ps),input_tokens=sum(p['input_tokens'] for p in ps),output_tokens=sum(p['output_tokens'] for p in ps),GPU_peak_bytes=max(p['GPU_peak_bytes'] for p in ps))
 cost['student']=dict(cases=8,source_capture_calls=8,full_student_final_readout_calls=8,temporal_fit_cases=8,spatial_fit_cases=8,temporal_optimizer_steps=sum(r['temporal_optimizer_steps'] for r in rows),spatial_optimizer_steps=sum(r['spatial_optimizer_steps'] for r in rows),seconds=sum(r['source_forward_wall_seconds'] for r in rows),GPU_peak_bytes=max(r['GPU_peak_bytes'] for r in rows))
 counter=[]
 for cid in sorted({r['case_id'] for r in vision}):
  job=next(j for j in jobs if j['case_id']==cid and j['candidate']['source_candidate_id']=='time_Native');key=cid+'_'+job['candidate']['canonical_id'];orig=read(BASE/'teacher'/f'{key}_alignment.json')
  if not (diagnostic/'outputs'/f'{key}_no_video_alignment.json').exists():continue
  for ctrl in ['no_video','reversed','replacement']:
   new=read(diagnostic/'outputs'/f'{key}_{ctrl}_alignment.json');counter.append(dict(case_id=cid,control=ctrl,original_reward=orig['reward'],counterfactual_reward=new['reward'],reward_delta=new['reward']-orig['reward'],own_query_only=True))
 report=dict(status='complete_actual_eight_case_F_T_S_TS_and_visual_gradient_diagnostics',groups=groups,rows=rows,gradients=gradients,visual_information_gain=vision,counterfactual_controls=counter,teacher_candidate_scores=teacher,cost=cost,metric_readout_count=metriccount,official_independent_max_error=maxmetric,teacher_probability_reconstruction_max_error=max_prob_error,shared_readout_loss_reconstruction_max_error=max_loss_error,teacher_forced_token_reconstruction_max_error=max_tf_error,GT_read=True,GT_only_after_all_student_and_diagnostic_barriers=True,historically_exposed=True,fresh_confirmation=False,teacher_training_overlap='unknown',CURRENT_unchanged=True,truth_binding_metadata=bindings,time=time.time())
 write(BASE/'ROOT_POSTSEAL_REPORT.json',report);write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='source_custody_actual_shared_readout_KL_native_MAP_spatial_full_readout_and_independent_metric_diagnostic_not_universal_method_qualification',count=8,hash=sha(BASE/'ROOT_POSTSEAL_REPORT.json'),metric_readouts=metriccount,GT_read=True,CPU_only=True,time=time.time()))
 print(json.dumps({'groups':groups,'cost':cost,'gradient_summary':[{k:g[k] for k in ['case_id','QA_vs_GT_shared_S0','SSL_replay_vs_same_GT_shared_S0']} for g in gradients],'gain_summary':{ds:statistics.mean(v['gain_per_token'] for v in vision if v['dataset']==ds) for ds in ['hc2','vidstg']},'maxmetric':maxmetric},indent=2))
if __name__=='__main__':main()
