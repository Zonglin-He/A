"""Frozen-artifact domain-gap diagnosis. No training, Teacher or Student calls.

Source-only clean/noisy denoiser probes and exact existing association CPU
replays are explicitly diagnostic, not replacement benchmark predictions.
"""
from pathlib import Path
import json,sys,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_posterior_v1 import BASE,read,put,load,sha,condition,decode_state
OUT=ROOT/'artifacts/stvg_joint_posterior_gap_diagnosis_v1'
def stats(v):
 import numpy as np
 x=np.asarray(v,float);x=x[np.isfinite(x)]
 return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),q10=float(np.quantile(x,.1)),q90=float(np.quantile(x,.9)),min=float(x.min()),max=float(x.max()))if len(x)else None

def main():
 import torch,numpy as np
 from scipy.stats import rankdata,spearmanr
 from vg_tta.stvg_joint_posterior_v1 import JointPrior,Association,roi,encode_bounds,logit,bounds
 from scripts.score_stvg_opd_p1_v1 import truths
 torch.set_num_threads(4);torch.manual_seed(20261018)
 assert not torch.cuda.is_initialized()
 plan=read(BASE/'PLAN_LOCK.json');source=read(BASE/'SOURCE_VALIDATION_POSTERIOR.json');target=read(BASE/'POSTSEAL_RESULTS.json')
 if not(OUT/'PLAN_LOCK.json').exists():put(OUT/'PLAN_LOCK.json',dict(status='locked_before_CPU_replay',scope='sealed artifact gap diagnosis, no new models or benchmark arms',new_training=0,new_Student_Teacher_or_GPU_calls=0,source_only_denoiser_probe='existing source-validation24, joint model, original noise levels0/8/15, paired GT-kernel and Native-kernel inputs; one forward per input, no optimization',association_replay='existing source validation24 and target32 ROI/query through frozen heads; compare target probabilities and source aggregate to original',target_GT_role='already exposed32 existing GT for descriptive postseal distribution audit, never neural input or selection',original_predictions_and_models_immutable=True))
 targets={d:truths(d,dict(parents=[r['query_ordinal']for r in plan['target']if r['dataset']==d]))for d in ['hc2','vidstg']}
 rows=[];association=[];feature_means={};zs={};ys={};conds={}
 for which in ['source','target']:
  for r in plan[which]:
   alias=r['alias'];z=load(BASE/'features'/which/(alias+'.pt'));c=condition(z,'cpu');zs[alias]=z;conds[alias]=c;ids=np.array(z['frame_ids']);lo,hi=int(ids[0]),int(ids[-1]+1);ds=r['dataset']
   if which=='source':
    y=load(BASE/'source_supervision'/(alias+'.pt'));gt=y['dense_source_GT'];span=y['span'];truth=y['state'];mask=y['mask'];split=r['split'];ys[alias]=y
   else:
    gt=targets[ds][0][r['query_ordinal']];span=targets[ds][1][r['query_ordinal']];a=max(span[0],lo);b=min(span[1],hi);assert b>a
    keys=np.array(sorted(gt));tt=np.linspace(a,min(b-1,keys[-1]),32);valid=(tt>=keys[0])&(tt<=keys[-1]);xy=np.stack([np.interp(tt,keys,np.array([gt[int(f)]for f in keys])[:,j])for j in range(4)],-1)/np.array([z['input']['width'],z['input']['height'],z['input']['width'],z['input']['height']]);bb=np.concatenate(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]),-1)
    truth=torch.zeros(33,4);truth[0,:2]=encode_bounds(torch.tensor((a-lo)/(hi-lo)),torch.tensor((b-lo)/(hi-lo)));truth[1:]=logit(torch.tensor(bb,dtype=torch.float32));mask=torch.zeros(33,4);mask[0,:2]=1;mask[1:]=torch.tensor(valid)[:,None];split='target_exposed_development'
   native=c['native'][0];err=(native-truth).square()*mask;kl=float(err.sum()/8)
   native_iv=np.array(z['native_interval']);duration=(hi-lo)/z['input']['fps'];gt_iv=np.array(span)
   feature_means.setdefault(which+'_'+ds,[]).append(dict(visual=z['maps'].mean((1,2,3)).numpy(),query=z['query'].mean(0).numpy()))
   row=dict(alias=alias,dataset=ds,split=split,model_source_domain=ds if which=='source'else('vidstg'if ds=='hc2'else'hc2'),context_seconds=float(duration),native_frames=len(ids),query_tokens=len(z['query']),GT_event_fraction=float((span[1]-span[0])/(hi-lo)),native_start_fraction=float((native_iv[0]-lo)/(hi-lo)),native_end_fraction=float((native_iv[1]-lo)/(hi-lo)),GT_start_fraction=float((max(span[0],lo)-lo)/(hi-lo)),GT_end_fraction=float((min(span[1],hi)-lo)/(hi-lo)),GT_event_clipped_to_grid=span[0]<lo or span[1]>hi,GT_box_area_mean=float((truth[1:,2:].sigmoid().prod(1)*mask[1:,0]).sum()/mask[1:,0].sum().clamp_min(1)),native_endpoint_abs_logit_gt8=int((native[0,:2].abs()>8).sum()),GT_endpoint_abs_logit_gt8=int((truth[0,:2].abs()>8).sum()),native_to_GT_temporal_logit_RMS=float(err[0,:2].mean().sqrt()),native_to_GT_spatial_logit_RMS=float((err[1:].sum()/mask[1:].sum()).sqrt()),terminal_Gaussian_center_KL_at_sigma2=kl,terminal_temporal_center_KL=float(err[0,:2].sum()/8),terminal_spatial_center_KL=float(err[1:].sum()/8))
   predictions=load(BASE/'source_validation_predictions'/(alias+'.pt'))['predictions']if split=='source_validation'else(load(BASE/'predictions'/(alias+'.pt'))['arms']if which=='target'else None)
   if predictions:
    iv=np.array(predictions['joint_diffusion']['interval']);anchors=z['expert']['anchors']['single4'];row.update(posterior_start_fraction=float((iv[0]-lo)/(hi-lo)),posterior_end_fraction=float((iv[1]-lo)/(hi-lo)),posterior_duration_fraction=float((iv[1]-iv[0])/(hi-lo)),admitted=len(anchors),posterior_event_observations=int(sum(iv[0]<=ids[a['position']]<iv[1]for a in anchors)))
   if which=='target':
    pr=load(BASE/'predictions'/(alias+'.pt'));trace=pr['traces']['joint_diffusion'];row.update(zero_observation_gradient_levels=sum(t['observation_gradient_norm']==0 for t in trace),first_observation_gradient_norm=trace[0]['observation_gradient_norm'],final_observation_gradient_norm=trace[-1]['observation_gradient_norm'])
   rows.append(row)
   if split not in ['source_validation','target_exposed_development']:continue
   model_ds=row['model_source_domain'];ad=load(BASE/'models'/(model_ds+'_association.pt'));net=Association().eval().requires_grad_(False);net.load_state_dict(ad['state']);anchors=z['expert']['anchors']['single4']
   t=torch.tensor([[(ids[a['position']]-lo)/(hi-lo)for a in anchors]],dtype=torch.float32);boxes=torch.tensor([[a['box']for a in anchors]],dtype=torch.float32).reshape(1,-1,4)
   with torch.no_grad():pp=net(roi(c['maps'],boxes,t)[0],c['query'].mean(1).expand(len(anchors),-1)).sigmoid().numpy()if anchors else np.empty(0)
   if which=='target':assert np.max(np.abs(pp-load(BASE/'predictions'/(alias+'.pt'))['association_prior'].numpy()),initial=0)<2e-5
   from vg_tta.tastvg_oracle_event5_v1 import box_iou
   for j,a in enumerate(anchors):
    fid=int(ids[a['position']]);known=fid in gt;g=None
    if known:
     b=np.array(a['box']);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*np.array([z['input']['width'],z['input']['height'],z['input']['width'],z['input']['height']]);g=float(box_iou(xy,np.array(gt[fid])))
    association.append(dict(alias=alias,dataset=ds,split=split,model_source_domain=model_ds,anchor=j+1,probability=float(pp[j]),precise_GT_available=known,GT_geometry_IoU=g,geometric_inlier=None if g is None else g>=.5,reference_area=float(a['box'][2]*a['box'][3]),event_relative_position=float((fid-lo)/(hi-lo))))
 groups={}
 for key in [('hc2','source_train'),('hc2','source_validation'),('vidstg','source_train'),('vidstg','source_validation'),('hc2','target_exposed_development'),('vidstg','target_exposed_development')]:
  rr=[r for r in rows if(r['dataset'],r['split'])==key];keys=[k for k,v in rr[0].items()if isinstance(v,(int,float))and not isinstance(v,bool)];groups['/'.join(key)]={k:stats([r[k]for r in rr])for k in keys};groups['/'.join(key)]['GT_event_clipped_to_grid_count']=sum(r['GT_event_clipped_to_grid']for r in rr)
 gap=[]
 for src,tgt in [('hc2','vidstg'),('vidstg','hc2')]:
  a=source['means'][src];b=target['means'][tgt];dd={}
  for metric in ['v','t','s']:
   observed=b['joint_diffusion'][metric]-a['joint_diffusion'][metric];baseline=b['Direct'][metric]-a['Direct'][metric];mechanism=(b['joint_diffusion'][metric]-b['Direct'][metric])-(a['joint_diffusion'][metric]-a['Direct'][metric]);assert abs(observed-baseline-mechanism)<1e-12
   dd[metric]=dict(source_Direct=a['Direct'][metric],target_Direct=b['Direct'][metric],source_joint=a['joint_diffusion'][metric],target_joint=b['joint_diffusion'][metric],total_joint_gap=observed,baseline_gap=baseline,change_in_joint_minus_Direct=mechanism)
  gap.append(dict(source_domain=src,target_domain=tgt,decomposition=dd))
 calib={}
 for model_ds,target_ds in [('hc2','vidstg'),('vidstg','hc2')]:
  for split,ds in [('source_validation',model_ds),('target_exposed_development',target_ds)]:
   ar=[r for r in association if r['dataset']==ds and r['split']==split and r['precise_GT_available']];pp=np.array([r['probability']for r in ar]);label=np.array([r['geometric_inlier']for r in ar]);n1=int(label.sum());n0=len(ar)-n1;auc=float((rankdata(pp)[label].sum()-n1*(n1+1)/2)/(n1*n0))if n1*n0 else None
   orig=source['association'][ds]if split=='source_validation'else read(BASE/'POSTSEAL_DIAGNOSTICS.json')['target_association_geometry'][ds];assert abs(auc-orig['AUROC'])<1e-10
   calib[model_ds+'/'+split]=dict(n=len(ar),inliers=n1,AUROC=auc,mean_probability_inliers=float(pp[label].mean()),mean_probability_noninliers=float(pp[~label].mean()),confident_ge09_total=int((pp>=.9).sum()),confident_ge09_noninliers=int(((pp>=.9)&~label).sum()),Brier=float(((pp-label)**2).mean()),reference_area_correlation=float(spearmanr(pp,[r['reference_area']for r in ar]).statistic),probability_IoU_correlation=float(spearmanr(pp,[r['GT_geometry_IoU']for r in ar]).statistic))
 curves=[]
 for r in read(BASE/'TRAINING_COMPLETION.json')['rows']:
  b=r['history'][r['selected_epoch']-1];end=r['history'][-1];curves.append(dict(source_domain=r['source_domain'],kind=r['kind'],selected_epoch=r['selected_epoch'],best_train_loss=b['train_loss'],best_validation_loss=b['source_validation_loss'],last_train_loss=end['train_loss'],last_validation_loss=end['source_validation_loss'],validation_increase_ratio=end['source_validation_loss']/b['source_validation_loss'],last10_train_mean=float(np.mean([h['train_loss']for h in r['history'][-10:]]))))
 probes=[];levels=torch.logspace(math.log10(2.),math.log10(.03),16);generator=torch.Generator().manual_seed(20261018)
 for ds in ['hc2','vidstg']:
  net=JointPrior().eval().requires_grad_(False);net.load_state_dict(load(BASE/'models'/(ds+'_joint_diffusion.pt'))['state'])
  for r in [r for r in plan['source']if r['dataset']==ds and r['split']=='source_validation']:
   alias=r['alias'];c=conds[alias];y=ys[alias];truth=y['state'][None];mask=y['mask'][None];native=c['native'];eps=torch.randn(truth.shape,generator=generator)
   for level in [0,8,15]:
    sig=levels[level];pair={}
    for key,center in [('GT_centered_training_kernel',truth),('Native_centered_application_kernel',native)]:
     x=center+sig*eps;x[:,0,2:]=0
     with torch.no_grad():pred=net(x,sig[None],c)
     def loss(v):
      error=(v-truth).square()*mask;return float((error[:,0,:2].mean()+error[:,1:].sum()/mask[:,1:].sum())/2)
     def components(v):
      er=(v-truth).square()*mask
      from vg_tta.tastvg_oracle_event5_v1 import official
      decoded=decode_state(v[0],zs[alias]);md=official(decoded['boxes'],dict(frame_ids=zs[alias]['frame_ids'],input=zs[alias]['input']),ys[alias]['dense_source_GT'],ys[alias]['span'],decoded['interval'],ds)
      return dict(temporal_mse=float(er[:,0,:2].mean()),spatial_mse=float(er[:,1:].sum()/mask[:,1:].sum()),vIoU=md['v'],tIoU=md['t'],sIoU=md['s'])
     inp=components(x);out=components(pred)
     pair[key]=dict(input_loss=loss(x),one_step_output_loss=loss(pred),**{'input_'+k:v for k,v in inp.items()},**{'output_'+k:v for k,v in out.items()})
    probes.append(dict(alias=alias,dataset=ds,level=level,sigma=float(sig),paired_same_noise=True,results=pair))
 probe_means={ds:{str(level):{key:{metric:float(np.mean([r['results'][key][metric]for r in probes if r['dataset']==ds and r['level']==level]))for metric in ['input_loss','one_step_output_loss','input_temporal_mse','output_temporal_mse','input_spatial_mse','output_spatial_mse','input_vIoU','output_vIoU','input_tIoU','output_tIoU','input_sIoU','output_sIoU']}for key in ['GT_centered_training_kernel','Native_centered_application_kernel']}for level in [0,8,15]}for ds in ['hc2','vidstg']}
 assert not torch.cuda.is_initialized()
 put(OUT/'DIAGNOSIS002.json',dict(status='complete',source_target_pairing=gap,group_distributions=groups,source_training_curves=curves,association_calibration=calib,source_only_denoiser_probe_means=probe_means,rows=rows,association_rows=association,source_only_probes=probes,CPU_only=True,new_optimizer_or_training_steps=0,new_Teacher_Student_calls=0,new_benchmark_predictions=0,causality='descriptive decomposition and code-local source-only replay; no claim of controlled target causal attribution'))
 print(json.dumps(dict(pairing=gap,calibration=calib,curves=curves,source_probes=probe_means),indent=2))
if __name__=='__main__':main()
