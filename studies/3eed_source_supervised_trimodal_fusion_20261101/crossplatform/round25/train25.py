"""Waymo-only finite supervised fusion fitting; no target inputs or annotations.
Two fixed seeds; model selection uses only the held-out Waymo VAL512.
"""
import sys,time,random,numpy as np,torch
from common25 import *
from fusion25 import TrimodalFusion,bank,part
F=torch.nn.functional

def evaluate(model,data,mode):
 model.eval();qs=[];vals=[]
 with torch.no_grad():
  for j in range(0,len(data['h']),32):
   b=part(data,slice(j,j+32));s,d=model(b,mode=mode);q=s.argmax(-1);qs+=q.cpu().tolist();vals+=b['ious'].gather(1,q[:,None]).squeeze(1).cpu().tolist()
 return {'mIoU_percent':float(np.mean(vals)*100),'Acc25_percent':float(np.mean(np.array(vals)>.25)*100),'Acc50_percent':float(np.mean(np.array(vals)>.5)*100)},np.array(qs)

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;verify(BASE/'source_features_attempt002');verify(BASE/'rgb_features_attempt002');verify(EXP/'source_labels_attempt002');out=BASE/'training_attempt001';out.mkdir(exist_ok=False);(out/'states_PRIVATE').mkdir();(out/'validation_PRIVATE').mkdir();start=time.perf_counter();labels={r['id']:load_npz(EXP/'source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'] for split in ['S3682','SV512'] for r in rows(split)};train=bank(rows('S3682'),labels);val=bank(rows('SV512'),labels);print('BANK25_LOADED',round(time.perf_counter()-start,1),flush=True)
 spec={'seeds':[0,1],'variants':['rgb','no_rgb'],'epochs':30,'batch_size':32,'optimizer':'AdamW','fusion_lr':.001,'matching_projection_lr':.0001,'weight_decay':.0001,'RGB_feature_dropout':.1,'GT_IoU_temperature':.05,'source_probability_anchor':.1,'residual_energy_weight':.001,'eligible_supervision':'source candidates with oracleIoU>=.05, all failures retained in evaluation','source_validation_selection':'maximum VAL512 mIoU, earliest epoch on ties; every5epochs and epoch1','target_supervision_or_selection':False,'whole_model_finetuning':False,'code_sha256':sha(__file__),'fusion_code_sha256':sha(ROOT/'crossplatform/round25/fusion25.py')};dump(EXP/'TRAIN_SPEC_LOCK.json',spec);history=[];selected=[]
 for seed in spec['seeds']:
  for mode in spec['variants']:
   torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);model=TrimodalFusion().cuda();original={k:v.detach().clone() for k,v in model.state_dict().items()};proj=list(model.projection.parameters());other=[p for n,p in model.named_parameters() if not n.startswith('projection.')];opt=torch.optim.AdamW([{'params':other,'lr':.001},{'params':proj,'lr':.0001}],weight_decay=.0001);run=f'{mode}_seed{seed}';best=None;traces=[];maxgrad=0.;optimizer_steps=0;group_grad_max={}
   # Source-equivalent initialization must hold on real multimodal input.
   with torch.no_grad():b=part(val,slice(0,16));s,_=model(b,mode);error=float(abs(s-b['scores']).max());assert error<2e-5
   for epoch in range(1,31):
    model.train();order=torch.randperm(len(train['h']));losses=[]
    for j in range(0,len(order),32):
     b=part(train,order[j:j+32]);opt.zero_grad();s,diag=model(b,mode=mode,rgb_dropout=.1 if mode=='rgb' else 0.);logp=(s.clamp_min(1e-8).log()).log_softmax(-1);gt=(b['ious']/.05).softmax(-1);eligible=b['ious'].max(-1).values>=.05;rank=F.kl_div(logp,gt,reduction='none').sum(-1);rank=(rank*eligible).sum()/eligible.sum().clamp_min(1);p0=b['scores'].clamp_min(1e-8);p0=p0/p0.sum(-1,keepdim=True);anchor=F.kl_div(logp,p0,reduction='batchmean');loss=rank+.1*anchor+.001*diag['delta_energy'];assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.));maxgrad=max(maxgrad,grad)
     if j==0:
      for group,prefixes in {'RGB_input_projection':('key.','value.','rnorm.'),'language_condition':('text.','tnorm.'),'fusion_gate':('gate.',),'query_matching':('projection.',)}.items():
       norm=float(torch.sqrt(sum(p.grad.square().sum() for n,p in model.named_parameters() if p.grad is not None and n.startswith(prefixes))));group_grad_max[group]=max(group_grad_max.get(group,0.),norm)
     opt.step();optimizer_steps+=1;losses.append(float(loss.detach()))
    if epoch==1 or epoch%5==0:
     met,q=evaluate(model,val,mode);entry=dict(run=run,seed=seed,variant=mode,epoch=epoch,loss=float(np.mean(losses)),**met);history.append(entry);traces.append(entry);np.savez_compressed(out/'validation_PRIVATE'/(run+f'_e{epoch:02d}.npz'),query_indices=q)
     # Immutable checkpoints permit the full source-only selection history.
     statepath=out/'states_PRIVATE'/(run+f'_e{epoch:02d}.npz');np.savez_compressed(statepath,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
     if best is None or met['mIoU_percent']>best['mIoU_percent']+1e-10:best=dict(entry,state_name=statepath.name,state_sha256=sha(statepath))
     print('TRAIN25',run,epoch,round(met['mIoU_percent'],4),'elapsed',round(time.perf_counter()-start,1),flush=True)
   delta=float(torch.sqrt(sum((v-original[k]).square().sum() for k,v in model.state_dict().items() if v.is_floating_point())));selected.append(dict(best,optimizer_steps=optimizer_steps,parameter_L2_delta=delta,max_gradient_L2=maxgrad,group_gradient_L2_max=group_grad_max,trainable_parameters=sum(p.numel() for p in model.parameters()),initial_source_replay_error=error));del model,opt
 dump(out/'HISTORY.json',history);dump(EXP/'SOURCE_SELECTED_LOCK.json',{'selected':selected,'selection_source':'WaymoVAL512 only','target_labels_or_inputs_used_to_fit':False,'target_results_not_read_before_freeze':True,'primary_seed':0,'secondary_seed':1,'protocol_sha256':sha(EXP/'TRAIN_SPEC_LOCK.json')});seal(out,'R25_source_only_RGB_fusion_training_selected_before_target_scores',n_train=3682,n_validation=512,optimizer_steps=sum(s['optimizer_steps'] for s in selected),target_feature_tensors_used_to_fit=0,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('TRAIN25_SEALED',selected,flush=True)
if __name__=='__main__':main()
