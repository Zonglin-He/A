"""Two source-only continuation branches, matched exposure and outer updates."""
from core78 import *
import time,random
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
plan=read(H78/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H78/'experiment/CODE_LOCK.json'));assert read(H78/'engineering_attempt001/RESULT.json')['state'].startswith('PASS_');base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/plan['init_state']);rr=source_rows('TRAIN512');val=source_rows('VAL128');out=B78/'trained_source_attempt001';assert not out.exists();out.mkdir();(out/'states_PRIVATE').mkdir();(out/'val_PRIVATE').mkdir();history=[];start=time.perf_counter();counts={};cache={}
for r in rr:
 z0=prefix(r);cache[r['id']]=(z0,load(casepath(r)),load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'])
for branch in ['joint','meta']:
 torch.manual_seed(0);np.random.seed(0);random.seed(0);m=Fusion78(base.contrastive_align_projection_image).cuda().eval();restore(m,S);m.requires_grad_(True);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.0001);counts[branch]={'outer_updates':0,'grounding_training_suffix_calls':0,'auxiliary_masked_fusion_calls':0,'validation_suffix_calls':0,'inner_gradient_calls':0,'outer_backward_calls':0};c=counts[branch]
 for epoch in range(1,5):
  m.requires_grad_(True);st=time.perf_counter();ls=[];auxs=[];g0s=[];g1s=[];order=np.random.default_rng(78000+epoch).permutation(len(rr))
  for j in order:
   r=rr[int(j)];z0,d0,y0=cache[r['id']];z=tensor_prefix(z0);d=tensor_data(d0,r);y=torch.tensor(y0,dtype=torch.float64,device='cuda');pm,im=maskpair(r,d,(78000+2*epoch,78001+2*epoch));opt.zero_grad(set_to_none=True)
   if branch=='meta':fast,aux,gn=inner78(m,z,d,pm,im,create_graph=True);c['inner_gradient_calls']+=1
   else:aux=auxiliary78(m,z,d,pm,im)
   c['auxiliary_masked_fusion_calls']+=2;p0,_,_=main78(m,base,z,d);g0=ground78(p0,y);c['grounding_training_suffix_calls']+=1
   if branch=='meta':p1,_,_=main78(m,base,z,d,fast);g1=ground78(p1,y);loss=.5*g0+.5*g1+aux;c['grounding_training_suffix_calls']+=1
   else:g1=None;loss=g0+aux
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(norm);opt.step();c['outer_updates']+=1;c['outer_backward_calls']+=1;ls.append(float(loss.detach()));auxs.append(float(aux.detach()));g0s.append(float(g0.detach()));g1s.append(float(g1.detach()) if g1 is not None else None)
   if len(ls)%128==0:print('TRAIN78',branch,epoch,len(ls),'loss',round(np.mean(ls),5),'seconds',round(time.perf_counter()-st,1),flush=True)
  path=out/'states_PRIVATE'/(branch+'_e'+str(epoch)+'.npz');np.savez_compressed(path,**state(m));record,details=evaluate_source(m,base,val,out/'val_PRIVATE'/(branch+'_e'+str(epoch)+'.npz'));c['validation_suffix_calls']+=256;row={'branch':branch,'epoch':epoch,**record,'outer_loss_mean':float(np.mean(ls)),'auxiliary_mean':float(np.mean(auxs)),'train_ground0_mean':float(np.mean(g0s)),'train_ground1_mean':float(np.mean(g1s)) if branch=='meta' else None,'seconds':time.perf_counter()-st,'state_path':path.relative_to(ROOT).as_posix(),'state_sha256':sha(path)};history.append(row);dump(out/'HISTORY_PROGRESS.json',history);dump(out/'val_PRIVATE'/(branch+'_e'+str(epoch)+'_SCALARS.json'),details);print('SOURCEVAL78',row,flush=True)
 assert fingerprint(base)==fp;del m;torch.cuda.empty_cache()
choice=max(range(1,5),key=lambda epoch:(np.mean([x['one_mIoU_percent'] for x in history if x['epoch']==epoch]),-epoch));selected={b:next(x for x in history if x['branch']==b and x['epoch']==choice) for b in ['joint','meta']};dump(out/'SELECTED_SOURCE_ONLY.json',{'shared_epoch':choice,'selection':'earliest_max_equal_branch_mean_source_VAL_postupdate_mIoU','models':selected});dump(out/'HISTORY.json',history);dump(out/'COUNTS.json',{'counts_by_branch':counts,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'target_or_FINAL_read':False,'original_model_unchanged':fingerprint(base)==fp,'matching_copy_source_trained':True,'same_source_exposure_and_outer_updates_not_equal_compute':True});seal(out,'R78_SOURCE_ONLY_JOINT_META_CONTINUATIONS_SELECTED_BEFORE_TARGET',n=640);print(read(out/'SELECTED_SOURCE_ONLY.json'));print(read(out/'COUNTS.json'))
