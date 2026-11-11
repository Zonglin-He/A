"""Finite new-module source training; no target/GT/model knowledge selection."""
from common76 import *
from model76 import *
import time,random
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint

def main():
 p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/TRAIN_CODE_LOCK.json'));assert read(H/'engineering_model_attempt001/RESULT.json')['state'].startswith('PASS_');assert read(BASE/'prepared976_attempt001/SEALED.json')['n']==976
 base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);basefp=fingerprint(base);train=source_rows('TRAIN512');val=source_rows('VAL128');assert all(r['platform']=='waymo' for r in train+val);start=time.perf_counter();out=BASE/'trained_source_attempt002';assert not out.exists();out.mkdir();(out/'states_PRIVATE').mkdir();(out/'val_PRIVATE').mkdir();history=[];selected={};calls=0;cache={}
 # Cache just source data on CPU. Target media/features/labels are not loaded.
 for r in train+val:
  z=prefix(r);g=load(casepath(r));y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];cache[r['id']]=(z,g,y)
 def item(r):
  z,g,y=cache[r['id']];return tensor_prefix(z),tensor_data(g,r),y
 for mode in MODES:
  torch.manual_seed(0);np.random.seed(0);random.seed(0);m=Fusion(base.contrastive_align_projection_image).cuda();initial=state(m);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);best=-1;bestepoch=None
  for epoch in range(1,9):
   losses=[];auxloss=[];gr=[];ii=np.random.default_rng(76000+epoch).permutation(len(train));epstart=time.perf_counter();m.train()
   for step,j in enumerate(ii):
    r=train[int(j)];z,d,y=item(r);pm,im=masks(r,d,epoch);opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
     pp,q,_=m.main(base,z,d,mode);aux,parts=m.auxiliary(z,d,mode,pm,im);target=torch.softmax(torch.tensor(y,device='cuda',dtype=torch.float64)/.1,0);ground=-(target*pp.clamp_min(1e-300).log()).sum() if np.max(y)>0 else pp.sum()*0;loss=ground+aux
    assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step();calls+=1;losses.append(float(ground.detach()));auxloss.append(float(aux.detach()));gr.append(float(gn))
    if(step+1)%128==0:print('TRAIN76',mode,epoch,step+1,'mean',round(float(np.mean(losses)),5),'sec',round(time.perf_counter()-epstart,1),flush=True)
   m.eval();vv=[];qq=[];probs=[]
   with torch.no_grad():
    for r in val:
     z,d,y=item(r);pp,_,_=m.main(base,z,d,mode);calls+=1;which=int(pp.argmax());vv.append(float(y[which]));qq.append(which);probs.append(pp.cpu().numpy())
   miou=float(np.mean(vv)*100);path=out/'states_PRIVATE'/(mode+'_e'+str(epoch).zfill(2)+'.npz');np.savez_compressed(path,**state(m));np.savez_compressed(out/'val_PRIVATE'/(mode+'_e'+str(epoch).zfill(2)+'.npz'),scores=np.stack(probs),IoU=np.array(vv),selected=np.array(qq));row={'mode':mode,'epoch':epoch,'train_n':512,'val_n':128,'ground_loss_mean':float(np.mean(losses)),'aux_loss_mean':float(np.mean(auxloss)),'grad_norm_mean':float(np.mean(gr)),'val_mIoU_percent':miou,'val_Acc25_percent':float(np.mean(np.array(vv)>.25)*100),'val_Acc50_percent':float(np.mean(np.array(vv)>.5)*100),'gate_I':float(torch.tanh(m.gate_I).detach()),'gate_P':float(torch.tanh(m.gate_P).detach()),'gate_late':float(torch.tanh(m.gate_late).detach()),'seconds':time.perf_counter()-epstart};history.append(row);dump(out/'HISTORY_PROGRESS.json',history)
   if miou>best+1e-12:best=miou;bestepoch=epoch;selected[mode]={'epoch':epoch,'source_val_mIoU_percent':miou,'state_path':path.relative_to(ROOT).as_posix(),'state_sha256':sha(path),'selection':'sourceVAL128_only;earliest_epoch_tie;trained_epochs1to8'}
   print('VAL76',row,flush=True)
  assert fingerprint(base)==basefp;del m;torch.cuda.empty_cache()
 assert fingerprint(base)==basefp;dump(out/'HISTORY.json',history);dump(out/'SELECTED_SOURCE_ONLY.json',selected);cost={'train_n':512,'val_n':128,'modes':MODES,'epochs_each':8,'new_source_grounding_train_or_val_suffix_calls':calls,'original_3EED_unchanged':True,'target_labels_or_features_read':False,'training_precision':'BF16_frozen_transformer_and_new_features;FP32_match_logits;FP32_validation','seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())};dump(out/'COUNTS.json',cost);seal(out,'R76_SOURCE_PREPARED_MODULES_SOURCE_VAL_ONLY_SELECTED_BEFORE_TARGET',n=640);print(cost)
if __name__=='__main__':main()
