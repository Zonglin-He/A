"""Ten bounded source head preparations; base representations/fusion never updated."""
from core91 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval();loadbase91(m,'R76');initial_interface=headstate91(m);initial={b:loadbase91(m,b) for b in BASES91};del base;torch.cuda.empty_cache();train=source_rows('TRAIN512');val=source_rows('VAL128');cache={};start=time.perf_counter()
for r in train+val:
 assert r['platform']=='waymo';cache[r['id']]=(load(B91/'source_features640_attempt001/samples_PRIVATE'/(r['id']+'.npz')),load(B89/'source_features640_attempt001/routes_PRIVATE'/(r['id']+'.npz')),load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'],load(B91/'source_competitors640_attempt001/samples_PRIVATE'/(r['id']+'.npz')))
if args.engineering:
 train=[r for r in train if (cache[r['id']][3]['competitor_assignment_PRIVATE']>=0).any() and len(cache[r['id']][1]['token_roles'])>1][:2];assert len(train)==2;val=train
def item(r):
 a,rt,y,ng=cache[r['id']];route=tensor_route89(rt);views=[{k:torch.as_tensor(a[str(v)+'_'+k],device='cuda') for k in ['h','u','late','local','own','text','relation_text','mask','baseline']} for v in range(3)];return views,route,torch.as_tensor(y,device='cuda',dtype=torch.float64),ng
O=B91/('engineering_source_train_attempt001' if args.engineering else 'trained_source_attempt001');(O/'states_PRIVATE').mkdir(parents=True,exist_ok=False);(O/'val_PRIVATE').mkdir();(O/'pairs_PRIVATE').mkdir();pairs={};coverage=[]
# Frozen-base hard representative per known different object, before any head learning.
with torch.no_grad():
 for b in BASES91:
  restore(m,initial[b]);freeze91(m)
  for ix,r in enumerate(train+val):
   vv,rt,y,ng=item(r);p,_,_=cached91(m,vv[0],rt,'full_disc');s=p.cpu().numpy();ass=ng['competitor_assignment_PRIVATE'];reps=[]
   for obj in sorted(set(ass[ass>=0].tolist())):
    qs=np.flatnonzero(ass==obj);reps.append(int(qs[np.argmax(s[qs])]))
   reps=sorted(reps,key=lambda j:(-s[j],j))[:4];pairs[b,r['id']]=np.array([(int(ng['positive_query_PRIVATE']),j) for j in reps],np.int64).reshape(-1,2);np.savez_compressed(O/'pairs_PRIVATE'/(b+'_'+r['id']+'.npz'),pairs_PRIVATE=pairs[b,r['id']]);coverage.append({'row':('TRAIN' if r in train else 'VAL')+str(ix+1).zfill(4),'base':b,'pair_count':len(reps),'native_top1_is_accepted_negative':int(s.argmax()) in reps,'best_target_native_rank':int((s>s[int(ng['positive_query_PRIVATE'])]).sum()+1)})
dump(O/'PAIR_COVERAGE_ROWS.json',coverage);seal(O/'pairs_PRIVATE','R91_FIXED_SOURCE_COMPETITION_PAIRS_BEFORE_HEAD_OPTIMIZATION')
hist=[];stats={};epochs=1 if args.engineering else 4
for b in BASES91:
 for head in HEADS91:
  name=b+'_'+head;restore(m,initial[b]);copy_head91(m,initial_interface);freeze91(m);ps=list(m.interface.parameters());opt=torch.optim.AdamW(ps,lr=1e-4,weight_decay=1e-4);stat={'attempted_updates':0,'effective_updates':0,'no_learning_signal_updates':0,'allocated_head_parameters':sum(p.numel() for p in ps),'active_gradient_parameters_max':0};stats[name]=stat
  for ep in range(1,epochs+1):
   for p in ps:p.requires_grad_(True)
   losses=[];gs=[];ds=[];gns=[];order=np.random.default_rng(91000+ep).permutation(len(train));order=np.tile(order,12) if args.engineering else order;begin=time.perf_counter()
   for i in order:
    r=train[int(i)];vv,rt,y,ng=item(r);pp=torch.as_tensor(pairs[b,r['id']],device='cuda');opt.zero_grad(set_to_none=True);L,parts=source_loss91(m,vv,rt,y,pp,head);assert torch.isfinite(L);stat['attempted_updates']+=1;gn=0.
    if L.requires_grad:
     L.backward();stat['active_gradient_parameters_max']=max(stat['active_gradient_parameters_max'],sum(p.numel() for p in ps if p.grad is not None and bool((p.grad!=0).any())));gn=float(torch.nn.utils.clip_grad_norm_(ps,1.));assert np.isfinite(gn)
     if gn>0:opt.step();stat['effective_updates']+=1
     else:stat['no_learning_signal_updates']+=1
    else:stat['no_learning_signal_updates']+=1
    losses.append(float(L.detach()));gs.append(float(parts['ground'].detach()));ds.append(float(parts['disc'].detach()));gns.append(gn)
   freeze91(m);scores=[];ious=[];records=[]
   with torch.no_grad():
    for r in val:
     vv,rt,y,ng=item(r);pp=[cached91(m,a,rt,head)[0] for a in vv];iy=[float(y[int(p.argmax())]) for p in pp];scores.append(torch.stack(pp).cpu().numpy());ious.append(iy);records.append({'row':'SOURCEVAL'+str(len(ious)).zfill(4),'full_IoU':iy[0],'masked_mean_IoU':float(np.mean(iy[1:]))})
   s=state(m);assert all(np.array_equal(v,initial[b][k]) for k,v in s.items() if not k.startswith('interface.'));f=O/'states_PRIVATE'/(name+'_e'+str(ep)+'.npz');np.savez_compressed(f,**s);np.savez_compressed(O/'val_PRIVATE'/(name+'_e'+str(ep)+'.npz'),scores=np.stack(scores),IoU=np.array(ious));dump(O/'val_PRIVATE'/(name+'_e'+str(ep)+'_SCALARS.json'),records);iy=np.array(ious);row={'base':b,'head':head,'name':name,'epoch':ep,'train_n':len(train),'val_n':len(val),'train_loss_mean':float(np.mean(losses)),'train_ground_mean':float(np.mean(gs)),'train_disc_mean':float(np.mean(ds)),'gradient_norm_mean':float(np.mean(gns)),'source_val_mIoU_percent':float(iy[:,0].mean()*100),'source_val_masked_mIoU_percent':float(iy[:,1:].mean()*100),'source_val_Acc25_percent':float((iy[:,0]>.25).mean()*100),'source_val_Acc50_percent':float((iy[:,0]>.5).mean()*100),'seconds':time.perf_counter()-begin,'state_path':f.relative_to(ROOT).as_posix(),'state_sha256':sha(f)};hist.append(row);dump(O/'HISTORY_PROGRESS.json',hist);print('SOURCE91',row,flush=True)
choice=max(range(1,epochs+1),key=lambda e:(np.mean([r['source_val_mIoU_percent'] for r in hist if r['epoch']==e]),-e));selected={'shared_epoch':choice,'selection':'earliest_max_mean_of_all_ten_zero_update_sourceVAL_branches','models':{b+'_'+h:next(r for r in hist if r['base']==b and r['head']==h and r['epoch']==choice) for b in BASES91 for h in HEADS91}};dump(O/'SELECTED_SOURCE_ONLY.json',selected);dump(O/'HISTORY.json',hist);dump(O/'COUNTS.json',{'source_train_n':len(train),'source_val_n':len(val),'epochs_each':epochs,'branches':stats,'source_R76_fusion_and_original_3EED_frozen':True,'source_postfusion_caching_valid_because_fusion_frozen':True,'target_inputs_or_labels_read':False,'same_attempt_budget_not_equal_effective_updates_or_capacity':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(O,'R91_SOURCE_ONLY_DISCRIMINATIVE_HEAD_PREPARATIONS_SEALED',n=len(train)+len(val));print(selected)
