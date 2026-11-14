"""Source-only capture plus original bidirectional preparation and fixed full-disc head."""
from core97 import *
import argparse,random
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['drone','quad'],required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();pl=a.source;setsource97(pl);base=base97(pl);fp=fingerprint(base);out=B97/(pl+('_engineering' if a.engineering else ''));out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();train=rows97(pl,'TRAIN512');val=rows97(pl,'VAL128');train=train[:2] if a.engineering else train;val=val[:2] if a.engineering else val;cache={};start=time.perf_counter()
 for n,r in enumerate(train+val):
  with torch.no_grad():z,g,rec,proof=sample97(base,r)
  target=jread(r['metapath'])['ground_info'][r['object_index']]['bbox_3d'];gt=transform_gt(target,pl,r['suffix']);boxes=z['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);y=candidate_ious(boxes,gt)[0];np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'_prefix.npz'),**z);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'_geometry.npz'),**g);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'_source_labels.npz'),ious=y);cache[r['id']]=(z,g,y)
  if (n+1)%64==0:print('SOURCE97_CAPTURE',pl,n+1,len(train)+len(val),flush=True)
 torch.manual_seed(0);np.random.seed(0);random.seed(0);m=Fusion(base.contrastive_align_projection_image).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);hist=[];steps=[];epochs=1 if a.engineering else 8;(out/'fusion_states_PRIVATE').mkdir()
 def item(r):
  zz,gg,y=cache[r['id']];return tensor_prefix(zz),tensor_data(gg,r),y
 for ep in range(1,epochs+1):
  m.train();tt=time.perf_counter()
  for step,j in enumerate(np.random.default_rng(76000+ep).permutation(len(train))):
   r=train[int(j)];z,d,y=item(r);pm,im=masks(r,d,ep);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    p,_,_=m.main(base,z,d,'bidirectional');aux,_=m.auxiliary(z,d,'bidirectional',pm,im);target=torch.softmax(torch.as_tensor(y,device='cuda',dtype=torch.float64)/.1,0);ground=-(target*p.clamp_min(1e-300).log()).sum() if y.max()>0 else p.sum()*0;L=ground+aux
   assert torch.isfinite(L);L.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step();steps.append({'stage':'fusion','epoch':ep,'step':step,'loss':float(L.detach()),'gradient_norm':float(gn)})
   if(step+1)%128==0:print('SOURCE97_TRAIN',pl,ep,step+1,round(time.perf_counter()-tt,1),flush=True)
  m.eval();iy=[]
  with torch.no_grad():
   for r in val:
    z,d,y=item(r);p,_,_=m.main(base,z,d,'bidirectional');iy.append(float(y[int(p.argmax())]))
  f=out/'fusion_states_PRIVATE'/f'e{ep:02}.npz';np.savez_compressed(f,**state(m));hist.append({'epoch':ep,'source_val_mIoU_percent':float(np.mean(iy)*100),'state_path':str(f),'state_sha256':sha(f)});jdump(out/'FUSION_HISTORY.json',hist);jdump(out/'SOURCE_STEPS.json',steps);print('SOURCE97_VAL',pl,hist[-1],flush=True)
 choice=max(hist,key=lambda x:(x['source_val_mIoU_percent'],-x['epoch']));fusion=load(choice['state_path']);del m;torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval();missing,extra=m.load_state_dict({k:torch.as_tensor(v,device='cuda') for k,v in fusion.items()},strict=False);assert not extra and all(k.startswith('interface.') for k in missing);m.pool_mode='strict';m.variant='full';m.use_interface=True;initial=state(m);featurecache={};pairs={};coverage=[]
 for n,r in enumerate(train+val):
  z0,g,y=cache[r['id']];z,d,_=item(r);rt,info=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);m.route=route;pm,im=maskpair79(r,d)
  with torch.no_grad():vv=[native_feature91(m,base,z,d,a,b) for a,b in [(None,None)]+list(zip(pm,im))];p,_,_=cached91(m,vv[0],route,'full_disc')
  pos,ass,c=competitor97(r,z0,rt,y);reps=[];scores=p.cpu().numpy()
  for obj in sorted(set(ass[ass>=0].tolist())):
   qs=np.flatnonzero(ass==obj);reps.append(int(qs[np.argmax(scores[qs])]))
  reps=sorted(reps,key=lambda j:(-scores[j],j))[:4];pairs[r['id']]=np.array([(pos,j) for j in reps],np.int64).reshape(-1,2);featurecache[r['id']]=([{k:v.detach().cpu().numpy() for k,v in view.items()} for view in vv],rt);coverage.append({'id':r['id'],'split':'train' if n<len(train) else 'val','pairs':len(reps),**c});np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'_route.npz'),**rt)
  if(n+1)%128==0:print('SOURCE97_HEAD_CACHE',pl,n+1,flush=True)
 jdump(out/'PAIR_COVERAGE_PRIVATE.json',coverage);freeze91(m);ps=list(m.interface.parameters());opt=torch.optim.AdamW(ps,lr=1e-4,weight_decay=1e-4);hh=[];(out/'head_states_PRIVATE').mkdir()
 def headitem(r):
  vv,rt=featurecache[r['id']];return [{k:torch.as_tensor(v,device='cuda') for k,v in view.items()} for view in vv],tensor_route89(rt),torch.as_tensor(cache[r['id']][2],device='cuda',dtype=torch.float64),torch.as_tensor(pairs[r['id']],device='cuda')
 for ep in range(1,(1 if a.engineering else 4)+1):
  for p in ps:p.requires_grad_(True)
  for step,j in enumerate(np.random.default_rng(91000+ep).permutation(len(train))):
   r=train[int(j)];vv,rt,y,pp=headitem(r);opt.zero_grad(set_to_none=True);L,_=source_loss91(m,vv,rt,y,pp,'full_disc');assert torch.isfinite(L);gn=0.
   if L.requires_grad:L.backward();gn=float(torch.nn.utils.clip_grad_norm_(ps,1));assert np.isfinite(gn);opt.step()
   steps.append({'stage':'head','epoch':ep,'step':step,'loss':float(L.detach()),'gradient_norm':gn})
  freeze91(m);iy=[]
  with torch.no_grad():
   for r in val:
    vv,rt,y,_=headitem(r);p,_,_=cached91(m,vv[0],rt,'full_disc');iy.append(float(y[int(p.argmax())]))
  S=state(m);assert all(np.array_equal(v,initial[k]) for k,v in S.items() if not k.startswith('interface.'));f=out/'head_states_PRIVATE'/f'e{ep:02}.npz';np.savez_compressed(f,**S);hh.append({'epoch':ep,'source_val_mIoU_percent':float(np.mean(iy)*100),'state_path':str(f),'state_sha256':sha(f)});jdump(out/'HEAD_HISTORY.json',hh);jdump(out/'SOURCE_STEPS.json',steps);print('SOURCE97_HEAD_VAL',pl,hh[-1],flush=True)
 selected=max(hh,key=lambda x:(x['source_val_mIoU_percent'],-x['epoch']));np.savez_compressed(out/'SELECTED_SOURCE_ONLY.npz',**load(selected['state_path']));assert fingerprint(base)==fp;jdump(out/'SOURCE_COMPLETE.json',{'source':pl,'source100_sha256':source97(pl)['checkpoint_sha256'],'original_source100_unchanged':True,'fusion_selected':choice,'head_selected':selected,'selection':'each fixed single branch earliest best sourceVAL epoch, no target metrics','source_train_n':len(train),'source_val_n':len(val),'target_data_read':False,'seconds':time.perf_counter()-start,'updates':len(steps),'tta_parameters':sum(p.numel() for n,p in m.tta_parameters()),'selected_sha256':sha(out/'SELECTED_SOURCE_ONLY.npz')});print('SOURCE97_COMPLETE',pl,round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':main()
