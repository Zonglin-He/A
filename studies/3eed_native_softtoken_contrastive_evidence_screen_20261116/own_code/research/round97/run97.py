"""Fixed R91 source-specific six-way TTA; no task labels, no tuning, frozen A256."""
from core97 import *
import argparse
ARMS97=['original_source','prepared_frozen','direct_mean3','direct_feature3','physical_raw','physical_bt','wrong_physical_raw','wrong_physical_bt']
def target_capture97(base,r,src):
 pts,ix,_=sample_points(r)
 if r['platform']=='waymo':
  from utils.transform_waymo import transform_to_front_view
  pts,_=transform_to_front_view(pts,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pts=np.ascontiguousarray(pts,np.float32)
 z0,rec=capture(base,pts,ix,r['caption']);rec.update(source_checkpoint_sha256=source97(src)['checkpoint_sha256'],source_platform=src);key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();cache=B97/'shared_physical_geometry_PRIVATE';cache.mkdir(exist_ok=True);f=cache/(r['id']+'_'+key[:20]+'.npz');pf=f.with_suffix('.json')
 if f.exists():
  proof=jread(pf);assert proof['geometry_seed_and_sample_sha256']==key and proof['image_sha256']==sha(r['imagepath']) and proof['meta_sha256']==sha(r['metapath']);g=load(f);g['query_anchor_indices']=np.asarray(z0['query_points_sample_inds'],np.int64)
 else:
  g,proof=geometry(r,z0);proof.update(geometry_seed_and_sample_sha256=key,meta_sha256=sha(r['metapath']));np.savez_compressed(f,**g);jdump(pf,proof)
 return z0,g,rec,proof
def case97(base,m,S,r,z0,g,rt):
 restore(m,S);freeze91(m);z=tensor_prefix(z0);d=tensor_data(g,r);route=tensor_route89(rt);m.route=route;m.use_interface=True;m.variant='full';pm,im=maskpair79(r,d);ws=weight_sets({'id':'UNUSED_UNIFORM'},d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform'];result={};records={};start=time.perf_counter()
 with torch.no_grad():
  original=native_scores(torch.as_tensor(z0['query'],device='cuda'),torch.as_tensor(z0['tokens'],device='cuda'),z['token_mask'].bool()).double().softmax(0);result['original_source']=original.cpu().numpy();vv=[feature91(m,base,z,d,route,S,a,b) for a,b in [(None,None)]+list(zip(pm,im))];p,mean,feat,e=mean_reads91(m,vv,route,S,'R76','full_disc');result.update(prepared_frozen=p.cpu().numpy(),direct_mean3=mean.cpu().numpy(),direct_feature3=feat.cpu().numpy());rp,rq=p.detach(),e['projected_query'].detach()
 for wrong in [False,True]:
  grad,raw,rec=physical_direction91(m,base,z,d,S,pm,im,ws,wrong);direction=-(1/(grad.norm()+1e-6)).clamp_max(1)*grad;before=rec['objective_before'];kind='wrong_physical' if wrong else 'physical';trials=[];accepted=False
  for k,alpha in enumerate(GRID91):
   delta=raw if k==0 else alpha*direction;freeze91(m);theta_step91(m,S,delta)
   with torch.no_grad():
    f=feature91(m,base,z,d,route,S);p,e,_=scorer91(m,f,route,S,'R76','full_disc');aux=float(uniform_aux91(m,z,d,pm,im,ws,wrong));keep=float(keep91(p,e['projected_query'],rp,rq));total=aux+keep;bound=before+1e-4*float(grad.double()@delta.double());ok=np.isfinite(total) and total<=bound+1e-10
   trial={'alpha':alpha,'physical':aux,'keep':keep,'objective':total,'bound':bound,'accepted':bool(ok)};trials.append(trial)
   if k==0:result[kind+'_raw']=p.cpu().numpy();records[kind+'_raw']={**rec,'alpha':alpha,'accepted_by_loss_rule':False,'actual_parameter_L2':float(delta.norm()),'update_applied':bool(delta.norm()>0),'trials':[trial]}
   if ok:
    result[kind+'_bt']=p.cpu().numpy();records[kind+'_bt']={**rec,'alpha':alpha,'accepted_by_loss_rule':True,'actual_parameter_L2':float(delta.norm()),'update_applied':bool(delta.norm()>0),'trials':trials};accepted=True;break
  if not accepted:result[kind+'_bt']=result['prepared_frozen'].copy();records[kind+'_bt']={**rec,'alpha':0.,'actual_parameter_L2':0.,'accepted_by_loss_rule':False,'update_applied':False,'trials':trials}
 restore(m,S);freeze91(m);scores=np.stack([result[a] for a in ARMS97]);assert scores.shape==(8,256) and np.isfinite(scores).all() and np.allclose(scores.sum(1),1,atol=1e-10);return scores,records,{'seconds':time.perf_counter()-start,'unknown_roles':len(rt['token_roles'])==0,'visible_seeds':int(g['visible'].sum()),'tta_parameters':sum(p.numel() for n,p in m.tta_parameters())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['waymo','drone','quad'],required=True);ap.add_argument('--target',choices=['waymo','drone','quad'],required=True);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();src,tgt=a.source,a.target;assert src!=tgt;plan=jread(H97/'experiment'/('ENGINEERING_CODE_LOCK.json' if a.engineering else 'TARGET_CODE_LOCK.json'))
 for p,s in plan['code_sha256'].items():assert sha(R97/p)==s,p
 if src!='waymo':assert jread(B97/src/'SOURCE_COMPLETE.json')['original_source100_unchanged']
 setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);ss=hashlib.sha256(b''.join(n.encode()+v.tobytes() for n,v in S.items())).hexdigest();rows=rows97(tgt,'target');rows=[rows[i] for i in [0,len(rows)//2,len(rows)-1]] if a.engineering else rows;out=B97/('engineering_'+src+'_to_'+tgt if a.engineering else src+'_to_'+tgt);out.mkdir(exist_ok=True);(out/'samples_PRIVATE').mkdir(exist_ok=True);(out/'records_PRIVATE').mkdir(exist_ok=True);start=time.perf_counter();nnew=0
 for n,r in enumerate(rows):
  f=out/'samples_PRIVATE'/(r['id']+'.npz');rf=out/'records_PRIVATE'/(r['id']+'.json')
  if f.exists():
   assert rf.exists();v=load(f);assert str(v['source_state_sha256'])==ss and int(v['anno_index'])==r['anno_index'];continue
  with torch.no_grad():z0,g,rec,proof=target_capture97(base,r,src)
  z,d=tensor_prefix(z0),tensor_data(g,r);rt,info=role97(base,m,r,z0,g,z,d);scores,records,cost=case97(base,m,S,r,z0,g,rt);assert cost['tta_parameters']==102498;np.savez_compressed(f,source_boxes=z0['boxes'],arm_names=np.array(ARMS97),scores=scores,queries=scores.argmax(1),point_indices=z0['point_indices'],source_state_sha256=np.array(ss),anno_index=np.array(r['anno_index']));jdump(rf,{'readout_status':cost,'source':src,'target':tgt,'input_receipt':rec,'geometry':proof,'route':info,'updates':records});nnew+=1
  if (n+1)%32==0 or a.engineering:print('RUN97',src,tgt,n+1,len(rows),'seconds',round(time.perf_counter()-start,1),'last_seconds',round(cost['seconds'],3),flush=True)
 assert fingerprint(base)==fp;final=state(m);assert all(np.array_equal(v,final[k]) for k,v in S.items());files=list((out/'samples_PRIVATE').glob('*.npz'))+list((out/'records_PRIVATE').glob('*.json'));assert len(files)==2*len(rows);jdump(out/'SEALED.json',{'state':'R97_FIXED_R91_DIRECTION_SEALED_BEFORE_TASK_SCORES','source':src,'target':tgt,'n':len(rows),'new_predictions':nnew,'arms':ARMS97,'primary':'physical_bt','source_state_sha256':ss,'source100_checkpoint_sha256':source97(src)['checkpoint_sha256'],'files':{str(p.relative_to(out)):sha(p) for p in files},'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'physical_cache':'only exact same source-independent geometry seed/sample hashes; own anchor indices rebound to current model; native audit recomputes from raw','target_task_GT_read':False,'FINAL_access':False,'source_original_model_unchanged':True,'reset_after_every_input':True});print('RUN97_COMPLETE',src,tgt,len(rows),round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':main()
