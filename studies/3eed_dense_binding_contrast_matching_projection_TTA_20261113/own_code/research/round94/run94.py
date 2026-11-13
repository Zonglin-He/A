"""Actual current-model binding contrasts and final-projection TTA, sealed before GT."""
from core94 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();rows=read(H94/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for pl in ['drone','quad'] for r in [x for x in rows if x['platform']==pl][:4]]
O=B94/('engineering_prediction336_attempt001' if args.engineering else 'prediction336_attempt001')
for n in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE','evidence_PRIVATE']:(O/n).mkdir(parents=True,exist_ok=False)
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();restore(m,S);Shead=patch94(m);counts={'n':len(rows),'arms':len(ARMS94),'new_counterfactual_native_calls':0,'native_zero_parity_calls':0,'head_trial_calls':0,'TTA_parameters':sum(p.numel() for p in head94(m)),'source_training':False,'target_GT_read':False,'FINAL_access':False};start=time.perf_counter();statuses=[];coverage=[]
for nr,r in enumerate(rows):
 restore(m,S);z0,z,d,rt,route,pm,im,ws=data91(r);f=[];parts=[]
 with torch.no_grad():
  for v in range(3):
   a=load(P91/'features_PRIVATE'/(r['id']+'_view'+str(v)+'.npz'));ff=feature_to92(a);f.append(ff);parts.append(parts92(m,ff,route,S))
  correct=torch.stack([x[0] for x in parts]);r0=parts[0][1];wrong=[];perms=[]
  for c in range(2):
   perm=permutation94(r,d,c);perms.append(perm);dd=bound94(d,perm);assert torch.equal(dd['p2i'].ge(0).any(1),d['p2i'].ge(0).any(1));ww=[]
   for a,b in [(None,None)]+list(zip(pm,im)):
    ff=feature91(m,base,z,dd,route,S,a,b);bb,_,_,_=parts92(m,ff,route,S);ww.append(bb);counts['new_counterfactual_native_calls']+=1
   wrong.append(torch.stack(ww))
  wrong=torch.stack(wrong);targets,p0,delta=targets94(correct,wrong,r0,r);old=load(P93/'samples_PRIVATE'/(r['id']+'.npz'));pp=dict(zip(old['arm_names'],old['scores']));res={k:pp['competition_projected_bt' if k=='R93_bt' else 'competition_projected_raw' if k=='R93_raw' else k] for k in REFS94};assert np.max(abs(p0.cpu().numpy()-res['R91_frozen']))<1e-6
  res.update(binding_direct=targets['binding'].cpu().numpy(),binding_optimal_mixture=((p0+targets['binding'])/2).cpu().numpy(),reverse_direct=targets['reverse'].cpu().numpy(),shuffle_direct=targets['shuffle'].cpu().numpy(),viewmean_direct=targets['viewmean'].cpu().numpy(),viewmean_optimal_mixture=((p0+targets['viewmean'])/2).cpu().numpy())
 np.savez_compressed(O/'evidence_PRIVATE'/(r['id']+'.npz'),correct_base_PRIVATE=correct.cpu().numpy(),wrong_base_PRIVATE=wrong.cpu().numpy(),r0_PRIVATE=r0.cpu().numpy(),delta_PRIVATE=delta.cpu().numpy(),permutations_PRIVATE=np.stack(perms),query_control_PRIVATE=qperm94(r),**{k+'_target_PRIVATE':v.cpu().numpy() for k,v in targets.items()});np.savez_compressed(O/'features_PRIVATE'/(r['id']+'.npz'),**{k:v.cpu().numpy() for k,v in f[0].items()});rec={}
 for name in ['binding','shuffle','reverse','viewmean']:
  probs,recs,states=optimize94(m,f[0],r0,targets[name].detach(),p0.detach(),3 if name=='binding' else 1,Shead);rec[name]=recs
  for k,v in states.items():np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+name+'_'+k+'.npz'),**v)
  counts['head_trial_calls']+=sum(len(x['trials']) for x in recs);res[name+'_bt1']=probs['bt1']
  if name=='binding':res['binding_bt3']=probs['bt3'];res['binding_raw1']=probs['raw1']
 dump(O/'records_PRIVATE'/(r['id']+'.json'),rec);ss=np.stack([res[k] for k in ARMS94]);qq=ss.argmax(1);assert np.isfinite(ss).all();np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS94),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);st={}
 oldstatus=next(x['arm_statuses'] for x in read(P93/'READOUT_STATUS_PRIVATE.json') if x['id']==r['id'])
 for arm in ARMS94:
  if arm in REFS94:st[arm]=oldstatus['competition_projected_bt' if arm=='R93_bt' else 'competition_projected_raw' if arm=='R93_raw' else arm]
  else:st[arm]={'unknown':not bool(d['visible'].any()),'fallback':False,'no_parameter_update':arm in DIRECT94 or (rec[arm.split('_')[0]][2 if arm=='binding_bt3' else 0]['parameter_L2']==0)}
 statuses.append({'id':r['id'],'arm_statuses':st});coverage.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'candidate_count':256,'visible_points':int(d['visible'].sum()),'correction_RMS':float(delta.square().mean().sqrt()),'direct_changes_top1':int(targets['binding'].argmax())!=int(p0.argmax()),'direct_target_KL':float((targets['binding']*(targets['binding'].log()-p0.log())).sum()),'no_source_rank_or_same_sign_gate':True})
 if (nr+1)%16==0 or args.engineering:print('RUN94',nr+1,len(rows),flush=True)
restore(m,S);assert fingerprint(base)==fp;counts.update(seconds=time.perf_counter()-start,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()));dump(O/'COUNTS.json',counts);dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'EVIDENCE_COVERAGE_ROWS.json',coverage);seal(O,'R94_DENSE_BINDING_CONTRAST_AND_ACTUAL_FINAL_PROJECTION_UPDATES_SEALED',n=len(rows),arm_names=ARMS94);print(counts)
