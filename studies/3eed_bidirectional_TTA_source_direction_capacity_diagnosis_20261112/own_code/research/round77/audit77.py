"""Independent saved-state replay and scalar diagnosis; no source GT controls are deployed."""
from pathlib import Path
import sys,time,itertools
ROOT=Path(__file__).resolve().parents[2];H77=ROOT/'research/round77';P=ROOT/'crossplatform/round77/source128_attempt001'
sys.path.insert(0,str(ROOT/'research/round76'))
from common76 import *
from model76 import *
from target76 import objective
sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from model51 import load_source,fingerprint

def main():
 plan=read(H77/'experiment/DESIGN_LOCK.json');pin(plan);pin(read(H77/'experiment/AUDIT_CODE_LOCK.json'));ps=read(P/'SEALED.json')
 for f,h in ps['files'].items():assert sha(P/f)==h,('seal',f)
 base=load_source(read(ROOT/plan['source_plan']));torch.set_num_threads(4);torch.set_grad_enabled(False);fp=fingerprint(base);m=Fusion(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/plan['source_prepared_module']);restore(m,S);names=dict(m.tta_parameters());theta={n:S[n] for n in names};arms=ps['arm_names'];rr=source_rows('VAL128');maxp=maxloss=maxradius=0.;rows=[];events=[];modules=[];steps=[];start=time.perf_counter();capacity_scopes={};maincalls=0
 for idx,r in enumerate(rr):
  sid=r['id'];z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);out=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/'records_PRIVATE'/(sid+'.json'));grad=load(P/'gradients_PRIVATE'/(sid+'.npz'));yy=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(sid+'.npz'))['ious'];y=torch.softmax(torch.tensor(yy,dtype=torch.float64,device='cuda')/.1,0);assert len(yy)==256 and list(out['arm_names'])==arms;assert np.array_equal(out['queries'],out['scores'].argmax(1)) and np.array_equal(out['source_boxes'],z0['boxes']) and np.array_equal(out['selected_original_boxes'],z0['boxes'][out['queries']]);restore(m,S);refp,refq,_=m.main(base,z,d,'bidirectional');maincalls+=1;pm=[];im=[]
  for rep in [76001,76002]:a,b=masks(r,d,rep);pm.append(a);im.append(b)
  gl=lambda pp:float(-(y*pp.clamp_min(1e-300).log()).sum()) if rec['label_eligible'] else 0.
  for arm,st in rec['states'].items():
   restore(m,S);f=P/'states_PRIVATE'/(sid+'_'+arm+'.npz');assert sha(f)==st['sha256'];patch=load(f);parameters=dict(m.match.named_parameters()) if st['scope']=='match' else dict(m.tta_parameters());assert set(patch)==set(parameters)
   for n,v in parameters.items():v.copy_(torch.as_tensor(patch[n],device='cuda'))
   if st['scope']=='tta':assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items() if n not in patch)
   else:assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items() if not n.startswith('match.'))
   pp,q,_=m.main(base,z,d,'bidirectional');L,a,k=objective(m,base,z,d,refp,refq,pm,im,arm=='wrong_binding_aux3');maincalls+=2;ar=arms.index(arm);maxp=max(maxp,float(abs(pp.cpu().numpy()-out['scores'][ar]).max()));ep=rec['endpoints'][arm];maxloss=max(maxloss,abs(gl(pp)-ep['ground_loss']),abs(float(a)-ep['auxiliary']),abs(float(L)-ep['total_aux_keep']));assert maxp<1e-10 and maxloss<1e-9
   if arm.startswith(('aux_raw','ground_same_radius')):
    delta=np.concatenate([(patch[n]-theta[n]).ravel() for n in names]);gkey='aux_Adam_delta' if arm.startswith('aux_raw') else 'matched_ground_Adam_delta';mult=10 if arm.endswith('10') else 1;maxradius=max(maxradius,float(abs(delta-mult*grad[gkey]).max()));assert maxradius<1e-6
   deltaL2=float(np.sqrt(sum(float(np.square(patch[n].astype(float)-(S['match.'+n] if st['scope']=='match' else S[n])).sum()) for n in patch)));capacity_scopes[arm]=sum(v.size for v in patch.values())
   events.append({'row':f'S{idx+1:05d}','arm':arm,'edited_parameters':capacity_scopes[arm],'edited_scope_L2':deltaL2,**ep})
  restore(m,S);q0=int(refp.argmax());queries=dict(zip(arms,map(int,out['queries'])));values={a:float(yy[queries[a]]) for a in arms};row={'row':f'S{idx+1:05d}','id_PRIVATE':sid,'scene_PRIVATE':r['scene'],'platform':'waymo','queries':queries,'IoU':values,'oracle_IoU':float(yy.max()),'label_eligible':rec['label_eligible']};rows.append(row)
  for name in ['aux3_steps','wrong_binding_aux3_steps','ground_fit10_steps','ground_match10_steps']:
   for j,v in enumerate(rec[name]):
    if name.endswith('aux3_steps'):
     assert v['accepted'] and v['loss_after']<=v['loss_before']-max(1e-12,abs(v['loss_before'])*1e-6) or not v['accepted'] and v['before_sha256']==v['after_sha256'] and v['optimizer_before_sha256']==v['optimizer_after_sha256']
    else:assert not v['accepted'] or v['ground_after']<=v['ground_before']-max(1e-12,abs(v['ground_before'])*1e-6)
    steps.append({'row':row['row'],'arm':name[:-6],'step':j+1,'accepted':v['accepted'],'before':v.get('loss_before',v.get('ground_before')),'after':v.get('loss_after',v.get('ground_after')),'parameter_step_L2':v['parameter_step_L2']})
  # Recompute BOTH native gradients, not merely the producer's scalar claims.
  m.requires_grad_(False);parameters=[p for _,p in m.tta_parameters()]
  for pp in parameters:pp.requires_grad_(True)
  with torch.enable_grad():
   pcheck,_,_=m.main(base,z,d,'bidirectional');LG=-(y*pcheck.clamp_min(1e-300).log()).sum() if rec['label_eligible'] else pcheck.sum()*0
   gg0=torch.autograd.grad(LG,parameters,allow_unused=True);LA=objective(m,base,z,d,refp,refq,pm,im)[0];ga0=torch.autograd.grad(LA,parameters,allow_unused=True);maincalls+=2
  flatten=lambda gs:torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for g,p in zip(gs,parameters)]).detach().cpu().numpy()
  assert np.max(abs(flatten(gg0)-grad['ground_gradient']))<1e-7 and np.max(abs(flatten(ga0)-grad['aux_gradient']))<1e-7
  m.requires_grad_(False)
  # Independently recompute gradient dot products from saved flat vectors.
  ga=grad['aux_gradient'].astype(float);gg=grad['ground_gradient'].astype(float);dd=grad['aux_Adam_delta'].astype(float)
  assert abs(float(gg@dd)-rec['ground_gradient_dot_aux_Adam_delta'])<1e-7
  dot=float(gg@dd);events.append({'row':row['row'],'arm':'gradient_diagnostic','aux_gradient_L2':float(np.linalg.norm(ga)),'ground_gradient_L2':float(np.linalg.norm(gg)),'raw_gradient_cosine':None if np.linalg.norm(ga)*np.linalg.norm(gg)==0 else float(ga@gg/(np.linalg.norm(ga)*np.linalg.norm(gg))),'ground_gradient_dot_aux_Adam_delta':dot,'first_Adam_radius':rec['first_Adam_radius']})
  modules.extend([{'row':row['row'],**x} for x in rec['parameter_gradient_rows']])
  if(idx+1)%16==0:print('AUDIT77',idx+1,round(time.perf_counter()-start,1),flush=True)
 assert fingerprint(base)==fp
 summary={};refs=['source','bidirectional_frozen','aux1','aux3','aux_raw10']
 for arm in arms:
  vv=np.array([x['IoU'][arm] for x in rows]);s={'n':128,'mIoU_percent':float(vv.mean()*100),'Acc25_percent':float((vv>.25).mean()*100),'Acc50_percent':float((vv>.5).mean()*100),'query_changes_vs_frozen':sum(x['queries'][arm]!=x['queries']['bidirectional_frozen'] for x in rows)}
  for ref in refs:
   d=vv-np.array([x['IoU'][ref] for x in rows]);s.update({f'gain_vs_{ref}_pp':float(d.mean()*100),f'improved_vs_{ref}':int((d>1e-10).sum()),f'worsened_vs_{ref}':int((d< -1e-10).sum())})
  summary[arm]=s
 gradrows=[x for x in events if x['arm']=='gradient_diagnostic'];active=[x for x in gradrows if x['ground_gradient_L2']>0 and x['first_Adam_radius']>0];favorable=sum(x['ground_gradient_dot_aux_Adam_delta']<0 for x in active);gate=(summary['aux_raw10']['mIoU_percent']>summary['bidirectional_frozen']['mIoU_percent']+1e-8 and summary['aux_raw10']['mIoU_percent']>summary['aux1']['mIoU_percent']+1e-8 and summary['aux_raw10']['Acc25_percent']>=summary['bidirectional_frozen']['Acc25_percent'] and favorable>len(active)/2)
 neg=[{'row':x['row'],'arm':a,'reference':b,'delta_pp':100*(x['IoU'][a]-x['IoU'][b])} for x in rows for a,b in itertools.permutations(arms,2) if x['IoU'][a]-x['IoU'][b]<-1e-10];scenes=sorted({x['scene_PRIVATE'] for x in rows});clusters=[{'scene_alias':f'W_S{i+1:03d}','n':len([x for x in rows if x['scene_PRIVATE']==sc]),'arm':a,'mIoU_percent':float(np.mean([x['IoU'][a] for x in rows if x['scene_PRIVATE']==sc])*100)} for i,sc in enumerate(scenes) for a in arms]
 out=H77/'source_audit_score_attempt002';assert not out.exists();out.mkdir();receipt={'state':'PASS_R77_ALL128_SAVED_STATES_NATIVE_REPLAY_AND_SOURCE_LABEL_DIAGNOSTIC','n':128,'arms':len(arms),'states_replayed':len(rec['states'])*128,'max_probability_error':maxp,'max_loss_error':maxloss,'max_direction_reconstruction_error':maxradius,'source_recordings':len(scenes),'task_values':128*len(arms),'negative_comparisons':len(neg),'source_label_controlled_only':True,'target_GT_read':False,'FINAL_access':False,'main_suffix_calls':maincalls,'seconds':time.perf_counter()-start,'prediction_seal_sha256':sha(P/'SEALED.json')}
 for name,x in [('AUDIT_RESULT',receipt),('ALL_TASK_ROWS_PRIVATE',rows),('SUMMARY',summary),('ENDPOINT_SCALARS',events),('PARAMETER_GRADIENT_SCALARS',modules),('STEP_SCALARS',steps),('ALL_ORDERED_NEGATIVES',neg),('RECORDING_SCALARS',clusters),('TARGET_GATE',{'pass':gate,'active_gradients':len(active),'favorable_first_order_directions':favorable,'unfavorable_first_order_directions':sum(x['ground_gradient_dot_aux_Adam_delta']>0 for x in active),'scope':'source_exposed_controlled_diagnostic_only','new_target_run':gate})]:dump(out/(name+'.json'),x)
 seal(out,receipt['state']);print(summary);print(read(out/'TARGET_GATE.json'));print(receipt)
if __name__=='__main__':main()
