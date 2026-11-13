"""Same sealed R94 evidence, actual nonlinear weight updates, unseen mask C used only after fitting."""
from core95 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();rows=read(H95/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[r for pl in ['drone','quad'] for r in [x for x in rows if x['platform']==pl][:4]]
O=B95/('engineering_prediction8_attempt001' if args.engineering else 'prediction336_attempt001')
for n in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE','C_features_PRIVATE']:(O/n).mkdir(parents=True,exist_ok=False)
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=source92();restore(m,S);Shead=patch94(m);start=time.perf_counter();status=[];coverage=[];counts={'n':len(rows),'arms':len(ARMS95),'new_C_calls':0,'head_trial_calls':0,'TTA_parameters':18496,'new_G_calls':0,'source_training':False,'target_GT_read':False,'FINAL_access':False}
for nr,r in enumerate(rows):
 restore(m,S);f=feature_to92(load(P94/'features_PRIVATE'/(r['id']+'.npz')));ev=load(P94/'evidence_PRIVATE'/(r['id']+'.npz'));r0=torch.as_tensor(ev['r0_PRIVATE'],device='cuda');target=torch.as_tensor(ev['binding_target_PRIVATE'],device='cuda').log();p0=torch.as_tensor(ev['correct_base_PRIVATE'][0]+ev['r0_PRIVATE'],device='cuda').softmax(0);mix=((p0+target.exp())/2).log();shuf=torch.as_tensor(ev['shuffle_target_PRIVATE'],device='cuda').log();results={};records={};last={}
 for name in ['lm','lm_mixture','lm_shuffle','lm_capped']:
  final,states,trials,rec=lm95(m,f,r0,mix if name=='lm_mixture' else shuf if name=='lm_shuffle' else target,Shead,name);rec['trials']=trials;records[name]=rec;last[name]=final;counts['head_trial_calls']+=len(trials)
  for k,v in states.items():np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+name+'_'+k+'.npz'),**v)
  np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+name+'_final.npz'),**final)
 # The heldout view is not opened by optimizer or gain ratio.
 restore(m,S);z0,z,d,rt,route,pm,im,ws=data91(r);hp,hi=masks(r,d,76003)
 with torch.no_grad():
  fc=feature91(m,base,z,d,route,S,hp,hi);counts['new_C_calls']+=1;np.savez_compressed(O/'C_features_PRIVATE'/(r['id']+'.npz'),**{k:v.cpu().numpy() for k,v in fc.items()});put94(m,Shead);pC=read94(m,fc,r0);delta=torch.as_tensor(ev['delta_PRIVATE'],device='cuda');qC=(pC.log()+delta).softmax(0);qF=target.exp();q4=torch.stack([(torch.as_tensor(b,device='cuda')+r0+delta).softmax(0) for b in ev['correct_base_PRIVATE']]+[qC]).mean(0);mixC=(pC+qC)/2;shC=(pC.log()+delta[torch.as_tensor(ev['query_control_PRIVATE'],device='cuda')]).softmax(0);results.update(frozen_C=pC.cpu().numpy(),direct_C=qC.cpu().numpy(),direct_FC=((qF+qC)/2).cpu().numpy(),direct_four=q4.cpu().numpy(),mixture_FC=((mix.exp()+mixC)/2).cpu().numpy(),shuffle_FC=((shuf.exp()+shC)/2).cpu().numpy())
  for name,st in last.items():put94(m,st);pF=read94(m,f,r0);pC1=read94(m,fc,r0);results[name+'_F']=pF.cpu().numpy();results[name+'_C']=pC1.cpu().numpy();results[name+'_FC']=((pF+pC1)/2).cpu().numpy()
 old=load(P94/'samples_PRIVATE'/(r['id']+'.npz'));oo=dict(zip(old['arm_names'],old['scores']));og=load(B94/'solver_repair336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));gg=dict(zip(og['arm_names'],og['scores']));refs={'R94_direct':'binding_direct','R94_mixture':'binding_optimal_mixture','R94_Adam':'binding_bt1'}
 for k in REFS95:results[k]=gg['gn_bt'] if k=='R94_GN' else oo[refs.get(k,k)]
 ss=np.stack([results[k] for k in ARMS95]);qq=ss.argmax(1);np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS95),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);dump(O/'records_PRIVATE'/(r['id']+'.json'),records);oldst=next(x['arm_statuses'] for x in read(P94/'READOUT_STATUS_PRIVATE.json') if x['id']==r['id']);st={}
 for k in ARMS95:
  if k in REFS95:st[k]={'unknown':False,'fallback':False,'no_parameter_update':False} if k=='R94_GN' else oldst[refs.get(k,k)]
  elif k in DIRECT95:st[k]={'unknown':False,'fallback':False,'no_parameter_update':True}
  else:st[k]={'unknown':False,'fallback':records[k.rsplit('_',1)[0]]['accepted_steps']==0,'no_parameter_update':records[k.rsplit('_',1)[0]]['parameter_L2']==0}
 status.append({'id':r['id'],'arm_statuses':st});coverage.append({'row':'DEV'+str(nr+1).zfill(4),'platform':r['platform'],'relative_fit_error':records['lm']['relative_error'],'source_teacher_top1_agree':int(p0.argmax())==int(target.argmax()),'LM_F_matches_teacher':int(results['lm_F'].argmax())==int(target.argmax()),'C_not_used_for_update_acceptance_or_stopping':True})
 if (nr+1)%16==0 or args.engineering:print('RUN95',nr+1,len(rows),flush=True)
restore(m,S);assert fingerprint(base)==fp;counts.update(seconds=time.perf_counter()-start,peak_GPU_bytes=int(torch.cuda.max_memory_allocated()));dump(O/'COUNTS.json',counts);dump(O/'READOUT_STATUS_PRIVATE.json',status);dump(O/'EXECUTION_ROWS.json',coverage);seal(O,'R95_ADAPTIVE_LM_AND_UNFITTED_C_PREDICTIONS_SEALED',n=len(rows),arm_names=ARMS95);print(counts)
