"""One all-DEV336 fixed-source experiment, models and task labels kept separate."""
from target90 import *
from views90 import frozen_reads90
import argparse
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();lock=read(H90/'experiment/TARGET_LOCK.json');pin(lock);assert read(H90/'independent_source_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');sel=read(B90/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');S={b:load(ROOT/sel['models'][b]['state_path']) for b in BRANCHES90};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval();load_R76_90(m);initial=state(m);rows=read(H90/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
O=B90/('engineering_target_attempt001' if args.engineering else 'prediction336_attempt001')
for k in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE','conditional_PRIVATE']:(O/k).mkdir(parents=True,exist_ok=False)
legacy_status={r['id']:r['arm_statuses'] for r in read(B89/'prediction336_attempt001/READOUT_STATUS_PRIVATE.json')};abc_status={r['id']:r['arm_statuses']['frozen_ABC'] for r in read(ROOT/'crossplatform/round88/prediction336_with_control_attempt001/READOUT_STATUS_PRIVATE.json')};status=[];start=time.perf_counter();cost={'accepted_updates':0,'trial_calls':0};parametercounts={}
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d,rt,route,pm,im,ws=data90(r);m.route=route;old=load(B89/'prediction336_attempt001/samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));abc=load(ROOT/'crossplatform/round88/prediction336_with_control_attempt001/samples_PRIVATE'/(sid+'.npz'));ap0=dict(zip(abc['arm_names'],abc['scores']));mapping={'source':'source','frozen_bi':'frozen_bi','R59':'R59','R89_candidate_frozen':'candidate_frozen','R89_candidate_tta1':'candidate_tta1','R89_continue_tta1':'continue_tta1'};result={k:op[v].copy() for k,v in mapping.items()};result['frozen_ABC']=ap0['frozen_ABC'].copy();st={k:legacy_status[sid][v].copy() for k,v in mapping.items()};st['frozen_ABC']=abc_status[sid].copy();record={};unknown=not bool(rt['supports'].any())
 for name,b,variant,pool,kind in specs90():
  restore(m,S[b]);configure90(m,b,variant);m.pool_mode=pool;m.requires_grad_(False)
  with torch.no_grad():
   if name in BRANCHES90:p0,mean,feature,e0=frozen_reads90(m,base,z,d,pm,im)
   else:p0,q0,e0=m.main(base,z,d,'bidirectional')
  if name in BRANCHES90+VARIANTS90:
   result[name+'_frozen']=p0.cpu().numpy();st[name+'_frozen']={'unknown':unknown if b.startswith('candidate') else False,'fallback':unknown if b.startswith('candidate') else False,'no_parameter_update':True}
  if name in BRANCHES90:
   result[name+'_mean3']=mean.cpu().numpy();result[name+'_feature3']=feature.cpu().numpy();st[name+'_mean3']=st[name+'_frozen'].copy();st[name+'_feature3']=st[name+'_frozen'].copy()
  if name=='candidate_joint':np.savez_compressed(O/'conditional_PRIVATE'/(sid+'_initial.npz'),conditional=e0['conditional'].cpu().numpy(),conditional_valid=e0['conditional_valid'].cpu().numpy(),subject_weights=e0['subject_weights'].cpu().numpy())
  ref,out,patch,tr,refs,_=update90(m,base,z,d,pm,im,ws,3 if name=='candidate_joint' else 1,kind,shuffle90(r) if kind=='shuffle' else None);assert np.array_equal(ref,p0.cpu().numpy());record[name]=tr;cost['accepted_updates']+=sum(t['accepted'] for t in tr);cost['trial_calls']+=sum(len(t['trials']) for t in tr);parametercounts[name]=sum(p.numel() for n,p in m.tta_parameters())
  for step,pp in out.items():
   arm=arm90(name,step);result[arm]=pp;np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch[step]);t=tr[step-1];st[arm]={'unknown':unknown if b.startswith('candidate') else False,'fallback':unknown or t['parameter_L2']==0 if b.startswith('candidate') else t['parameter_L2']==0,'no_parameter_update':t['parameter_L2']==0}
 restore(m,initial);m.requires_grad_(False);m.use_interface=False;m.route=None;m.pool_mode='strict'
 with torch.no_grad():assert np.array_equal(m.main(base,z,d,'bidirectional')[0].cpu().numpy(),op['frozen_bi'])
 scores=np.stack([result[k] for k in ARMS90]);q=scores.argmax(-1);assert np.isfinite(scores).all();np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),scores=scores,arm_names=np.array(ARMS90),queries=q,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][q],mask_point=torch.stack(pm).cpu().numpy(),mask_image=torch.stack(im).cpu().numpy());dump(O/'records_PRIVATE'/(sid+'.json'),record);status.append({'id':sid,'arm_statuses':st})
 if (nr+1)%16==0 or args.engineering:print('TARGET90',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp;dump(O/'READOUT_STATUS_PRIVATE.json',status);dump(O/'COUNTS.json',{'n':len(rows),'arms':len(ARMS90),**cost,'TTA_parameter_counts':parametercounts,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'original_3EED_changed':False,'new_binding_router_updated':True,'final_discriminator_out_MLP_updated':False,'task_GT_read':False});seal(O,'R90_ALL_SOURCE_PREPARED_TARGET_ENDPOINTS_SEALED',n=len(rows),arm_names=ARMS90);print(read(O/'COUNTS.json'))
