"""All source-selected zero-update/three-reader/TTA and mechanism controls."""
from target_core89 import *
import time,argparse
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();plan=read(H89/'experiment/TARGET_LOCK.json');pin(plan);assert read(H89/'independent_source_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');selected=read(B89/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');states={b:load(ROOT/selected['models'][b]['state_path']) for b in BRANCHES89};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));torch.manual_seed(0);m=Fusion89(base.contrastive_align_projection_image).cuda().eval();load_R76(m);initial=state(m);fp=fingerprint(base);rows=read(H89/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
O=B89/('engineering_target_attempt001' if args.engineering else 'prediction336_attempt001')
for k in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE','features_PRIVATE']:(O/k).mkdir(parents=True,exist_ok=False)
oldstatuses={r['id']:r['arm_statuses'] for r in read(ROOT/'crossplatform/round88/prediction336_with_control_attempt001/READOUT_STATUS_PRIVATE.json')};counts={'main_calls':0,'objective_calls':0,'accepted_updates':0,'trial_calls':0};status=[];start=time.perf_counter()
def main(*a,**kw):counts['main_calls']+=1;return m.main(*a,**kw)
def obj(*a,**kw):counts['objective_calls']+=1;counts['main_calls']+=3;return objective_target89(*a,**kw)
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d,a,pm,im,ws=item89(r);m.route=tensor_route89(a);legacy=load(ROOT/'crossplatform/round88/prediction336_with_control_attempt001/samples_PRIVATE'/(sid+'.npz'));old=dict(zip(legacy['arm_names'],legacy['scores']));mapping=dict(zip(BASE89,['source','frozen_bi','R59','R87_hard','full_direct','full_tta1','full_frozen3']));result={x:old[y].copy() for x,y in mapping.items()};st={x:dict(oldstatuses[sid][y]) for x,y in mapping.items()};rec={};unknown=not bool(a['supports'].any())
 specs=[(b,b,'full' if b!='query' else 'query_only','full') for b in BRANCHES89]+[(v,'candidate',v,'full') for v in VARIANTS89]+[('candidate_physical','candidate','full','physical'),('candidate_no_correspondence','candidate','full','no_correspondence')]
 for name,branch,variant,kind in specs:
  restore(m,states[branch]);m.requires_grad_(False);m.use_interface=branch!='continue';m.variant=variant
  with torch.no_grad():refp,refq,e=main(base,z,d,'bidirectional');refs=e['local_reads'].detach()
  if name in BRANCHES89+VARIANTS89:
   result[name+'_frozen']=refp.cpu().numpy();st[name+'_frozen']={'unknown':unknown if branch=='candidate' else False,'fallback':unknown if branch=='candidate' else False,'no_parameter_update':True}
  if name in BRANCHES89:
   with torch.no_grad():views=[main(base,z,d,'bidirectional',point_mask=x,image_mask=y)[0] for x,y in zip(pm,im)];result[name+'_mean3']=torch.stack([refp]+views).mean(0).cpu().numpy()
   st[name+'_mean3']=dict(st[name+'_frozen'])
  if name=='candidate':np.savez_compressed(O/'features_PRIVATE'/(sid+'_initial.npz'),**features89(e))
  enable79(m);trace=[];steps=3 if name=='candidate' else 1
  for step in range(1,steps+1):
   L,p,parts=obj(m,base,z,d,refp,refq,refs,pm,im,ws,kind);ps=[p for _,p in m.tta_parameters()];raw=flat89(torch.autograd.grad(L,ps,allow_unused=True),ps);g=raw*min(1.,1/(float(raw.norm())+1e-6));oldparams=[p.detach().clone() for p in ps];dot=float(raw@g);accepted=False;alpha=0.;trials=[]
   if float(g.norm())>1e-12:
    for scale in [1/2**j for j in range(9)]:
     apply89(m,oldparams,-.1*scale*g)
     with torch.no_grad():lv,pv,pvparts=obj(m,base,z,d,refp,refq,refs,pm,im,ws,kind)
     bound=float(L)-1e-4*.1*scale*max(0,dot);ok=float(lv)<=bound+1e-10;trials.append({'scale':scale,'actual_total':float(lv),'bound':bound,'accepted':ok});counts['trial_calls']+=1
     if ok:accepted=True;alpha=scale;counts['accepted_updates']+=1;break
   if not accepted:apply89(m,oldparams,torch.zeros_like(g))
   with torch.no_grad():after,pv,aparts=obj(m,base,z,d,refp,refq,refs,pm,im,ws,kind)
   assert not accepted or float(after)<=float(L)+1e-10
   delta=float(torch.sqrt(sum((p-torch.as_tensor(states[branch][n],device='cuda')).square().sum() for n,p in m.tta_parameters())));trace.append({'step':step,'accepted':accepted,'scale':alpha,'before':float(L),'after':float(after),'parts_before':{k:float(v) for k,v in parts.items()},'parts_after':{k:float(v) for k,v in aparts.items()},'gradient_norm_raw':float(raw.norm()),'gradient_norm_clipped':float(g.norm()),'directional_derivative':dot,'parameter_L2':delta,'trials':trials})
   if step in [1,3]:
    arm=name+str(step) if name in ['candidate_physical','candidate_no_correspondence'] else name+'_tta'+str(step);result[arm]=pv.cpu().numpy();np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch89(m));st[arm]={'unknown':unknown if branch=='candidate' else False,'fallback':unknown or delta==0 if branch=='candidate' else delta==0,'no_parameter_update':delta==0}
    if name=='candidate':
     with torch.no_grad():check,_,extra=main(base,z,d,'bidirectional');assert float(abs(check-pv).max())<1e-12
     np.savez_compressed(O/'features_PRIVATE'/(sid+'_step'+str(step)+'.npz'),**features89(extra))
  rec[name]=trace
 restore(m,initial);m.route=None;m.use_interface=False;m.requires_grad_(False)
 with torch.no_grad():assert np.array_equal(main(base,z,d,'bidirectional')[0].cpu().numpy(),old['frozen_bi'])
 scores=np.stack([result[k] for k in ARMS89]);q=scores.argmax(-1);assert np.isfinite(scores).all();np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),scores=scores,arm_names=np.array(ARMS89),queries=q,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][q],mask_point=torch.stack(pm).cpu().numpy(),mask_image=torch.stack(im).cpu().numpy());dump(O/'records_PRIVATE'/(sid+'.json'),rec);status.append({'id':sid,'arm_statuses':st});print('RUN_TARGET89',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp and all(np.array_equal(v.detach().cpu().numpy(),initial[k]) for k,v in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',status);dump(O/'COUNTS.json',{'n':len(rows),'arms':len(ARMS89),**counts,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'TTA_parameters':sum(p.numel() for _,p in m.tta_parameters()),'prepared_heads_updated_at_test':False,'target_GT_read':False});seal(O,'R89_ALL_SOURCE_SELECTED_ZERO_UPDATE_AND_TTA_ENDPOINTS_SEALED',n=len(rows),arm_names=ARMS89);print(read(O/'COUNTS.json'))
