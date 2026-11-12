"""Fixed actual spatial reading vs same-information updates, no full pseudo label."""
from core88 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();start=time.perf_counter();plan=read(H88/'experiment/DESIGN_LOCK.json');pin(plan);base=load_source(read(ROOT/plan['source_plan']));m=Fusion88(base.contrastive_align_projection_image).cuda().eval();S=load(ROOT/plan['init_state']);restore(m,S);fp=fingerprint(base);rows=read(H88/'experiment/DEV336_PRIVATE.json')
if args.engineering:rows=[next(r for r in rows if r['platform']==p) for p in ['drone','quad']]
O=B88/('engineering_prediction_attempt001' if args.engineering else 'prediction336_attempt001')
for k in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE']:(O/k).mkdir(parents=True,exist_ok=False)
counts={'main_calls':0,'objective_calls':0,'accepted_updates':0,'trial_calls':0};statuses=[];metrics=[]
def main(*a,**kw):counts['main_calls']+=1;return m.main(*a,**kw)
def obj(*a,**kw):counts['objective_calls']+=1;counts['main_calls']+=3;return objective88(*a,**kw)
for nr,r in enumerate(rows):
 sid=r['id'];z0,z,d=item80(r);ef=B88/('engineering_evidence_attempt001' if args.engineering else 'evidence336_attempt001');a=load(ef/'samples_PRIVATE'/(sid+'.npz'));region={'point':a['region_point'],'image':a['region_image']};ab=[mask86(r,d,region,k) for k in [88001,88002]];pm=[x[0] for x in ab];im=[x[1] for x in ab];ws=local_weights85(d,pm,im,region);m.route=tensor_route88(a);old=load(B87/'prediction336_attempt001/samples_PRIVATE'/(sid+'.npz'));oldmap=dict(zip(old['arm_names'],old['scores']));secondary=load(B87/'structural_prediction336_attempt001/samples_PRIVATE'/(sid+'.npz'));ss=dict(zip(secondary['arm_names'],secondary['scores']));result={x:oldmap[x].copy() for x in ['source','frozen_bi','frozen_ABC','R59']};result.update(R87_full=oldmap['full_tta1'],R87_repaired=ss['full_tta1'],R87_hard=oldmap['hard_alias_tta1']);st={x:{'unknown':False,'fallback':False,'no_parameter_update':True} for x in result};rec={};save={}
 for variant in MODES88+['physical_only','no_rank','no_correspondence']:
  kind=variant if variant in ['physical_only','no_rank','no_correspondence'] else 'full';mode=variant if variant in MODES88 else 'full';restore(m,S);m.requires_grad_(False);m.route_mode=mode
  with torch.no_grad():refp,refq,extra=main(base,z,d,'bidirectional');refs=extra['local_reads'].detach()
  if variant in MODES88:result[variant+'_direct']=refp.cpu().numpy();st[variant+'_direct']={'unknown':not bool(a['supports'].any()),'fallback':False,'no_parameter_update':True}
  enable79(m);trace=[];steps=3 if variant=='full' else 1
  for k in range(1,steps+1):
   L,p,before=obj(m,base,z,d,refp,refq,refs,pm,im,ws,kind);ps=[p for _,p in m.tta_parameters()];gg=torch.autograd.grad(L,ps,allow_unused=True);g0=flat86(gg,ps);g=g0*min(1.,1/(float(g0.norm())+1e-6));initial=[p.detach().clone() for p in ps];dot=float(g0@g);trials=[];accepted=False;alpha=0.
   if dot>1e-24:
    for scale in [1/2**j for j in range(9)]:
     apply86(m,initial,-.1*scale*g)
     with torch.no_grad():lv,pv,parts=obj(m,base,z,d,refp,refq,refs,pm,im,ws,kind)
     bound=float(L)-1e-4*.1*scale*dot;ok=float(lv)<=bound+1e-10;trials.append({'scale':scale,'actual_total':float(lv),'bound':bound,'accepted':ok});counts['trial_calls']+=1
     if ok:accepted=True;alpha=scale;counts['accepted_updates']+=1;break
   if not accepted:apply86(m,initial,torch.zeros_like(g))
   with torch.no_grad():lv,pv,parts=obj(m,base,z,d,refp,refq,refs,pm,im,ws,kind)
   assert not accepted or float(lv)<=float(L)+1e-10
   delta=float(torch.sqrt(sum((p-torch.tensor(S[n],device='cuda')).square().sum() for n,p in m.tta_parameters())));one={'step':k,'accepted':accepted,'scale':alpha,'before':float(L),'after':float(lv),'parts_before':{n:float(v) for n,v in before.items()},'parts_after':{n:float(v) for n,v in parts.items()},'gradient_norm_clipped':float(g.norm()),'gradient_norm_raw':float(g0.norm()),'directional_derivative':dot,'parameter_L2':delta,'trials':trials};trace.append(one)
   if k in [1,3]:
    arm=variant+'_tta'+str(k) if variant in MODES88 else variant+'1';result[arm]=pv.cpu().numpy();np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch85(m));st[arm]={'unknown':not bool(a['supports'].any()),'fallback':delta==0,'no_parameter_update':delta==0}
  rec[variant]=trace
 restore(m,S);m.route_mode='off';m.requires_grad_(False)
 with torch.no_grad():assert np.array_equal(main(base,z,d,'bidirectional')[0].cpu().numpy(),oldmap['frozen_bi'])
 scores=np.stack([result[x] for x in ARMS88]);assert np.isfinite(scores).all();queries=scores.argmax(-1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS88),scores=scores,queries=queries,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][queries],mask_point=torch.stack(pm).cpu().numpy(),mask_image=torch.stack(im).cpu().numpy());dump(O/'records_PRIVATE'/(sid+'.json'),rec);statuses.append({'id':sid,'arm_statuses':st});print('RUN88',nr+1,len(rows),flush=True)
assert fingerprint(base)==fp and all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',{'n':len(rows),'arms':len(ARMS88),**counts,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False,'TTA_parameters':sum(p.numel() for _,p in m.tta_parameters())});seal(O,'R88_ALL_ROUTED_READOUTS_AND_PARAMETER_UPDATES_SEALED',n=len(rows),arm_names=ARMS88);print(read(O/'COUNTS.json'))
