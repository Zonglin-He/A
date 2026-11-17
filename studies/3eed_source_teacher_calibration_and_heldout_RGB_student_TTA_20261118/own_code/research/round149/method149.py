"""Full frozen-F point reader, A-image student, independently audited streaming output.

The source-only teacher factor selected alpha1 for every source, so cal_A is
an exact alias of plain_A and cal_AB reuses pinned parent R145 predictions.
"""
from common149 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from common149 import R,H,B,rd,sha,dump,check,utc
sys.path.insert(0,str(R/'research/round145'));from position145 import normalize145
from storage149 import storage149
import argparse,subprocess
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()

def main():
 check();assert rd(H/'PREPARATION_AUDIT.json')['status']=='PASS' and rd(H/'RESOURCE_PROFILE.json')['status']=='PASS_BEFORE_MODEL';assert all(a==1 for a in rd(H/'SOURCE_ALPHA_LOCK.json')['alpha'].values());storage149();reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering_method' if args.engineering else 'method');root.mkdir(exist_ok=False);n=0;fits=0;chunk=[];ci=0;met={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'POSITION_ROWS_PRIVATE.json')};parentseal=rd(R/'crossplatform/round148/method/SEALED.json')['files']
 if not args.engineering:assert rd(H/'TTA_ENGINEERING_AUDIT.json')['status']=='PASS'
 def flush():
  nonlocal chunk,ci
  if not chunk:return
  ci+=1;d=root/'audit_chunks_PRIVATE';d.mkdir(exist_ok=True);p=d/(str(ci).zfill(4)+'.json');paths=[]
  for r in chunk:
   D=root/(r['source']+'_to_'+r['target']);sid=r['id_PRIVATE']
   for folder,suf in [('samples_PRIVATE','.npz'),('features_TEMP_PRIVATE','.npz'),('states_PRIVATE','.npz'),('records_PRIVATE','.json')]:paths.append(D/folder/(sid+suf))
  dump(p,dict(root=str(root.relative_to(R)),engineering=args.engineering,rows=chunk,files={str(q.relative_to(root)):sha(q) for q in paths}));log=p.with_suffix('.log');torch.cuda.empty_cache()
  with log.open('w') as f:
   proc=subprocess.run([sys.executable,str(R/'tools/experiment_guard_round149.py'),'--resource','cpu','--timeout-seconds','1800','--',sys.executable,str(H/'audit_chunk149.py'),'--chunk',str(p)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert proc.returncode==0,(str(log),proc.returncode);audit=p.with_name(p.stem+'_AUDIT.json');au=rd(audit);assert au['status']=='PASS' and au['n']==len(chunk) and au['chunk_manifest_sha256']==sha(p)
  if not args.engineering:
   for q in paths:
    if q.parent.name=='features_TEMP_PRIVATE':
     assert q.resolve().is_relative_to(root.resolve()) and au['native_features_sha256'][str(q.relative_to(root))]==sha(q);q.unlink()
  chunk=[];storage149();print('CHUNK149_COMMITTED',ci,au['n'],au['newfits'],flush=True)
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_TEMP_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   rel=name+'/samples_PRIVATE/'+sid+'.npz';assert sha(R/'crossplatform/round148/method'/rel)==parentseal[rel];old=np.load(R/'crossplatform/round148/method'/rel);op=dict(zip(old['arm_names'],old['scores']));scores={k:v.copy() for k,v in op.items()};ev=load(B/'evidence_PRIVATE'/name/(sid+'.npz'));eligible=bool(ev['eligible']);assert bool(ev['same_teacher']);ass=ev['assignment'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));ff={};ss={};records={};bridge=0.
   if eligible:
    gA,_=routegeom135(e,z0);g=merge_graphs141([gA]);d=tensor_data(g,e['row']);d['wrong_permutation']=torch.arange(1024,device='cuda');rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));route=tensor_route89(rt);f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=normalize145(base,dict(tensor_prefix(z0),vis0=P[None]),met[src,tgt,sid]['delta']);pos=np.load(R/'crossplatform/round145/image'/name/'position_PRIVATE'/(sid+'.npz'));assert np.array_equal(z['pos0'].cpu().numpy(),pos['pos0']) and all(np.array_equal(z['position'+str(i)].cpu().numpy(),pos['positions'][i]) for i in range(6));Sg=torch.as_tensor(ev['S'],device='cuda');Cg=torch.as_tensor(ev['C'],device='cuda')
   for arm in ['plain_A_TTA','plain_A_wrong_TTA']:
    if not eligible or ('wrong' in arm and not bool(ev['plain_effective_wrong'])):
     scores[arm]=scores['plain_A_TTA'].copy() if 'wrong' in arm and eligible else op['Frozen_A'].copy();records[arm]=dict(reused=True,parent_arm='plain_A_TTA' if 'wrong' in arm and eligible else 'Frozen_A');continue
    target=torch.as_tensor(ev['plain_wrong_target'] if 'wrong' in arm else ev['plain_target'],device='cuda')
    def task(lp):
     pt=taskloss(lp,gs,Sg,Cg,None,'partial');gm=torch.stack([lp[torch.as_tensor(ix,device='cuda')].max() for ix in gs]);ls=gm[Cg].log_softmax(0);rank=(target*(target.clamp_min(1e-300).log()-ls)).sum();return pt+rank,dict(partial=pt,conditional_B_image_KL=rank)
    calls=[]
    def inputguard(_m,_a):assert torch.equal(_a[0].detach(),P);calls.append(1)
    hook=m.point_adapter.register_forward_pre_hook(inputguard)
    with rgb_banks141(m,[]):zero,prob,feat,statefit,record=fit136(base,m,S,z,d,route,task,args.engineering)
    hook.remove();assert calls;err=float(abs(zero-op['norm_Aimage']).max());bridge=max(bridge,err);assert err<1e-7,('Aimage_zero_bridge',err);idx=np.linspace(0,1023,32).astype(np.int64);feat['adapted_A_probe']=feat['adapted_A'][idx];feat['probe_indices']=idx
    if not args.engineering:del feat['adapted_A']
    record.update(actual_F_input_calls=len(calls),actual_F_input_sha256=hashlib.sha256(P.cpu().numpy().tobytes()).hexdigest(),student_RGB_banks=1,teacher_RGB='B only',student_keeps_Fpoints_including_registered_B_points=True,full_adapted_A_saved=args.engineering);records[arm]=record;scores[arm]=prob;fits+=1
    for k,v in feat.items():ff[arm+'_'+k]=v
    for k,v in statefit.items():ss[arm+'_'+k]=v
   for arm,key in [('cal_A_TTA','plain_A_TTA'),('cal_A_wrong_TTA','plain_A_wrong_TTA'),('cal_AB_TTA','norm_TTA'),('cal_AB_wrong_TTA','norm_wrong_TTA')]:scores[arm]=scores[key].copy()
   scores['cal_A_direct']=ev['direct_A'];scores['cal_AB_direct']=ev['direct_AB'];scores['cal_B']=ev['cal_B'];scores['cal_FA_FB_mean']=(op['norm_Aimage']+ev['cal_B'])/2;scores['cal_FAB_FB_mean']=(op['norm_joint_zero']+ev['cal_B'])/2;restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_TEMP_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss);dump(D/'records_PRIVATE'/(sid+'.json'),dict(eligible=eligible,fits=records,GT_read=False,source_initialization_only=True,zero_bridge_error=bridge,seconds=time.perf_counter()-tick));chunk.append(dict(source=src,target=tgt,id_PRIVATE=sid,has_B=e['has_B']));n+=1
   if len(chunk)>=32:flush()
   if args.engineering or n%32==0:print('METHOD149',args.engineering,src,n,fits,flush=True)
  assert fingerprint(base)==finger;del base,m;torch.cuda.empty_cache()
 flush();dump(root/'SEALED.json',dict(n=n,newfits=fits,GT_read=False,primary='cal_A_TTA',alpha1_exact_aliases=True,temporary_native_features_CPU_audited_then_deleted=not args.engineering,source_alpha_lock_sha256=sha(H/'SOURCE_ALPHA_LOCK.json'),FINAL_access=False,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}));print('METHOD149_COMPLETE',n,fits,flush=True)
if __name__=='__main__':main()
