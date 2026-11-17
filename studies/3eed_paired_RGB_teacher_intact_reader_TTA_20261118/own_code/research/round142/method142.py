"""Same Up4896/R136 update; real B-image teacher, full paired-image final reader."""
from common142 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from common142 import R,H,B,rd,sha,dump,check,utc
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()

def checkmethod():
 check();assert rd(H/'IMAGE_DECISION.json')['passed'] and rd(H/'SOURCE_AUDIT.json')['status']=='PASS'
 for p,h in rd(H/'experiment/TTA_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p

def main():
 checkmethod();assert shutil.disk_usage(R).free>=20*1024**3;reg=rd(H/('TTA_ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering_method' if args.engineering else 'method');root.mkdir(exist_ok=False);n=0;fits=0
 if not args.engineering:assert rd(H/'TTA_ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   old=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));scores={k:op[k].copy() for k in ['Frozen_A','Frozen_F','R135_A','R135_F','C_B_A','direct_C_B_A','F_RGBB_only','F_RGBB_only_wrong','F_Bimage_only_teacher']};ev=load(B/'evidence_PRIVATE'/name/(sid+'.npz'));eligible=bool(ev['eligible']);scores['direct_image_B']=ev['direct'];scores['direct_wrong_image_B']=ev['wrong_direct'];ass=ev['assignment'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];ff={};ss={};records={};z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'))
   if eligible:
    gA,_=routegeom135(e,z0);gB,_=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rd(R/e['registration_path']));g=merge_graphs141([gA,gB]);d=tensor_data(g,e['row']);rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));route=tensor_route89(rt);f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=dict(tensor_prefix(z0),vis0=P[None]);im=torch.as_tensor(gB['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255;Sg=torch.as_tensor(ev['S'],device='cuda');Cg=torch.as_tensor(ev['C'],device='cuda')
   for arm in ['image_B_TTA','wrong_image_B_TTA']:
    if not eligible or (arm.startswith('wrong') and not bool(ev['effective_wrong'])):
     scores[arm]=scores['image_B_TTA'].copy() if arm.startswith('wrong') else op['F_RGBB_only'].copy();records[arm]={'reused':True};continue
    target=torch.as_tensor(ev['wrong_target'] if arm.startswith('wrong') else ev['target'],device='cuda')
    def task(lp):
     pt=taskloss(lp,gs,Sg,Cg,None,'partial');gm=torch.stack([lp[torch.as_tensor(ix,device='cuda')].max() for ix in gs]);ls=gm[Cg].log_softmax(0);rank=(target*(target.clamp_min(1e-300).log()-ls)).sum();return pt+rank,{'partial':pt,'conditional_B_image_KL':rank}
    with rgb_banks141(m,[im]):zero,prob,feat,statefit,record=fit136(base,m,S,z,d,route,task,args.engineering)
    assert abs(zero-op['F_RGBB_only']).max()<1e-7;records[arm]=record|{'point_keep_anchor':'intact source-adapted F1024','final_reader':'complete Fpoints RGB_A_B','teacher':'frozen Fpoints RGB_B only'};scores[arm]=prob;fits+=1
    for key,val in feat.items():ff[arm+'_'+key]=val
    for key,val in statefit.items():ss[arm+'_'+key]=val
   restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss);dump(D/'records_PRIVATE'/(sid+'.json'),{'eligible':eligible,'fits':records,'GT_read':False,'source_initialization_only':True,'seconds':time.perf_counter()-tick,'image_B_teacher_and_C_fixed_before_updates':True});n+=1
   if args.engineering or n%32==0:print('METHOD142',args.engineering,src,n,fits,flush=True)
   if n%32==0:assert shutil.disk_usage(R).free>=12*1024**3
  assert fingerprint(base)==finger;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'fits':fits,'GT_read':False,'primary':'image_B_TTA','source_kappas_sha256':sha(H/'SOURCE_KAPPAS_LOCK.json'),'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('METHOD142_COMPLETE',n,fits,flush=True)
if __name__=='__main__':main()
