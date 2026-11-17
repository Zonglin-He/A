"""Same Up4896/R136 update; real B-image teacher, full paired-image final reader."""
from common146 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from common146 import R,H,B,rd,sha,dump,check,utc
sys.path.insert(0,str(R/'research/round145'));from position145 import normalize145
from storage146 import storage146
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()

def checkmethod():
 check();assert rd(H/'IMAGE_AUDIT.json')['status']=='PASS'
 for p,h in rd(H/'experiment/TTA_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p

def main():
 checkmethod();storage146();reg=rd(H/('TTA_ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering_method' if args.engineering else 'method');root.mkdir(exist_ok=False);n=0;fits=0;met={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'POSITION_ROWS_PRIVATE.json')}
 if not args.engineering:assert rd(H/'TTA_ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   old=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));scores={k:v.copy() for k,v in op.items()};ev=load(B/'evidence_PRIVATE'/name/(sid+'.npz'));eligible=bool(ev['eligible']) and e['backfilled'];scores['bf_direct']=ev['direct'];scores['bf_wrong_direct']=ev['wrong_direct'];ass=ev['assignment'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];ff={};ss={};records={};z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'))
   if eligible:
    gA,_=routegeom135(e,z0);gB,_=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rd(R/e['registration_path']));g=merge_graphs141([gA,gB]);d=tensor_data(g,e['row']);rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));route=tensor_route89(rt);f=load(R/e['backfill_features_path']);A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=normalize145(base,dict(tensor_prefix(z0),vis0=P[None]),met[src,tgt,sid]['delta']);pos=np.load(B/'image'/name/'position_PRIVATE'/(sid+'.npz'));assert np.array_equal(z['pos0'].cpu().numpy(),pos['pos0']) and all(np.array_equal(z['position'+str(i)].cpu().numpy(),pos['positions'][i]) for i in range(6));im=torch.as_tensor(gB['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255;Sg=torch.as_tensor(ev['S'],device='cuda');Cg=torch.as_tensor(ev['C'],device='cuda')
   for arm in ['bf_TTA','bf_wrong_TTA']:
    if not e['backfilled']:
     parent_key={'bf_TTA':'norm_TTA','bf_wrong_TTA':'norm_wrong_TTA'}[arm];scores[arm]=op[parent_key].copy();records[arm]={'reused':True,'parent_arm':parent_key};continue
    if not eligible or (('wrong' in arm) and not bool(ev['effective_wrong'])):
     scores[arm]=scores['bf_TTA'].copy() if ('wrong' in arm) else op['bf_zero'].copy();records[arm]={'reused':True};continue
    target=torch.as_tensor(ev['wrong_target'] if ('wrong' in arm) else ev['target'],device='cuda')
    def task(lp):
     pt=taskloss(lp,gs,Sg,Cg,None,'partial');gm=torch.stack([lp[torch.as_tensor(ix,device='cuda')].max() for ix in gs]);ls=gm[Cg].log_softmax(0);rank=(target*(target.clamp_min(1e-300).log()-ls)).sum();return pt+rank,{'partial':pt,'conditional_B_image_KL':rank}
    with rgb_banks141(m,[im]):zero,prob,feat,statefit,record=fit136(base,m,S,z,d,route,task,args.engineering)
    assert abs(zero-op['bf_zero']).max()<1e-7;records[arm]=record|{'point_keep_anchor':'intact source-adapted F1024','final_reader':'complete Fpoints RGB_A_B','teacher':'frozen Fpoints RGB_B only'};scores[arm]=prob;fits+=1
    for key,val in feat.items():ff[arm+'_'+key]=val
    for key,val in statefit.items():ss[arm+'_'+key]=val
   for arm,key in {'bf_direct':'norm_direct','bf_wrong_direct':'norm_wrong_direct'}.items():
    if not e['backfilled']:scores[arm]=op[key].copy()
   for arm,key in {'inc_zero':'bf_zero','inc_direct':'bf_direct','inc_Bimage':'bf_Bimage','inc_FA_FB_mean':'bf_FA_FB_mean','inc_FAB_FB_mean':'bf_FAB_FB_mean','inc_wrong_TTA':'bf_wrong_TTA','inc_raw_joint':'bf_raw_joint'}.items():scores[arm]=scores[key].copy() if e['backfilled'] else op['norm_TTA'].copy()
   restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss);dump(D/'records_PRIVATE'/(sid+'.json'),{'eligible':eligible,'fits':records,'GT_read':False,'source_initialization_only':True,'seconds':time.perf_counter()-tick,'image_B_teacher_and_C_fixed_before_updates':True,'normalized_position_bridge_checked_if_eligible':True});n+=1
   if args.engineering or n%32==0:print('METHOD146',args.engineering,src,n,fits,flush=True)
   if n%32==0:storage146()
  assert fingerprint(base)==finger;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'fits':fits,'GT_read':False,'primary':'bf_TTA','source_kappas_sha256':sha(H/'SOURCE_KAPPAS_LOCK.json'),'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('METHOD146_COMPLETE',n,fits,flush=True)
if __name__=='__main__':main()
