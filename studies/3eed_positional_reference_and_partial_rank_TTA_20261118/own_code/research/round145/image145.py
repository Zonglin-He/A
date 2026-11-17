"""Frozen normalized-reader controls, real/wrong B binding and canonical means."""
from common145 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from position145 import normalize145,savepos145
from common145 import R,H,B,rd,sha,dump,check,utc
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
ARMS=['norm_Aimage','norm_Bimage','norm_joint_wrong','norm_B_wrong']
def main():
 check();assert shutil.disk_usage(R).free>=18*1024**3;reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));meta={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(R/'research/round144/POSITION_ROWS_PRIVATE.json')};root=B/('image_engineering' if args.engineering else 'image');root.mkdir(exist_ok=False);n=nf=0;mx=0.
 if not args.engineering:assert rd(H/'IMAGE_ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);np.savez_compressed(root/(src+'_POSITION_MODULE_PRIVATE.npz'),**{k:v.detach().cpu().numpy() for k,v in base.pos_embed.state_dict().items()})
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','banks_PRIVATE','position_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   old=np.load(R/'crossplatform/round144/formal'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(old['arm_names'],old['scores']));ps['norm_joint_zero']=ps['aligned_reference'].copy();restore(m,S);freeze91(m);ff={};bb={};pp={};rec={'has_B':e['has_B'],'source_parameters_updated':0,'GT_read':False,'delta':meta[src,tgt,sid]['delta']}
   if not e['has_B']:
    for arm in ARMS:ps[arm]=ps['Frozen_A'].copy()
   else:
    z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));gA,_=routegeom135(e,z0);gB,_=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rd(R/e['registration_path']));bad=wrong_extra_graph141(gB,src+'|'+sid+'|R142imageonlywrong');rec['wrong_effective']=bool(not np.array_equal(gB['p2i'],bad['p2i']) or not np.array_equal(gB['i2p'],bad['i2p']));rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=normalize145(base,dict(tensor_prefix(z0),vis0=P[None]),rec['delta']);savepos145(pp,z,z0,rec['delta']);oldpos=np.load(R/'crossplatform/round144/formal'/name/'position_PRIVATE'/(sid+'.npz'));assert np.array_equal(pp['pos0'],oldpos['aligned_reference_pos0']) and np.array_equal(pp['positions'],oldpos['aligned_reference_positions']);rec['R144_positions_exact']=True
    if args.engineering:
     zero,_,_=read_bank141(base,m,z,[gA,gB],rt,e['row']);err=float(abs(zero-ps['norm_joint_zero']).max());mx=max(mx,err);assert err<1e-7;nf+=1;rec['normalized_zero_bridge_error']=err
    specs={'norm_Aimage':[gA],'norm_Bimage':[gB],'norm_joint_wrong':[gA,bad],'norm_B_wrong':[bad]}
    for arm,graphs in specs.items():
     prob,a,mg=read_bank141(base,m,z,graphs,rt,e['row']);ps[arm]=prob;nf+=1
     for k,v in a.items():ff[arm+'_'+k]=v
     for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']:bb[arm+'_'+k]=mg[k]
   ps['norm_FA_FB_mean']=(ps['norm_Aimage']+ps['norm_Bimage'])/2;ps['norm_FAB_FB_mean']=(ps['norm_joint_zero']+ps['norm_Bimage'])/2;assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(ps.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(ps)),queries=arr.argmax(1),source_boxes=old['source_boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'banks_PRIVATE'/(sid+'.npz'),**bb);np.savez_compressed(D/'position_PRIVATE'/(sid+'.npz'),**pp);rec['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
   if args.engineering or n%64==0:print('IMAGE145',args.engineering,src,n,nf,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'full_reader_forwards':nf,'max_R144_zero_bridge_error':mx,'parameters_updated':0,'target_GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('IMAGE145_COMPLETE',n,nf,flush=True)
if __name__=='__main__':main()
