"""Frozen positional-reference intervention, complete reader, fixed physical geometry."""
from common144 import *
sys.path.insert(0,str(R/'research/round141'));from core141 import *
from common144 import R,H,B,rd,sha,dump,check,utc
import argparse,shutil
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
NEW=['aligned_reference','opposite_reference','half_zero_aligned']

def main():
 check();assert shutil.disk_usage(R).free>=16*1024**3;reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));meta={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'POSITION_ROWS_PRIVATE.json')};root=B/('engineering' if args.engineering else 'formal');root.mkdir(exist_ok=False);nf=0;n=0;bridge=0.;posbridge=0.
 if not args.engineering:assert rd(H/'ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);np.savez_compressed(root/(src+'_POSITION_MODULE_PRIVATE.npz'),**{k:v.detach().cpu().numpy() for k,v in base.pos_embed.state_dict().items()})
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','position_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   old=np.load(R/'crossplatform/round142/method'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(old['arm_names'],old['scores']));means=np.load(R/'crossplatform/round142/supplementary_mean'/name/(sid+'.npz'));ps.update(dict(zip(means['arm_names'],means['scores'])));ff={};pp={};record={'has_B':e['has_B'],'delta':meta[src,tgt,sid]['delta'],'parameters_updated':0,'GT_read':False};restore(m,S);freeze91(m)
   if not e['has_B']:
    for arm in NEW:ps[arm]=ps['Frozen_A'].copy()
   else:
    z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));gA,_=routegeom135(e,z0);gB,_=geo141.additional(e['A_sensor'],e['pair_PRIVATE']['B'],z0,rd(R/e['registration_path']));rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);z=dict(tensor_prefix(z0),vis0=P[None]);delta=float(record['delta']);pp['physical_xyz']=z0['xyz'];pp['original_positions']=np.stack([z0['position'+str(i)] for i in range(6)]);pp['delta']=np.array(delta)
    if args.engineering:
     with torch.no_grad():pzero=base.pos_embed(z['xyz']).transpose(1,2).contiguous()
     err=float(abs(pzero-z['pos0']).max());posbridge=max(posbridge,err);assert err<2e-5
     zero,_,_=read_bank141(base,m,dict(z,pos0=pzero),[gA,gB],rt,e['row']);err=float(abs(zero-ps['F_RGBB_only']).max());bridge=max(bridge,err);assert err<2e-6;nf+=1;record['zero_bridge_error']=err
    for arm,sign in [('aligned_reference',1),('opposite_reference',-1)]:
     xyz=z['xyz'].clone();xyz[:,:,2]+=sign*delta
     with torch.no_grad():position=base.pos_embed(xyz).transpose(1,2).contiguous()
     zz=dict(z,pos0=position)
     for i in range(6):
      key='position'+str(i);zz[key]=z[key].clone();zz[key][:,:,2]+=sign*delta;assert torch.equal(zz[key][:,:,3:],z[key][:,:,3:]) and torch.equal(zz[key][:,:,:2],z[key][:,:,:2])
     p,a,_=read_bank141(base,m,zz,[gA,gB],rt,e['row']);ps[arm]=p;nf+=1
     for k,v in a.items():ff[arm+'_'+k]=v
     pp[arm+'_pos0']=position.cpu().numpy();pp[arm+'_positions']=np.stack([zz['position'+str(i)].cpu().numpy() for i in range(6)]);pp[arm+'_pos_input']=xyz.cpu().numpy()
    ps['half_zero_aligned']=(ps['F_RGBB_only']+ps['aligned_reference'])/2
   assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(ps.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(ps)),queries=arr.argmax(1),source_boxes=old['source_boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'position_PRIVATE'/(sid+'.npz'),**pp);record['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),record);n+=1
   if args.engineering or n%64==0:print('POSITION144',args.engineering,src,n,nf,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'full_reader_forwards':nf,'parameters_updated':0,'max_engineering_zero_probability_error':bridge,'max_engineering_pos0_recompute_error':posbridge,'GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('POSITION144_COMPLETE',n,nf,flush=True)
if __name__=='__main__':main()
