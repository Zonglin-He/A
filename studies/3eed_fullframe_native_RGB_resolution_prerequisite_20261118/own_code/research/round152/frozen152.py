"""Real full-frame raw pixels on dynamic grids through the unchanged model."""
from common152 import *
from core152 import *
from common152 import R,H,B,rd,sha,dump,check,utc
from geometry152 import specs,load
from storage152 import storage152
import argparse,subprocess
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();ARMS=['full_high','full_low','full_wrong']

def main():
 check();assert rd(H/'PREPARATION_AUDIT.json')['status']=='PASS' and rd(H/'RESOURCE_PROFILE.json')['status']=='PASS_BEFORE_MODEL';storage152();reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering_frozen' if args.engineering else 'frozen');root.mkdir(exist_ok=False);n=0;nf=0;chunk=[];ci=0;parent=rd(R/'crossplatform/round150/frozen/SEALED.json')['files'];parent149=rd(R/'crossplatform/round149/method/SEALED.json')['files'];geometry={(e['source'],e['target'],e['id_PRIVATE']):e for e in rd(H/'GEOMETRY_ROWS_PRIVATE.json')}
 if not args.engineering:assert rd(H/'FROZEN_ENGINEERING_AUDIT.json')['status']=='PASS'
 def flush():
  nonlocal chunk,ci
  if not chunk:return
  ci+=1;D=root/'audit_chunks_PRIVATE';D.mkdir(exist_ok=True);p=D/(str(ci).zfill(4)+'.json');paths=[]
  for e in chunk:
   dn=root/(e['source']+'_to_'+e['target']);sid=e['id_PRIVATE']
   for f,suf in [('samples_PRIVATE','.npz'),('features_TEMP_PRIVATE','.npz'),('graphs_TEMP_PRIVATE','.npz'),('records_PRIVATE','.json')]:paths.append(dn/f/(sid+suf))
  dump(p,dict(root=str(root.relative_to(R)),engineering=args.engineering,rows=chunk,files={str(q.relative_to(root)):sha(q) for q in paths}));torch.cuda.empty_cache()
  with p.with_suffix('.log').open('w') as f:proc=subprocess.run([sys.executable,str(R/'tools/experiment_guard_round152.py'),'--resource','cpu','--timeout-seconds','1800','--',sys.executable,str(H/'audit_chunk152.py'),'--chunk',str(p)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert proc.returncode==0,(str(p),proc.returncode);a=rd(p.with_name(p.stem+'_AUDIT.json'));assert a['status']=='PASS' and a['chunk_manifest_sha256']==sha(p)
  if not args.engineering:
   for q in paths:
    if q.parent.name in ['features_TEMP_PRIVATE','graphs_TEMP_PRIVATE']:
     assert q.resolve().is_relative_to(root.resolve()) and a['temporary_files_sha256'][str(q.relative_to(root))]==sha(q);q.unlink()
  chunk=[];storage152();print('CHUNK152_COMMITTED',ci,a['n'],a['forwards'],flush=True)
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tick=time.perf_counter();tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_TEMP_PRIVATE','graphs_TEMP_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   oldpath=R/'crossplatform/round150/frozen'/name/(sid+'.npz');assert sha(oldpath)==parent[name+'/'+sid+'.npz'];old=np.load(oldpath);parentpath=R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz');assert sha(parentpath)==parent149[name+'/samples_PRIVATE/'+sid+'.npz'];a=np.load(parentpath);op=dict(zip(a['arm_names'],a['scores']));z0=load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));geom=geometry[src,tgt,sid];ff={};banks={};scores={};record=dict(has_B=e['has_B'],full_frame_image_shape=[512,768,3],GT_read=False,source_parameters_updated=0,arms={},seconds=0);bridge=0.
   if e['has_B']:
    specification=specs(e,z0);rt=load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));f=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));A=torch.as_tensor(z0['vis0'][0],device='cuda');P=torch.where(torch.as_tensor(f['valid_B'],device='cuda')[:,None],(A+torch.as_tensor(f['B_vis'],device='cuda'))/2,A);pos=load(R/'crossplatform/round145/image'/name/'position_PRIVATE'/(sid+'.npz'));z=dict(tensor_prefix(z0),vis0=P[None],pos0=torch.as_tensor(pos['pos0'],device='cuda'))
    for i in range(6):z['position'+str(i)]=torch.as_tensor(pos['positions'][i],device='cuda')
    if args.engineering:
     from crop151 import target_geometry
     gg,_,_=target_geometry(e,z0);zero,_,_=read_bank141(base,m,z,gg,rt,e['row']);bridge=float(abs(zero-op['norm_joint_zero']).max());assert bridge<1e-7;nf+=1
    for arm,graphs in specification.items():
     assert [hashlib.sha256(g['image'].tobytes()).hexdigest() for g in graphs]==geom['arms'][arm]['pixel_hashes'];ordered=graphs[1:]+graphs[:1];expected=[(torch.as_tensor(g['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255-.5)*2 for g in ordered];calls=[];cnn=[];pcalls=[]
     def rgb_input(_m,_a):assert len(calls)<len(expected) and torch.equal(_a[0],expected[len(calls)]);calls.append(1)
     def rgb_output(_m,_a,o):
      if args.engineering:cnn.append(o.detach().cpu().numpy())
     def point_input(_m,_a):assert torch.equal(_a[0],P);pcalls.append(1)
     hooks=[m.rgb.register_forward_pre_hook(rgb_input),m.rgb.register_forward_hook(rgb_output),m.point_adapter.register_forward_pre_hook(point_input)]
     try:prob,feat,mg=read_dynamic(base,m,z,graphs,rt,e['row'])
     finally:
      for hook in hooks:hook.remove()
     assert len(calls)==len(graphs) and pcalls
     for key,hh in geom['arms'][arm]['graph_hashes'].items():assert hashlib.sha256(np.ascontiguousarray(mg[key]).tobytes()).hexdigest()==hh
     scores[arm]=prob;nf+=1
     for k,v in feat.items():ff[arm+'_'+k]=v
     for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']:banks[arm+'_'+k]=mg[k]
     if args.engineering:
      for j,v in enumerate(cnn):ff[arm+'_CNN'+str(j)]=v
     record['arms'][arm]=dict(RGB_input_calls_verified=len(calls),actual_F_input_calls=len(pcalls),actual_F_input_sha256=hashlib.sha256(P.cpu().numpy().tobytes()).hexdigest(),RGB_call_order_sha256=[hashlib.sha256(g['image'].tobytes()).hexdigest() for g in ordered],CNN_arrays_saved=args.engineering)
   else:
    for arm in ARMS:scores[arm]=op['norm_joint_zero'].copy() if e['has_B'] else op['Frozen_A'].copy()
   scores['full_high_zero_mean']=(scores['full_high']+op['norm_joint_zero'])/2
   for arm,key in [('full_high_parent_mean','full_high'),('full_low_parent_mean','full_low'),('full_wrong_parent_mean','full_wrong')]:scores[arm]=(scores[key]+op['norm_TTA'])/2
   restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=old['source_boxes']);np.savez_compressed(D/'features_TEMP_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'graphs_TEMP_PRIVATE'/(sid+'.npz'),**banks);record.update(zero_bridge_error=bridge,seconds=time.perf_counter()-tick);dump(D/'records_PRIVATE'/(sid+'.json'),record);chunk.append(dict(source=src,target=tgt,id_PRIVATE=sid,has_B=e['has_B']));n+=1
   if len(chunk)>=2:flush()
   if args.engineering or n%64==0:print('FROZEN152',args.engineering,src,n,nf,flush=True)
  assert fingerprint(base)==finger;del base,m;torch.cuda.empty_cache()
 flush();dump(root/'SEALED.json',dict(n=n,full_reader_forwards=nf,GPU_peak_bytes=int(torch.cuda.max_memory_allocated()),parameter_updates=0,GT_read=False,primary='full_high',temporary_features_graphs_audited_then_deleted=not args.engineering,FINAL_access=False,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}));print('FROZEN152_COMPLETE',n,nf,flush=True)
if __name__=='__main__':main()
