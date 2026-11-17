"""CPU geometry, original pixels, CNN engineering and full final-readout audit."""
from audit_geometry151 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--chunk',required=True);args=ap.parse_args();ARMS=['high_local','low_local','wrong_local','duplicate_local','region_swap_local']
def main():
 check();torch.set_num_threads(4);p=Path(args.chunk).resolve();assert p.is_relative_to(B);ch=rd(p);root=R/ch['root'];assert root.is_relative_to(B);n=0;nf=0;maxp=0.;maxg=0.;maxcnn=0.;temp={};reg={(e['source'],e['target'],e['row']['id']):e for e in rd(H/'INPUT_REGISTRY_PRIVATE.json')};last=None;W=None
 for rel,h in ch['files'].items():assert sha(root/rel)==h
 for r in ch['rows']:
  src,tgt,sid=r['source'],r['target'],r['id_PRIVATE'];e=reg[src,tgt,sid];name=src+'_to_'+tgt;D=root/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9);rec=rd(D/'records_PRIVATE'/(sid+'.json'));assert rec['GT_read']==False and rec['source_parameters_updated']==0;recipes=rd(B/'recipes_PRIVATE'/name/(sid+'.json'))['recipes'];z=dict(np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz')));ff=np.load(D/'features_TEMP_PRIVATE'/(sid+'.npz'));banks=np.load(D/'graphs_TEMP_PRIVATE'/(sid+'.npz'))
  if recipes:
   ev=dict(np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz')));spec=independently_build(e,z,ev,recipes,True)
   if last!=src:W=tensors(dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))));last=src
   rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'))));f=np.load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));P=np.where(f['valid_B'][:,None],(z['vis0'][0]+f['B_vis'])/2,z['vis0'][0]);ph=hashlib.sha256(P.tobytes()).hexdigest()
   for arm,graphs in spec.items():
    ex=merged(graphs)
    for k,v in ex.items():
     ac=banks[arm+'_'+k]
     if k.endswith('weight'):
      err=float(abs(v-ac).max(initial=0));maxg=max(maxg,err);assert err<1e-5,(arm,k,err)
     else:assert np.array_equal(v,ac),(arm,k)
    f=tensors({k:ff[arm+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']});calc=read135(f,rt,W);err=float(abs(calc-ps[arm]).max());maxp=max(maxp,err);assert err<5e-6,(arm,err);record=rec['arms'][arm];ordered=graphs[1:]+graphs[:1];assert record['RGB_input_calls_verified']==len(graphs) and record['actual_F_input_calls']>0 and record['actual_F_input_sha256']==ph and record['RGB_call_order_sha256']==[hashlib.sha256(g['image'].tobytes()).hexdigest() for g in ordered]
    if ch['engineering']:
     for j,g in enumerate(ordered):
      err=float(abs(cnn(g['image'],W)-ff[arm+'_CNN'+str(j)]).max());maxcnn=max(maxcnn,err);assert err<1e-4,(arm,j,err)
    else:assert not record['CNN_arrays_saved']
    nf+=1
  else:
   key='norm_joint_zero' if e['has_B'] else 'Frozen_A'
   for arm in ARMS:assert np.array_equal(ps[arm],op[key])
  assert np.array_equal(ps['high_zero_mean'],(ps['high_local']+op['norm_joint_zero'])/2)
  for arm,key in [('high_parent_mean','high_local'),('low_parent_mean','low_local'),('wrong_parent_mean','wrong_local'),('duplicate_parent_mean','duplicate_local'),('region_swap_parent_mean','region_swap_local')]:assert np.array_equal(ps[arm],(ps[key]+op['norm_TTA'])/2)
  assert rec['zero_bridge_error']<1e-7
  for fld in ['features_TEMP_PRIVATE','graphs_TEMP_PRIVATE']:
   path=D/fld/(sid+'.npz');temp[str(path.relative_to(root))]=sha(path)
  n+=1
 dump(p.with_name(p.stem+'_AUDIT.json'),dict(status='PASS',n=n,forwards=nf,max_CPU_probability_error=maxp,max_graph_weight_error=maxg,max_engineering_CNN_error=maxcnn,GT_read=False,chunk_manifest_sha256=sha(p),files=ch['files'],temporary_files_sha256=temp,rows=ch['rows'],source_state_unchanged=True,full_transformer_not_CPU_regenerated=True,utc=utc()));print('CHUNK151_AUDIT',n,nf,maxp,maxg,maxcnn,flush=True)
if __name__=='__main__':main()
