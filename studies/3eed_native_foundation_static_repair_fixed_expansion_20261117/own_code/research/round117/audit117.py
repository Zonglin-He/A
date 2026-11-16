"""Independent CPU native projection/readout and all fresh sensor prerequisite seals."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round117';B=R/'crossplatform/round117';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112

def main():
 mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==1536 and len(mat['new_arms'])==17 and len(mat['arms'])==23;plan=rd(H/'experiment/DESIGN_LOCK.json')
 for p,h in plan['input_sha256'].items():assert sha(R/p)==h
 n=aliases=heads=ties=0;mx=0.;registry=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules']
 for name,seal in mat['directions'].items():
  src,_,tgt=name.split('_');D=B/'formal'/name;assert sha(D/'SEALED.json')==seal
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  with np.load(R/registry[src]['state_path']) as z:S={k:z[k] for k in z.files if k.startswith('match.')};head={k[6:]:v for k,v in S.items()}
  for ix,r in enumerate(rd(R/f'research/round109/input/TARGET_{tgt.upper()}256_PRIVATE.json')):
   sid=r['id'];meta=rd(D/'records_PRIVATE'/(sid+'.json'));a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round109'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   for k in po:
    if k not in mat['new_arms']:assert np.array_equal(po[k],op[k])
   if meta['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[k],op['prepared_frozen']) for k in mat['new_arms']);continue
   assert meta['gradient_updates']==0 and not meta['TTA'];assert meta['inputs']['original_boxes_fixed'] and meta['inputs']['source_original_probability_error']<5e-6
   with np.load(D/'features_PRIVATE'/(sid+'.npz')) as ff:
    cache={v:{k:ff['native_'+v+'_'+k] for k in ['h','u','late','mask','r']} for v in ['A','B','C','fusedABC']};basep={}
    for view,z in cache.items():basep[view]=cpu112(z['h'],z['u'],z['mask'],z['late'],z['r'],head)
    for arm in mat['new_arms']:
     if arm=='repaired_ABC_mean':p=np.mean([basep[v] for v in ['A','B','C']],axis=0)
     else:
      view=arm.split('_')[-1];z=cache[view];late=z['late'];res=z['r']
      if arm.startswith('native_'):late=late*0;res=res*0
      elif arm.startswith('rOnly_'):late=late*0
      elif arm.startswith('lateOnly_'):res=res*0
      elif arm.startswith(('fixed_calibration_','reverse_calibration_')):late=ff[arm+'_late']
      p=cpu112(z['h'],z['u'],z['mask'],late,res,head)
     er=float(abs(p-po[arm]).max());mx=max(mx,er);assert er<5e-6
     if int(p.argmax())!=int(po[arm].argmax()):assert po[arm].max()-po[arm][int(p.argmax())]<=2*er+1e-12;ties+=1
     heads+=1
   if (ix+1)%64==0:print('AUDIT117',name,ix+1,flush=True)
 assert n==1536 and aliases==110 and heads==24242;out={'state':'PASS_ALL17_NEW_CPU_NATIVE_READOUTS_FIXED_GEOMETRY_SENSOR_AND_ALIAS_SEALS','n':n,'aliases':aliases,'new_CPU_endpoints':heads,'max_probability_error':mx,'CPU_near_ties':ties,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
