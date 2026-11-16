"""Independent CPU readout and immutable source/alias audit before new task scoring."""
from pathlib import Path
import sys,json,numpy as np,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round115';B=R/'crossplatform/round115';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112

def main():
 mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==768 and len(mat['new_arms'])==6;assert rd(H/'ENGINEERING.json')['n']==12;n=heads=aliases=ties=0;mx=0.
 for name,h in mat['directions'].items():
  D=B/'formal'/name;assert sha(D/'SEALED.json')==h
  for p,v in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==v
  src,_,tgt=name.split('_')
  with np.load(R/'crossplatform/round113'/(src+'_SOURCE_READOUT_PRIVATE.npz')) as sh:head={k[6:]:sh[k] for k in sh.files if k.startswith('match.')}
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   sid=r['id'];meta=rd(D/'records_PRIVATE'/(sid+'.json'));p=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(p['arm_names'],p['scores']));prior=np.load(R/'crossplatform/round114/formal'/name/'samples_PRIVATE'/(sid+'.npz'));old=dict(zip(prior['arm_names'],prior['scores']));assert np.array_equal(p['source_boxes'],prior['source_boxes']) and np.array_equal(p['queries'],p['scores'].argmax(1));n+=1
   for k in po:
    if k not in mat['new_arms']:assert np.array_equal(po[k],old[k])
   if meta['protected_alias_fallback']:
    aliases+=1;assert all(np.array_equal(po[k],old['prepared_frozen']) for k in mat['new_arms']);continue
   assert meta['all_control_proposals_fixed_before_C'] is True and meta['evaluation']['C_loaded_only_after_all_fit_acceptance'] is True
   with np.load(R/'crossplatform/round113'/name/'features_PRIVATE'/(sid+'.npz')) as a,np.load(D/'features_PRIVATE'/(sid+'.npz')) as ff:
    cache={v:{k:a['frozen_anchor_'+v+'_'+k][0] if k in ['h','u'] else a['frozen_anchor_'+v+'_'+k] for k in ['h','u','mask','r']} for v in ['A','C','fusedABC']}
    for arm in mat['new_arms']:
     z=cache[arm.split('_')[-1]];pc=cpu112(z['h'],z['u'],z['mask'],ff[arm+'_late'],z['r'],head);er=float(abs(pc-po[arm]).max());mx=max(mx,er);assert er<5e-6
     if int(pc.argmax())!=int(po[arm].argmax()):assert po[arm].max()-po[arm][int(pc.argmax())]<=2*er+1e-12;ties+=1
     heads+=1
 assert n==768 and aliases==44 and heads==4344;result={'state':'PASS_ALL_NEW6_CPU_READOUTS_SEALS_SOURCE_GEOMETRY_PROTECTED_ALIASES','n':n,'new_CPU_readouts':heads,'aliases':aliases,'CPU_probability_max_error':mx,'CPU_near_ties':ties,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
