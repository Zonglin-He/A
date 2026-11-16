"""Independent CPU and immutable-checkpoint audit before any new GT scoring."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
from replay108 import ce_cpu108
R=Path(__file__).resolve().parents[2];H=R/'research/round108';B=R/'crossplatform/round108';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(4);matrix=rd(B/'MATRIX_SEALED.json');assert matrix['n']==768 and len(matrix['arms'])==29 and rd(H/'ENGINEERING.json')['n']==12;n=end=alias=0;mx=0.;sources=rd(R/'research/round97/experiment/DESIGN_LOCK.json')['sources']
 for name,hh in matrix['directions'].items():
  src,_,tgt=name.split('_');O=B/name;old=R/'crossplatform/round106'/name;assert sha(O/'SEALED.json')==hh
  for p,h in rd(O/'SEALED.json')['files'].items():assert sha(O/p)==h
  wf=B/(src+'_CE_WEIGHTS_PRIVATE.npz');assert sha(wf)==matrix['CE_weights'][wf.name];S=dict(np.load(wf));c=next(x for x in sources if x['source']==src);assert sha(c['checkpoint_path'])==c['checkpoint_sha256'];orig=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True);prefix='prediction_heads.5.sem_cls_scores_head.';native={k[len(prefix):]:v.numpy() for k,v in orig['model'].items() if k.startswith(prefix)};assert set(S)==set(native) and all(np.array_equal(S[k],v) for k,v in native.items());del orig
  for r in rd(R/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
   sid=r['id'];z=np.load(O/'samples_PRIVATE'/(sid+'.npz'));pp=dict(zip(z['arm_names'],z['scores']));pz=np.load(old/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(pz['arm_names'],pz['scores']));rec=rd(O/'records_PRIVATE'/(sid+'.json'));assert np.array_equal(z['source_boxes'],pz['source_boxes']);assert np.array_equal(z['queries'],z['scores'].argmax(1));n+=1
   for k in set(pp)&set(po):assert np.array_equal(pp[k],po[k])
   if rec['protected_alias_fallback']:
    alias+=1;assert not rec['proof'];assert all(np.array_equal(p,po['prepared_frozen']) for k,p in pp.items() if k.startswith('softtoken'));continue
   with np.load(old/'features_PRIVATE'/(sid+'.npz')) as f:
    for v in ['A','B','C','fusedABC']:
     h=f['real_'+v+'_h'];mask=f['real_'+v+'_mask'];p=f['real_'+v+'_p'];q,_=ce_cpu108(h,mask,S);assert rec['proof'][v]['input_h_sha256']==hashlib.sha256(h.tobytes()).hexdigest() and rec['proof'][v]['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest();qs=q[np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+sid).encode()).hexdigest()[:16],16)).permutation(256)];soft=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum();vals={'softtoken':q,'softtoken_geometric':soft((np.log(q)+np.log(p))/2),'softtoken_arithmetic':(q+p)/2,'softtoken_shuffled_geometric':soft((np.log(qs)+np.log(p))/2)}
     for k,x in vals.items():e=float(abs(x-pp[k+'_'+v]).max());mx=max(mx,e);assert e<5e-6 and x.argmax()==pp[k+'_'+v].argmax(),(name,sid,k,v,e);end+=1
  print('AUDIT108',name,flush=True)
 dump(H/'NATIVE_AUDIT.json',{'state':'PASS_ALL_IMMUTABLE_CHECKPOINT_SOFTTOKEN_WEIGHTS_NATIVE_CPU_READOUTS_BASELINES_MASKS_BOXES_FALLBACKS','n':n,'arms':29,'endpoints':end,'max_CPU_native_probability_error':mx,'protected_alias_fallback':alias,'full_native_engineering':12,'GT_used':False,'FINAL_access':False});print('AUDIT108_PASS',n,end)
if __name__=='__main__':main()
