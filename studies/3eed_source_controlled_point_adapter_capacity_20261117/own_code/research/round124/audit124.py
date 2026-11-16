"""Independent CPU source-head, step arithmetic, scope/norm and zero bridge audit."""
from pathlib import Path
import json,sys,hashlib,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round124';B=R/'crossplatform/round124';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==384 and rd(H/'ENGINEERING.json')['n']==6
 for p,h in rd(H/'experiment/DESIGN_LOCK.json')['files'].items():assert sha(R/p)==h
 reg=rd(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json');col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=heads=states=0;maxp=maxd=maxnorm=0.;fd=[]
 for src in ['waymo','drone','quad']:
  D=B/'formal'/src;assert sha(D/'SEALED.json')==mat['seals'][src]
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  assert sha(R/col[src]['state_path'])==col[src]['state_sha256'];S=dict(np.load(R/col[src]['state_path']));head={k[6:]:v for k,v in S.items() if k.startswith('match.')};radius=.01*np.sqrt(sum(np.square(v.astype(float)).sum() for k,v in S.items() if k.startswith('point_adapter.')))
  for file in (B/'engineering'/src/'records_PRIVATE').glob('*.json'):
   x=rd(file)['finite_difference']
   if x is not None:assert x['relative_error']<.1;fd.append(x['relative_error'])
  for e in [e for e in reg if e['source']==src]:
   sid=e['row']['id'];sample=np.load(D/'samples_PRIVATE'/(sid+'.npz'));features=np.load(D/'features_PRIVATE'/(sid+'.npz'));st=np.load(D/'states_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));prior=np.load(R/'crossplatform/round121/formal'/src/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(prior['arm_names'],prior['scores']));po=dict(zip(sample['arm_names'],sample['scores']));assert np.array_equal(sample['queries'],sample['scores'].argmax(1)) and np.array_equal(po['original_source'],op['source']) and np.array_equal(po['prepared'],op['prepared']);z=np.load(R/e['files']['prefix']['path']);assert np.array_equal(features['zero_h'],z['h']) and np.array_equal(features['zero_u'],z['tokens']);assert abs(rec['radius']-radius)<1e-9
   delta=torch.zeros(4896)
   for t in range(3):
    g=torch.from_numpy(st['step'+str(t+1)+'_gradient']);assert torch.isfinite(g).all();step=-radius*g/g.norm() if rec['eligible'] and g.norm()>0 else g*0;delta=delta+step
    if t in [0,2]:er=float(abs(delta-torch.from_numpy(st['oracle'+str(t+1)])).max());maxd=max(maxd,er);assert er<5e-6
   for t in [1,3]:
    rng=np.random.default_rng(int(hashlib.sha256(('R124|'+src+'|'+sid+'|'+str(t)).encode()).hexdigest()[:16],16));v=torch.tensor(rng.normal(size=4896),dtype=torch.float32);want=v/v.norm()*torch.from_numpy(st['oracle'+str(t)]).norm();er=float(abs(want-torch.from_numpy(st['random'+str(t)])).max());maxd=max(maxd,er);assert er<5e-6
   y=np.load(R/e['files']['label']['path'])['ious'];best=np.isclose(y,y.max(),rtol=0,atol=1e-8)
   for arm in ['zero','oracle1','oracle3','random1','random3']:
    delta=np.zeros(4896) if arm=='zero' else st[arm];er=abs(np.linalg.norm(delta.astype(float))-rec['endpoints'][arm]['parameter_L2']);maxnorm=max(maxnorm,er);assert er<1e-8 and rec['endpoints'][arm]['changed_tensors']<=2;pp=cpu112(features[arm+'_h'],features[arm+'_u'],features[arm+'_mask'],features['fixed_late'],features['fixed_r'],head);er=float(abs(pp-po[arm]).max());maxp=max(maxp,er);assert er<5e-6 and pp.argmax()==po[arm].argmax();loss=-np.log(max(pp[best].sum(),1e-300)) if rec['eligible'] else 0.;assert abs(loss-rec['endpoints'][arm]['GT_loss'])<5e-5;heads+=1;states+=1
   assert po['zero'].argmax()==op['context_zero'].argmax() and np.max(abs(po['zero']-op['context_zero']))<5e-6;n+=1
 assert n==384 and heads==1920 and states==1920 and len(fd)==6;out={'state':'PASS_ALL_SOURCE_ZERO_BRIDGES_CPU_HEADS_NORMALIZED_STEPS_NORMS_RESTORE_AND_6_ENGINEERING_FINITE_DIFFERENCES','n':n,'CPU_heads':heads,'states':states,'max_probability_error':maxp,'max_state_arithmetic_error':maxd,'max_norm_error':maxnorm,'finite_difference_n':len(fd),'finite_difference_max_relative_error':max(fd),'full_decoder_gradients_independently_CPU_recomputed':False,'gradient_validation':'six engineering central differences; all normalized-step arithmetic replayed from saved autograd gradients, no claim all CPU decoder backprop','source_GT_is_NOT_TTA':True,'target_data_read':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
