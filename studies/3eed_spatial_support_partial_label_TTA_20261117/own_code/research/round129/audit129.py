"""Independent post-seal CPU replay of all updated readouts and endpoint bounds."""
from pathlib import Path
import json,hashlib,numpy as np,torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round129';B=R/'crossplatform/round129';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(h,u,mask,late,res,W):
 x=torch.from_numpy(h)[None]
 for j in [0,2]:x=F.relu(F.linear(x,W[str(j)+'.weight'],W[str(j)+'.bias']))
 q=F.normalize(F.linear(x,W['4.weight'],W['4.bias'])[0]+torch.from_numpy(late),dim=-1);a=q@torch.from_numpy(u).T/.07;m=torch.from_numpy(mask).bool();s=a[:,m].logsumexp(-1)-a[:,~m].logsumexp(-1);return (s.double()+torch.from_numpy(res).double()).softmax(0).numpy()
def collapse(p,mask):return np.r_[p[~mask],p[mask].sum()]
def kl(a,b):return float(np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(b,1e-300)))))
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');cal=rd(H/'SOURCE_CALIBRATION.json');records=[];maximum=0.;n=0;aliases=0;nscope=0;reused=0;violations=0
 for path,hash_ in cal['source_files'].items():assert sha(R/path)==hash_
 source_predictions=0
 for src in ['waymo','drone','quad']:
  d=R/'crossplatform/round124/formal'/src;seal=rd(d/'SEALED.json')
  for path,hash_ in seal['files'].items():
   if path.startswith('samples_PRIVATE/'):
    assert sha(d/path)==hash_;source_predictions+=1
 assert source_predictions==384
 for path,hash_ in rd(H/'SCOPE_SEALED.json')['files'].items():assert sha(B/path)==hash_
 for path,hash_ in rd(H/'experiment/EXECUTION_LOCK.json')['files'].items():assert sha(R/path)==hash_
 for combo,hash_ in mat['seals'].items():
  co,name=combo.split('/');src=name.split('_')[0];D=B/'formal'/co/name;assert sha(D/'SEALED.json')==hash_;seal=rd(D/'SEALED.json')
  for p,h in seal['files'].items():assert sha(D/p)==h
  with np.load(R/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as a:W={k[6:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('match.')}
  for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
   rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));a=np.load(p);probs=dict(zip(a['arm_names'],a['scores']));assert a['scores'].shape==(25,256) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9);assert np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   if rec['protected_alias_fallback']:aliases+=1;assert not rec['scope']['scope_eligible'];continue
   sp=B/'scope_PRIVATE'/co/name/(p.stem+'.npz');scope=np.load(sp);assignment=scope['assignment'];groups=[np.flatnonzero(assignment==i) for i in range(int(assignment.max())+1)];Scheck=(scope['n_A']>=cal['minimum_support_points'])&(scope['n_B']>=cal['minimum_support_points'])&(scope['coverage_B']>=cal['minimum_registered_fraction']);assert np.array_equal(Scheck,scope['S']);Ccheck=np.zeros_like(Scheck)
   if Scheck.sum()>=2:
    for v in ['pA','pB']:
     mass=np.array([scope[v][g].mean() for g in groups])[Scheck];mass=mass/mass.sum();Ccheck[Scheck]|=(mass.max()-mass<=cal['tau'])
   assert np.array_equal(Ccheck,scope['C']);assert rec['scope']['partial_eligible']==bool(Scheck.sum()>=2 and Ccheck.any() and (Scheck&~Ccheck).any())
   if not rec['scope']['scope_eligible']:continue
   nscope+=1;ff=np.load(D/'features_PRIVATE'/(p.stem+'.npz'));state=np.load(D/'states_PRIVATE'/(p.stem+'.npz'))
   with np.load(R/f'crossplatform/round127/formal/{co}/{name}/features_PRIVATE/{p.stem}.npz') as old:late,res,mask=old['fixed_late'],old['fixed_r'],old['mask']
   for setting in ['zero','conditional','partial','partial_guard','wrong_guard','global_scope']:
    for v in ['F','C']:
     expected=read(ff[setting+'_'+v+'_h'],ff[setting+'_'+v+'_u'],mask,late,res,W);error=float(abs(expected-probs[setting+'_'+v]).max());maximum=max(maximum,error);assert error<5e-6,(setting,error)
   assignment=scope['assignment'];S=scope['S'][assignment];ws=scope['wrong_S'][assignment];assert np.sum(scope['S'])==np.sum(scope['wrong_S']) and np.sum(S)==np.sum(ws);assert np.sum(scope['C'])==np.sum(scope['wrong_C']) and scope['C'][assignment].sum()==scope['wrong_C'][assignment].sum();assert not np.any(scope['C']&~scope['S'])
   for kind in ['conditional','partial','partial_guard','wrong_guard']:
    delta=np.zeros(4896,np.float32);steps=rec['traces'].get(kind,[])
    for j,t in enumerate(steps):
     g=state[kind+'_step'+str(j+1)+'_gradient'];norm=np.linalg.norm(g)
     if t['accepted']:delta=(delta-rec['parameter_radius']*g/max(norm,1e-30)*t['fraction']).astype(np.float32)
     assert t['after']<=t['before']+1e-8
     if kind in ['partial_guard','wrong_guard']:
      violations+=int(t['unknown_KL']>cal['unknown_KL_epsilon']+1e-11);assert t['unknown_KL']<=cal['unknown_KL_epsilon']+1e-11
    assert np.max(abs(delta-state[kind]))<1e-6
    qS=ws if kind=='wrong_guard' else S;unknown=kl(collapse(probs['zero_F'],qS),collapse(probs[kind+'_F'],qS))
    if kind in ['partial_guard','wrong_guard']:assert unknown<=cal['unknown_KL_epsilon']+1e-11
    records.append({'cohort':co,'source':src,'target':name.split('_')[-1],'id_PRIVATE':p.stem,'setting':kind,'active':bool(steps),'accepted_steps':sum(t['accepted'] for t in steps),'parameter_L2':float(np.linalg.norm(state[kind])),'unknown_KL':unknown,'constraint_rejected_trials':sum(not q['unknown_budget_pass'] for t in steps for q in t['trials']) if kind in ['partial_guard','wrong_guard'] else 0,'bound_violating_trials':sum(not q['unknown_budget_pass'] for t in steps for q in t['trials']),'Armijo_rejected_trials':sum(not q['Armijo'] for t in steps for q in t['trials']),'all_steps_finite':all(np.isfinite(t['gradient_L2']) for t in steps)})
   if rec['global_same_scope_reused']:
    reused+=1
    with np.load(R/f'crossplatform/round127/formal/{co}/{name}/states_PRIVATE/{p.stem}.npz') as old:assert np.array_equal(state['global_scope'],old['fab_full'])
 assert n==2304 and aliases==154 and nscope==785 and violations==0
 out={'state':'PASS_SEALS_CPU_FULL_READOUT_PARAMETER_STEPS_UNKNOWN_BOUNDS','n':n,'protected_aliases':aliases,'scope_eligible':nscope,'reused_original_global_states':reused,'new_global_reference_refits':nscope-reused,'CPU_probability_max_error':maximum,'unknown_endpoint_violations':violations,'source_calibration_file_hashes_checked':len(cal['source_files']),'scope_and_unchanged_execution_hashes_checked':True,'task_GT_read':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(H/'PROCESS_ROWS_PRIVATE.json').write_text(json.dumps(records,indent=2)+'\n');print(out)
if __name__=='__main__':main()
