"""Independent sealed-state/CPU-readout/control-cardinality audit, no task GT."""
from pathlib import Path
import json,sys,hashlib,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round130';B=R/'crossplatform/round130';sys.path.insert(0,str(R/'research/round129'));import audit129 as A129
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
REF={k:k for k in ['original_source','prepared_A','legacy_ABC_context','zero_F','zero_C','direct_partial_F','direct_partial_C','partial_F','partial_C']}|{'R129_guard_F':'partial_guard_F','R129_guard_C':'partial_guard_C'}
FACTORS={'proper_half':.5,'proper_double':2.,'wrong_half':.5,'wrong_base':1.,'wrong_double':2.}
def main():
 torch.set_num_threads(4);design=rd(H/'experiment/DESIGN_LOCK.json');mat=rd(B/'MATRIX_SEALED.json');n=active=effective=alias=0;maximum=0.;process=[]
 for path,hash_ in (design['files']|design['dependencies']).items():assert sha(R/path)==hash_
 assert sha(H/'CONTROL_SEALED.json')==design['control_seal_sha256']
 for path,hash_ in rd(H/'CONTROL_SEALED.json')['files'].items():assert sha(B/path)==hash_
 for combo,hash_ in mat['seals'].items():
  co,name=combo.split('/');src=name.split('_')[0];D=B/'formal'/co/name;assert sha(D/'SEALED.json')==hash_
  for path,hash_ in rd(D/'SEALED.json')['files'].items():assert sha(D/path)==hash_
  with np.load(R/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as a:W={k[6:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('match.')}
  for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
   rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));meta=rec['control'];a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));assert a['scores'].shape==(23,256) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9) and np.array_equal(a['queries'],a['scores'].argmax(1));n+=1;oldroot=R/'crossplatform/round129/formal'/co/name
   with np.load(oldroot/'samples_PRIVATE'/(p.stem+'.npz')) as f:oldp=dict(zip(f['arm_names'],f['scores']));assert np.array_equal(a['source_boxes'],f['source_boxes']);assert all(np.array_equal(ps[k],oldp[v]) for k,v in REF.items())
   if meta['protected_alias_fallback']:alias+=1;assert not meta['partial_eligible'];continue
   ctrl=np.load(B/'control_PRIVATE'/co/name/(p.stem+'.npz'));assignment=ctrl['assignment'];S,C,Cw=ctrl['S'],ctrl['C'],ctrl['wrong_C'];size=np.bincount(assignment);assert not (Cw&~S).any() and Cw.sum()==C.sum() and Cw[assignment].sum()==C[assignment].sum();original=R/'crossplatform/round129/scope_PRIVATE'/co/name/(p.stem+'.npz');assert sha(original)==meta['R129_scope_sha256']
   with np.load(original) as f:assert all(np.array_equal(ctrl[k],f[k]) for k in ['S','C','assignment'])
   for nn in set(size[S]):assert Cw[S&(size==nn)].sum()==C[S&(size==nn)].sum()
   if not meta['partial_eligible']:
    for k in mat['new_arms']:assert np.array_equal(ps[k],ps['zero_'+k.rsplit('_',1)[1]])
    continue
   active+=1;effective+=meta['effective_shuffle'];assert meta['effective_shuffle']==bool((C!=Cw).any());ff=np.load(D/'features_PRIVATE'/(p.stem+'.npz'));state=np.load(D/'states_PRIVATE'/(p.stem+'.npz'));oldstate=np.load(oldroot/'states_PRIVATE'/(p.stem+'.npz'))
   with np.load(R/'crossplatform/round127/formal'/co/name/'features_PRIVATE'/(p.stem+'.npz')) as f:late,res,mask=f['fixed_late'],f['fixed_r'],f['mask']
   for setting,factor in FACTORS.items():
    trace=rec['traces'][setting];reuse=rec['identity_control_reuses_same_update'].get(setting);delta=np.zeros(4896,np.float32)
    for j,t in enumerate(trace):
     g=oldstate['partial_step'+str(j+1)+'_gradient'] if reuse=='partial' else state[(reuse or setting)+'_step'+str(j+1)+'_gradient'];norm=np.linalg.norm(g)
     if t['accepted']:delta=(delta-rec['base_parameter_radius']*factor*g/max(norm,1e-30)*t['fraction']).astype(np.float32)
     assert t['after']<=t['before']+1e-8
    assert abs(delta-state[setting]).max()<1e-6
    if reuse=='partial':assert np.array_equal(state[setting],oldstate['partial'])
    elif reuse:assert np.array_equal(state[setting],state[reuse])
    for view in ['F','C']:
     expected=A129.read(ff[setting+'_'+view+'_h'],ff[setting+'_'+view+'_u'],mask,late,res,W);error=float(abs(expected-ps[setting+'_'+view]).max());maximum=max(maximum,error);assert error<5e-6,(setting,error)
    process.append({'cohort':co,'source':src,'target':name.split('_')[-1],'id_PRIVATE':p.stem,'setting':setting,'factor':factor,'effective_shuffle':meta['effective_shuffle'],'accepted_steps':sum(t['accepted'] for t in trace),'parameter_L2':float(np.linalg.norm(state[setting])),'Armijo_rejected_trials':sum(not a['Armijo'] for t in trace for a in t['trials']),'task_before':trace[0]['before'],'task_after':trace[-1]['task_after'],'feature_keep_after':trace[-1]['feature_keep_after']})
   for view in ['F','C']:
    pp=ps['zero_'+view].copy();qs=S[assignment];qc=Cw[assignment];mass=pp[qs].sum();target=pp.copy();target[qs]=0;target[qc]=pp[qc]*mass/pp[qc].sum();target/=target.sum();assert abs(target-ps['direct_wrong_'+view]).max()<1e-12
 assert n==2304 and active==257 and effective==252 and alias==154
 out={'status':'PASS_SEALS_EXACT_REUSE_FIXED_S_CARDINALITY_PARAMETER_STEPS_CPU_READOUT','n':n,'active_inputs':active,'effective_shuffles':effective,'identity_controls':active-effective,'protected_aliases':alias,'CPU_probability_max_error':maximum,'all_references_bitexact_R129':True,'new_task_GT_read':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(H/'PROCESS_ROWS_PRIVATE.json').write_text(json.dumps(process,indent=2)+'\n');print(out)
if __name__=='__main__':main()
