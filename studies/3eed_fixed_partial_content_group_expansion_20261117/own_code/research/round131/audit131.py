"""Independent sealed input/group/step and CPU matching audit; no task GT."""
from pathlib import Path
import json,hashlib,sys,numpy as np,torch
R=Path(__file__).resolve().parents[2];H=R/'research/round131';B=R/'crossplatform/round131';sys.path.insert(0,str(R/'research/round129'));import audit129 as A129
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');lock=rd(H/'experiment/EXECUTION_LOCK.json');cal=rd(R/'research/round129/SOURCE_CALIBRATION.json');reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');index={(e['source']+'_to_'+e['target'],e['row']['id']):e for e in reg};n=aliases=active=effective=0;maximum=0.;process=[]
 for path,h in (lock['files']|lock['dependencies']).items():assert sha(R/path)==h
 model_lock=rd(H/'experiment/MODEL_CODE_LOCK.json');assert sha(H/'experiment/MODEL_CODE_LOCK.json')==lock['model_code_lock_sha256'];assert sha(B/'registration_PRIVATE/SEALED.json')==model_lock['registration_seal_sha256'];assert sha(H/'experiment/INPUT_LOCK.json')==model_lock['input_lock_sha256'];assert sha(H/'ENGINEERING.json')==lock['engineering_sha256']
 for path,h in rd(H/'experiment/EVALUATION_CODE_LOCK.json')['files'].items():assert sha(H/path)==h
 inp=rd(H/'experiment/INPUT_LOCK.json');assert sha(H/inp.get('selection_code_path','select131.py'))==inp['selection_code_sha256']
 for path,h in inp['files'].items():assert sha(H/path)==h
 for path,h in rd(B/'registration_PRIVATE/SEALED.json')['files'].items():assert sha(B/'registration_PRIVATE'/path)==h
 for name,h in mat['seals'].items():
  D=B/'formal'/name;src=name.split('_')[0];assert sha(D/'SEALED.json')==h
  for path,h in rd(D/'SEALED.json')['files'].items():assert sha(D/path)==h
  with np.load(R/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as f:W={k[6:]:torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('match.')}
  for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
   e=index[(name,p.stem)];a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));assert a['scores'].shape==(4,256) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9) and np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
   with np.load(R/'crossplatform/round97'/name/'samples_PRIVATE'/(p.stem+'.npz')) as old:
    assert np.array_equal(a['source_boxes'],old['source_boxes'])
    if e['protected_alias_fallback']:
     aliases+=1;expected=dict(zip(old['arm_names'],old['scores']))['prepared_frozen'];assert all(np.array_equal(v,expected) for v in ps.values());assert not (D/'features_PRIVATE'/(p.stem+'.npz')).exists() and not (D/'prefix_PRIVATE'/(p.stem+'.npz')).exists();continue
   ff=np.load(D/'features_PRIVATE'/(p.stem+'.npz'));pre=np.load(D/'prefix_PRIVATE'/(p.stem+'.npz'));ss=np.load(D/'states_PRIVATE'/(p.stem+'.npz'));assert rec['original_source_probability_error']<1e-7
   for tag,key in [('teacher_A','pA'),('teacher_B','pB'),('zero_F','zero_F'),('partial_F','partial_F'),('wrong_base_F','wrong_base_F')]:
    expected=A129.read(ff[tag+'_h'],ff[tag+'_u'],ff['mask'],ff['fixed_late'],ff['fixed_r'],W);actual=ff[key] if key in ['pA','pB'] else ps[key];error=float(abs(expected-actual).max());maximum=max(maximum,error);assert error<5e-6,(name,p.stem,tag,error)
   provider=rec['source_A_context'];assert sha(R/provider['A_late_path'])==provider['late_file_sha256'] and sha(R/provider['A_r_path'])==provider['r_file_sha256']
   with np.load(R/provider['A_late_path']) as f:assert np.array_equal(ff['fixed_late'],f['F_late'])
   with np.load(R/provider['A_r_path']) as f:assert np.array_equal(ff['fixed_r'],f['r0'])
   # Unique support-location counts, independently from stored counts.
   unique,inv=np.unique(pre['xyz'][0],axis=0,return_inverse=True);valid=np.zeros(len(unique),bool);np.logical_or.at(valid,inv,ff['valid_B']);boxes=a['source_boxes'];inside=(abs(unique[None]-boxes[:,None,:3])<=np.maximum(boxes[:,None,3:6],1e-6)/2).all(2);assignment=ff['assignment'];groups=[np.flatnonzero(assignment==j) for j in range(int(assignment.max())+1)];support=np.array([inside[g].any(0) for g in groups]);na=support.sum(1);nb=(support&valid[None]).sum(1);assert np.array_equal(na,ff['n_A']) and np.array_equal(nb,ff['n_B']);S=(na>=4)&(nb>=4)&(nb/np.maximum(na,1)>=.5);assert np.array_equal(S,ff['S']);C=np.zeros_like(S)
   if S.sum()>=2:
    for tag in ['pA','pB']:
     pp=np.array([ff[tag][g].mean() for g in groups])[S];pp/=pp.sum();C[S]|=pp.max()-pp<=cal['tau']
   assert np.array_equal(C,ff['C']);Cw=ff['wrong_C'];sizes=np.bincount(assignment);assert not (Cw&~S).any() and Cw.sum()==C.sum() and Cw[assignment].sum()==C[assignment].sum()
   for size in set(sizes[S]):assert Cw[S&(sizes==size)].sum()==C[S&(sizes==size)].sum()
   isactive=bool(rec['B_available'] and S.sum()>=2 and (S&~C).any());assert rec['partial_eligible']==isactive;active+=isactive;effective+=bool(isactive and (C!=Cw).any());assert rec['effective_shuffle']==bool(isactive and (C!=Cw).any())
   initial=ps['zero_F'];qs=S[assignment];qc=C[assignment];direct=initial.copy()
   if S.sum()>=2 and qc.any() and (qs&~qc).any():mass=direct[qs].sum();direct[qs]=0;direct[qc]=initial[qc]*mass/initial[qc].sum();direct/=direct.sum()
   assert abs(direct-ps['direct_partial_F']).max()<1e-12
   for setting in ['partial','wrong_base']:
    trace=rec['traces'][setting];delta=np.zeros(4896,np.float32);reuse=setting=='wrong_base' and rec['identity_control'];gname='partial' if reuse else setting
    for j,t in enumerate(trace):
     g=ss[gname+'_step'+str(j+1)+'_gradient'];norm=np.linalg.norm(g)
     if t['accepted']:delta=(delta-rec['parameter_radius']*g/max(norm,1e-30)*t['fraction']).astype(np.float32)
     assert t['after']<=t['before']+1e-8
    assert abs(delta-ss[setting]).max()<1e-6
    if not isactive:assert np.all(ss[setting]==0) and np.array_equal(ps[setting+'_F'],initial)
    process.append({'source':src,'target':e['target'],'id_PRIVATE':p.stem,'setting':setting,'active':isactive,'parameter_L2':float(np.linalg.norm(ss[setting])),'accepted_steps':sum(t['accepted'] for t in trace),'Armijo_rejected_trials':sum(not q['Armijo'] for t in trace for q in t['trials']),'task_before':trace[0]['before'] if trace else 0.,'task_after':trace[-1]['task_after'] if trace else 0.})
 assert n==len(reg)==1798 and aliases==0
 out={'status':'PASS_INPUT_SEALS_NEW_SOURCE_BRIDGE_SCOPE_LABELS_STEPS_CPU_READOUT','n':n,'protected_aliases':aliases,'neural_inputs':n-aliases,'active_inputs':active,'effective_shuffles':effective,'identity_controls':active-effective,'CPU_probability_max_error':maximum,'GT_read':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(H/'PROCESS_ROWS_PRIVATE.json').write_text(json.dumps(process,indent=2)+'\n');print(out)
if __name__=='__main__':main()
