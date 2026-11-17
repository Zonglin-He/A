"""Independent seals, registration policy, source threshold, sets, steps and CPU final reader."""
from common132 import *
import numpy as np,torch,sys,collections
sys.path.insert(0,str(R132/'research/round129'));from audit129 import read
sys.path.insert(0,str(R132/'research/round109'));from register109 import geometric,TC
def main():
 basecheck();torch.set_num_threads(4);mat=rd(B132/'MATRIX_SEALED.json');reg=rd(R132/'research/round131/INPUT_REGISTRY_PRIVATE.json');index={(e['source']+'_to_'+e['target'],e['row']['id']):e for e in reg};cal=rd(H132/'SOURCE_LOG_CALIBRATION.json');tau=rd(R132/'research/round129/SOURCE_CALIBRATION.json')['tau'];process=[];n=0;maximum=0.;counters=collections.Counter();old_diag={(r['source']+'_to_'+r['target'],r['id_PRIVATE']):r for r in rd(H132/'DIAGNOSIS_WITHOUT_GT_PRIVATE.json')}
 for lockname in ['REGISTRATION_CODE_LOCK','SOURCE_EXECUTION_LOCK','TARGET_EXECUTION_LOCK','FORMAL_LOCK','EVALUATION_CODE_LOCK']:
  for p,h in rd(H132/'experiment'/(lockname+'.json'))['files'].items():assert sha(R132/p)==h,p
 for p,h in rd(B132/'REGISTRATION_SEALED.json')['files'].items():assert sha(B132/p)==h
 for selection in (B132/'selections_PRIVATE').glob('*.json'):
  for at in rd(selection)['attempts']:
   r=rd(R132/at['path']);f=geometric(r.get('forward',{}));b=geometric(r.get('reverse',{}));cy=r.get('cycle',{});c=bool(cy.get('n',0)>=20 and cy.get('median_m',float('inf'))<=TC.maximum_holdout_median_m and cy.get('p90_m',float('inf'))<=TC.maximum_holdout_p90_m);assert r['geometry_cycle_supported']==bool(f and b and c)
 for a in rd(H132/'TARGET_B3_CANDIDATES_PRIVATE.json'):
  sel=rd(B132/'selections_PRIVATE'/('TARGET_'+a['A']['frame_PRIVATE']+'.json'));attempts=sel['attempts'];assert len(attempts)<=3
  for i,r in enumerate(attempts):assert r['rank']==i+1 and r['pair_PRIVATE']==a['candidates'][i]['pair']['pair_PRIVATE'] and sha(R132/r['path'])==r['sha256'];assert not any(x['geometry_cycle_supported'] for x in attempts[:-1])
  assert sel['selected_rank']==(attempts[-1]['rank'] if attempts and attempts[-1]['geometry_cycle_supported'] else None)
 for name,h in mat['seals'].items():
  D=B132/'formal'/name;assert sha(D/'SEALED.json')==h
  for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
  src=name.split('_')[0]
  with np.load(R132/f'crossplatform/round127/{src}_AUDIT_MODEL_PRIVATE.npz') as a:W={k[6:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('match.')}
  for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
   e=index[(name,p.stem)];a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));meta=rd(D/'records_PRIVATE'/(p.stem+'.json'));oldroot=R132/'crossplatform/round131/formal'/name;old=np.load(oldroot/'samples_PRIVATE'/(p.stem+'.npz'));ref=dict(zip(['zero','direct','partial','wrong'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']);assert a['scores'].shape==(12,256) and np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9) and np.array_equal(a['queries'],a['scores'].argmax(1));assert np.array_equal(a['scores'][:4],old['scores']);n+=1
   diag=old_diag[(name,p.stem)];oldrec=rd(oldroot/'records_PRIVATE'/(p.stem+'.json'));q0=int(ref['zero'].argmax());qp=int(ref['partial'].argmax());assert diag['query_changed']==bool(q0!=qp);assert diag['exact_box_changed']==bool(not np.array_equal(old['source_boxes'][q0],old['source_boxes'][qp]));expected_reason='no_usable_B' if not oldrec['B_available'] else ('actual_update' if oldrec['partial_eligible'] else 'C_equals_S');assert diag['reason']==expected_reason
   with np.load(oldroot/'states_PRIVATE'/(p.stem+'.npz')) as os:assert abs(np.linalg.norm(os['partial'])-diag['parameter_L2'])<1e-10
   counters['old_query_switches']+=bool(q0!=qp)
   assert np.array_equal(ps['LOG_zero'],ref['zero'])
   for kind in ['B3','LOG']:
    rec=meta[kind]
    if rec['reference_original']:
     assert all(np.array_equal(ps[kind+'_'+k],ref[k]) for k in ref);assert not (D/'features_PRIVATE'/(p.stem+'_'+kind+'.npz')).exists();counters[kind+'_reference']+=1;continue
    counters[kind+'_new']+=1;f=np.load(D/'features_PRIVATE'/(p.stem+'_'+kind+'.npz'));ss=np.load(D/'states_PRIVATE'/(p.stem+'_'+kind+'.npz'));pre=np.load(oldroot/'prefix_PRIVATE'/(p.stem+'.npz'))
    for tag,key in [('A','pA'),('B','pB'),('zero',kind+'_zero'),('partial',kind+'_partial'),('wrong',kind+'_wrong')]:
     pp=read(f[tag+'_h'],f[tag+'_u'],f['mask'],f['fixed_late'],f['fixed_r'],W);actual=f[key] if key in ['pA','pB'] else ps[key];error=float(abs(pp-actual).max());maximum=max(maximum,error);assert error<5e-6,(name,p.stem,kind,tag,error)
    with np.load(oldroot/'features_PRIVATE'/(p.stem+'.npz')) as of:assert np.array_equal(f['fixed_late'],of['fixed_late']) and np.array_equal(f['fixed_r'],of['fixed_r']) and np.array_equal(f['assignment'],of['assignment'])
    xyz,inv=np.unique(pre['xyz'][0],axis=0,return_inverse=True);valid=np.zeros(len(xyz),bool);np.logical_or.at(valid,inv,f['valid_B']);box=a['source_boxes'];inside=(abs(xyz[None]-box[:,None,:3])<=np.maximum(box[:,None,3:6],1e-6)/2).all(2);ass=f['assignment'];groups=[np.flatnonzero(ass==i) for i in range(ass.max()+1)];support=np.array([inside[g].any(0) for g in groups]);na=support.sum(1);nb=(support&valid[None]).sum(1);S=(na>=4)&(nb>=4)&(nb/np.maximum(na,1)>=.5);assert np.array_equal(na,f['n_A']) and np.array_equal(nb,f['n_B']) and np.array_equal(S,f['S']);C=np.zeros_like(S)
    if S.sum()>=2:
     for v in ['A','B']:
      pv=np.array([f['p'+v][g].mean() for g in groups])[S];pv/=pv.sum()
      if kind=='B3':C[S]|=pv.max()-pv<=tau
      else:lg=np.log(np.maximum(pv,1e-300));C[S]|=lg.max()-lg<=cal['kappa']
    assert np.array_equal(C,f['C']);cw=f['wrong_C'];assert not (cw&~S).any() and cw.sum()==C.sum() and cw[ass].sum()==C[ass].sum()
    sizes=np.bincount(ass)
    for size in set(sizes[S]):assert cw[S&(sizes==size)].sum()==C[S&(sizes==size)].sum()
    active=bool(f['valid_B'].any() and S.sum()>=2 and C.any() and (S&~C).any());assert active==rec['partial_eligible'];assert rec['effective_shuffle']==bool(active and not np.array_equal(C,cw))
    qs=S[ass];qc=C[ass];direct=ps[kind+'_zero'].copy()
    if S.sum()>=2 and C.any() and (S&~C).any():mass=direct[qs].sum();direct[qs]=0;direct[qc]=ps[kind+'_zero'][qc]*mass/ps[kind+'_zero'][qc].sum();direct/=direct.sum()
    assert abs(direct-ps[kind+'_direct']).max()<1e-12
    for label in ['partial','wrong']:
     delta=np.zeros(4896,np.float32);trace=rec['traces'][label];gradlabel='partial' if label=='wrong' and rec['identity_control'] else label
     for j,t in enumerate(trace):
      g=ss[gradlabel+'_step'+str(j+1)+'_gradient']
      if t['accepted']:delta=(delta-rec['parameter_radius']*g/max(np.linalg.norm(g),1e-30)*t['fraction']).astype(np.float32)
      assert t['after']<=t['before']+1e-8
     assert abs(delta-ss[label]).max()<1e-6
     if not active:assert not ss[label].any() and np.array_equal(ps[kind+'_'+label],ps[kind+'_zero'])
     process.append({'source':src,'target':e['target'],'id_PRIVATE':p.stem,'kind':kind,'label':label,'active':active,'accepted_steps':sum(t['accepted'] for t in trace),'parameter_L2':float(np.linalg.norm(ss[label])),'task_before':trace[0]['before'] if trace else 0.,'task_after':trace[-1]['task_after'] if trace else 0.})
 assert n==len(reg)==1798;out={'status':'PASS_ALL_FIXED_COHORT_REFERENCES_TWO_SEPARATE_RULES_STEPS_CPU_READOUT','n':n,'arms':12,'counters':dict(counters),'maximum_CPU_probability_error':maximum,'task_GT_read':False,'FINAL_access':False};dump(H132/'NATIVE_AUDIT.json',out);dump(H132/'PROCESS_ROWS_PRIVATE.json',process);print(out)
if __name__=='__main__':main()
