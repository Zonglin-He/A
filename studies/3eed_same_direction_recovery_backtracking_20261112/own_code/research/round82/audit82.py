"""Independent native R80 direction, states, all new search trials and readout audit."""
from core82 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H82/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H82/'experiment/AUDIT_CODE_LOCK.json'));P=B82/'prediction2640_attempt001';ss=read(P/'SEALED.json')
for f,h in ss['files'].items():assert sha(P/f)==h
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);start=time.perf_counter();maxp=maxstate=maxobj=maxheld=0.;states=trials=reused=0;endpoints=[]
for i,r in enumerate(allrows80()):
 sid=r['id']
 if sid in DEV_IDS:z0,z,d=item80(r)
 else:
  with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as archive:z0={k:archive['A_'+k].copy() for k in EXT_KEYS}
  z=tensor_prefix(z0);d=tensor_data(load(ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz')),r)
 pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));records=read(P/'records_PRIVATE'/(sid+'.json'));old=load(B80/'prediction2640_attempt001/samples_PRIVATE'/(sid+'.npz'));refs=dict(zip(old['arm_names'],old['scores']))
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');assert np.max(abs(refp.cpu().numpy()-refs['frozen_bi']))<1e-10;hp,hi,hw=holdout81(r,d);h0=float(aux82(m,z,d,hp,hi,hw))
 for wrong in [False,True]:
  arm='bt_wrong' if wrong else 'bt_correct';rec=records[arm];g,raw,ini=gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong);assert ini['objective_before']==rec['objective_before'] and float(g.norm())==rec['gradient_norm'];direction=raw/.5;trace=rec['attempts'];assert [t['alpha'] for t in trace]==GRID82[:len(trace)];assert rec['accepted']==bool(trace and trace[-1]['accepted']);assert not any(t['accepted'] for t in trace[:-1]);assert rec['alpha']==(trace[-1]['alpha'] if rec['accepted'] else 0.)
  if rec['reused_completed_R81_correct_search']:
   assert not wrong;oldP=old81(r);oldseal=read(oldP/'SEALED.json');fname='records_PRIVATE/'+sid+'.json';assert sha(oldP/fname)==oldseal['files'][fname];orec=read(oldP/fname)['all__ALL'];assert trace==[{**t,'reused_R81_trace':True} for t in orec['attempts']];assert rec['alpha']==orec['alpha'];assert all(t['objective']-t['bound']>1e-7 for t in trace[:-1]);reused+=1
  else:
   for t in trace:
    delta=t['alpha']*direction;set_step(m,S,delta)
    with torch.no_grad():L,au,ke=objective79(m,base,z,d,refp,refq,pm,im,ws,wrong)
    bound=ini['objective_before']+1e-4*float(g.double()@delta.double());assert abs(float(L)-t['objective'])<1e-8 and abs(bound-t['bound'])<1e-12;assert bool(np.isfinite(float(L)) and float(L)<=bound+1e-10)==t['accepted'];trials+=1
   restore(m,S)
  delta=rec['alpha']*direction;set_step(m,S,delta);expected=patch82(m);patch=load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz'));err=max(float(abs(expected[n]-patch[n]).max()) for n in patch);maxstate=max(maxstate,err);assert err<=1e-7;put_patch(m,patch)
  with torch.no_grad():
   pp,q,_=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;aux=aux82(m,z,d,pm,im,ws,wrong);L=aux.double()+keep;maxobj=max(maxobj,abs(float(L)-rec['objective_after']));assert maxobj<1e-7
   if rec['accepted']:assert float(L)<=ini['objective_before']+1e-10 and float(aux)<=ini['auxiliary_before']+1e-8
   actual=torch.cat([(v-torch.as_tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);assert abs(float(actual.norm())-rec['parameter_step_L2'])<1e-10;held=float(aux82(m,z,d,hp,hi,hw));maxheld=max(maxheld,abs(held-rec['heldout_true_auxiliary_after']),abs(h0-rec['heldout_true_auxiliary_before']));assert maxheld<1e-10;maxp=max(maxp,float(abs(pp.cpu().numpy()-saved['scores'][ARMS82.index(arm)]).max()));assert maxp<1e-10
   for n,v in m.state_dict().items():
    if n not in patch:assert np.array_equal(v.cpu().numpy(),S[n])
  restore(m,S)
  # Raw updates and third-mask values are checked independently from the new
  # accepted states. Raw wrong-binding still evaluates truth on the third mask.
  rp=rawpatch82(r,wrong);put_patch(m,rp)
  with torch.no_grad():hh=float(aux82(m,z,d,hp,hi,hw));tt=float(aux82(m,z,d,pm,im,ws));assert hh==rec['raw_heldout_true_auxiliary_after'] and tt==rec['raw_true_training_auxiliary_after']
  restore(m,S);states+=1;endpoints.append({'row':'X'+str(i+1).zfill(5),'platform':r['platform'],'arm':arm,**{k:v for k,v in rec.items() if k not in ['attempts','raw_record']}})
 assert np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']])
 for a,k in [('source','source'),('frozen_bi','frozen_bi'),('frozen_direct3','frozen_direct3'),('R80_raw_correct','uniform_sgd1'),('R80_raw_wrong','uniform_wrong_sgd1'),('frozen_ABC','frozen_ABC'),('R59','R59')]:assert np.array_equal(saved['scores'][ARMS82.index(a)],refs[k])
 if(i+1)%128==0:print('AUDIT82',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;O=H82/'independent_native_audit_attempt001';O.mkdir();result={'state':'PASS_R82_ALL_FIXED_DIRECTION_STATES_NEW_SEARCH_TRIALS_HELDOUT_AND_NATIVE_READOUTS','n':2640,'arm_names':ARMS82,'states':states,'all_new_search_objectives_recomputed':True,'new_search_trials_recomputed':trials,'completed_R81_correct_searches_reused':reused,'reused_search_trace_covered_by_prior_independent_R81_audit':True,'max_probability_error':maxp,'max_parameter_direction_error':maxstate,'max_final_objective_error':maxobj,'max_heldout_error':maxheld,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'GT_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',result);dump(H82/'ALL_ENDPOINT_SCALAR_ROWS.json',endpoints);print(result)
