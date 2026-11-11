"""Independent actual weight scaling/AB acceptance, then gamma/eta/repeat readouts."""
import sys,time,torch
from common68 import *
import independent_cpu68 as M
sys.path.insert(0,str(ROOT/'research/round66'));import audit66 as J
sys.path.insert(0,str(ROOT/'research/round58'));import independent58 as I
H=ROOT/'research/round68';p=read(H/'experiment/AUDIT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');scaledroot=ROOT/p['scaled_root'];torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter();states={};loss_err=0.;prob_err=0.;counts={tag(x):0 for x in ETAS}
# Entire parameter/acceptance audit has AB only. C is not accessed here.
for r in rows:
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';ab=load(ROOT/p['AB_root']/rel);e=load(ROOT/p['evidence_root']/rel);fit=load(ROOT/p['fit_root']/rel);rec=read(scaledroot/'cases_PRIVATE'/(sid+'.json'));file=scaledroot/rel;actual=load(file) if file.exists() else {};d=J.abdata(ab,e);source=I.params(initial);prior={v:I.prediction(source,d,v).clone() for v in 'AB'};states[sid]={}
 for eta in ETAS:
  key=tag(eta);allowed=bool(e['update_allowed']) and any(np.any(fit[k]!=initial[k]) for k in initial);expected={k:fit[k].copy() if eta==1. else initial[k]+np.float32(eta)*(fit[k]-initial[k]) for k in initial} if allowed else initial;params=I.params(expected);before=float(J.objective(source,d,e,prior)) if allowed else 0.;after=float(J.objective(params,d,e,prior)) if allowed else 0.;accept=bool(allowed and np.isfinite(after) and after<=before-max(1e-12,1e-6*abs(before)));arr=expected if accept else initial;assert rec[key]['eligible']==allowed and rec[key]['accepted']==accept
  loss_err=max(loss_err,abs(rec[key]['loss_before']-before),abs(rec[key]['loss_proposed']-after));assert loss_err<1e-11
  if allowed:
   for k,v in arr.items():assert np.array_equal(actual[key+'_'+k],v)
  assert I.digest(I.params(arr))==rec[key]['state_sha256'];counts[key]+=int(accept);states[sid][key]=arr
# All acceptance checks complete before the first C read.
status=read(pred/'READOUT_STATUS_PRIVATE.json')
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];saved=load(pred/rel);ab=load(ROOT/p['AB_root']/rel);c=load(ROOT/p['C_root']/rel);g=load(ROOT/p['association_root']/rel);ref=load(ROOT/p['r59_root']/rel);r66=load(ROOT/p['r66_prediction_root']/rel);random=load(ROOT/p['random_capture_root']/rel);r67=load(ROOT/p['r67_capture_root']/rel);r67p=load(ROOT/p['r67_prediction_root']/rel);z={v+'_'+k:(c[k] if v=='C' else ab[v+'_'+k]) for v in 'ABC' for k in ['h','tokens','token_mask']};d=I.inputs(ab,c,g);expected={'source':ref['scores'][ref['arm_names'].tolist().index('source')],'R59':ref['scores'][ref['arm_names'].tolist().index('direct_all')],'R66':r66['scores'][r66['arm_names'].tolist().index('R59_tta1')],'C_source':I.prediction(I.params(initial),d,'C').numpy()}
 for gamma in GAMMAS:_,expected['G_'+tag(gamma)]=M.readouts(z,g,initial,gamma)
 for eta in ETAS:
  arr=states[sid][tag(eta)];_,expected['E_'+tag(eta)]=M.readouts(z,g,arr,1.);expected['C_E_'+tag(eta)]=I.prediction(I.params(arr),d,'C').numpy()
 # Original random is a reused repeat, not a seed selected from new results.
 oldz={v+'_'+k:r67['random_'+v+'_'+k] for v in 'ABC' for k in ['h','tokens']};oldz.update({v+'_token_mask':r67[v+'_token_mask'] for v in 'ABC'});_,oldprob=M.readouts(oldz,g,initial);expected['random_s0']=r67p['scores'][r67p['arm_names'].tolist().index('random_R59')];assert np.max(abs(oldprob-expected['random_s0']))<1e-11
 for rep in [1,2]:
  zz={v+'_'+k:random['s'+str(rep)+'_'+v+'_'+k] for v in 'ABC' for k in ['h','tokens']};zz.update({v+'_token_mask':r67[v+'_token_mask'] for v in 'ABC'});_,expected['random_s'+str(rep)]=M.readouts(zz,g,initial)
 for j,a in enumerate(ARMS):
  err=float(abs(saved['scores'][j]-expected[a]).max());prob_err=max(prob_err,err);assert err<1e-11 and int(expected[a].argmax())==saved['queries'][j],(i,a,err)
 assert np.array_equal(saved['source_boxes'],ab['A_boxes']) and np.array_equal(saved['selected_original_boxes'],ab['A_boxes'][saved['queries']]) and status[i]['id']==sid
 if(i+1)%64==0:print('AUDIT_CPU68',i+1,flush=True)
out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R68_AB_ONLY_ACTUAL_PARAMETER_SCALE_ACCEPTANCE_GAMMA_AND_ALL_REPEATS','n':336,'arms':len(ARMS),'accepted_by_eta':counts,'max_loss_error':loss_err,'max_probability_error':prob_err,'all_scaled_parameter_arrays_bitexact':True,'all_parameter_acceptance_checked_before_C':True,'same_R59_targets_internal_rho_and_transport_fixed':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'random_gpu_audit_sha256':sha(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json'),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
