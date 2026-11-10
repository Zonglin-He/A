"""Independent AB-only gradient/one-step replay, then all C/direct/fusion readouts."""
import sys,time,torch
from common66 import *
sys.path.insert(0,str(ROOT/'research/round58'));import independent58 as I
sys.path.insert(0,str(ROOT/'research/round65'));import audit65 as M
ARMS=ARMS+['R59_v1_E_same_v2_pool','C_v1_E_same_v2_pool']

def abdata(ab,e):
 d={v:(torch.tensor(ab[v+'_h'])[None],torch.tensor(ab[v+'_tokens'])[None],torch.tensor(ab[v+'_token_mask']).bool()) for v in 'AB'};d.update(base=torch.tensor(e['base']).double(),B_W=torch.tensor(e['B_W']).double(),B_mass=float(e['B_fixed_mass']));return d

def objective(params,d,e,prior):
 S=torch.tensor(e['positive_query_mask']);C=torch.tensor(e['tested_query_mask']);setloss=[];keep=[]
 for v in 'AB':
  p=I.prediction(params,d,v);setloss.append(-torch.log(p[S].sum().clamp_min(1e-300)/p[C].sum().clamp_min(1e-300)));q=prior[v];keep.append((q*(q.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum())
 return sum(setloss)/2+sum(keep)/2

def independently_fit(initial,d,e):
 s=I.params(initial,True);optimizer=torch.optim.Adam(list(s.values()),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0);prior={v:I.prediction(s,d,v).detach().clone() for v in 'AB'};old={k:v.detach().clone() for k,v in s.items()};beforehash=I.digest(s);loss=objective(s,d,e,prior);before=float(loss.detach());optimizer.zero_grad(set_to_none=True);loss.backward();grad=float(torch.sqrt(sum(v.grad.square().sum() for v in s.values())));torch.nn.utils.clip_grad_norm_(list(s.values()),1.);optimizer.step();proposal={k:v.detach().clone() for k,v in s.items()};trials=[];accepted=False;scale=0.
 with torch.no_grad():
  for k in range(9):
   alpha=2.**(-k)
   for name,val in s.items():val.copy_(proposal[name] if k==0 else old[name]+alpha*(proposal[name]-old[name]))
   value=float(objective(s,d,e,prior));trials.append(value)
   if np.isfinite(value) and value<=before-max(1e-12,1e-6*abs(before)):accepted=True;scale=alpha;break
  if not accepted:
   for name,val in s.items():val.copy_(old[name])
  after=float(objective(s,d,e,prior));step=float(torch.sqrt(sum((val-old[name]).square().sum() for name,val in s.items())))
 return s,{'accepted':accepted,'scale':scale,'loss_before':before,'loss_after':after,'trials':trials,'gradient_norm':grad,'step_L2':step,'before_hash':beforehash,'after_hash':I.digest(s),'C_used':False}

def reweight(base,e,name='E'):
 out=base.copy();pool=e['tested_query_mask']
 if not pool.any():return out
 offset=np.zeros(256)
 for j,mask in enumerate(e['members']):
  if e['stable'][j]:offset[mask]=e[name][j]
 x=np.log(base[pool].clip(1e-300))+offset[pool];w=np.exp(x-x.max());w/=w.sum();out[pool]=float(base[pool].sum())*w;assert np.array_equal(out[~pool],base[~pool]);return out

def project_set(base,e):
 out=base.copy();pool=e['tested_query_mask'];S=e['positive_query_mask']
 if not pool.any() or not S.any() or np.array_equal(pool,S):return out
 out[pool]=0;out[S]=base[pool].sum()*base[S]/max(float(base[S].sum()),1e-300);assert np.array_equal(out[~pool],base[~pool]);return out

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/AUDIT_QUERY_PLAN_LOCK_v2.json');pin(p);cert=read(H/'independent_query_evidence_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['evidence_seal_sha256']==sha(ROOT/p['evidence_root']/'SEALED.json');rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');fitroot=ROOT/p['fit_root'];fs=read(fitroot/'SEALED.json');pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');assert ps['n']==336 and ps['arm_names']==ARMS;torch.set_num_threads(4);start=time.perf_counter();graderr=0.;prob=0.;fits={};accepted=0;statuses=read(pred/'READOUT_STATUS_PRIVATE.json')
 for rel,h in ps['files'].items():assert sha(pred/rel)==h
 # No C cache input appears in this entire fitting loop.
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(fitroot/rel)==fs['files'][rel];e=load(ROOT/p['evidence_root']/rel);ab=load(ROOT/p['AB_root']/rel);actual=read(fitroot/'cases_PRIVATE'/(sid+'.json'));weights=load(fitroot/rel)
  if bool(e['update_allowed']):
   fitted,tr=independently_fit(initial,abdata(ab,e),e);accepted+=int(tr['accepted'])
   for name in initial:assert np.array_equal(fitted[name].detach().numpy(),weights[name]),('fit_weights',i,name)
   for k,v in tr.items():
    if type(v) is float:graderr=max(graderr,abs(v-actual[k]));assert abs(v-actual[k])<1e-11,(i,k)
    elif isinstance(v,list):assert len(v)==len(actual[k]) and max([abs(a-b) for a,b in zip(v,actual[k])]+[0])<1e-11
    else:assert v==actual[k],(i,k)
  else:
   assert not actual['accepted'] and actual['step_L2']==0 and actual['C_used'] is False
   for name in initial:assert np.array_equal(initial[name],weights[name])
  fits[sid]=weights
 # C is opened only after all336 fits are independently reproduced from AB.
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';e=load(ROOT/p['evidence_root']/rel);ab=load(ROOT/p['AB_root']/rel);c=load(ROOT/'crossplatform/round57/capture336_attempt001'/rel);g=load(ROOT/p['association_root']/rel);old=load(ROOT/p['r59_root']/rel);z=load(pred/rel);d=I.inputs(ab,c,g);params=I.params(fits[sid]);source=I.params(initial);expected={a:old['scores'][list(old['arm_names']).index(b)] for a,b in [('source','source'),('frozen_ABC','frozen_ABC'),('R59','direct_all')]}
  with torch.no_grad():expected['C_source']=I.prediction(source,d,'C').numpy();expected['C_tta1']=I.prediction(params,d,'C').numpy()
  for stem,base in [('ABC',expected['frozen_ABC']),('R59',expected['R59']),('C',expected['C_source'])]:expected[stem+'_E_direct']=reweight(base,e)
  for stem,base in [('R59',expected['R59']),('C',expected['C_source'])]:expected[stem+'_set_direct']=project_set(base,e)
  v1=load(ROOT/p['original_evidence_root']/rel);same=dict(e,E=v1['E']);expected['R59_v1_E_same_v2_pool']=reweight(expected['R59'],same);expected['C_v1_E_same_v2_pool']=reweight(expected['C_source'],same)
  for arm,key in [('R59_full_only','full_only'),('R59_category_only','category_only'),('R59_control_only','control_only')]:expected[arm]=reweight(expected['R59'],e,key)
  zz={v+'_'+k:(c[k] if v=='C' else ab[v+'_'+k]) for v in 'ABC' for k in ['h','tokens','token_mask']};expected['ABC_tta1'],expected['R59_tta1']=M.readouts(zz,g,fits[sid]);assert z['arm_names'].tolist()==ARMS and np.array_equal(z['source_boxes'],ab['A_boxes']) and np.array_equal(z['source_scores'],ab['A_scores']) and np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);actualfit=read(fitroot/'cases_PRIVATE'/(sid+'.json'));ss={}
  for j,arm in enumerate(ARMS):
   error=float(abs(z['scores'][j]-expected[arm]).max());prob=max(prob,error);assert error<1e-11 and int(expected[arm].argmax())==z['queries'][j],('readout',i,arm,error);ss[arm]={'unknown':False,'fallback':bool((arm.endswith('_direct') or arm in ['R59_full_only','R59_category_only','R59_control_only']) and not e['tested_query_mask'].any()),'no_parameter_update':not (arm.endswith('tta1') and actualfit['step_L2']>0)}
  assert statuses[i]['id']==sid and statuses[i]['arm_statuses']==ss
 out=H/'independent_matched_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R66_INDEPENDENT_EVIDENCE_AB_ONLY_GRADIENT_AND_ALL_DIRECT_C_READOUTS','n':336,'arm_names':ARMS,'source_full_forward_replays':cert['source_full_forward_replays'],'independent_region_encoder_decoder_replays':cert['independent_encoder_decoder_replays'],'AB_eligible_fits':read(fitroot/'COSTS.json')['counts']['eligible'],'AB_accepted_updates':accepted,'all_fitted_parameter_arrays_bitexact':True,'AB_only_objective_and_step_acceptance_reproduced_before_C_access':True,'max_gradient_trace_error':graderr,'max_probability_error':prob,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_no_update_rows_retained':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'evidence_audit_sha256':sha(H/'independent_query_evidence_audit_attempt001/AUDIT_RESULT.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',result);print(result)
if __name__=='__main__':main()
