"""All-input evidence/group/geometry/gradient audit, then all-input readouts.
No task labels. GPU source/replay coverage is separately prelocked and reported.
"""
import sys,time,torch
from common69 import *
import independent69 as I
from independent66 import support as independent_point_support
sys.path.insert(0,str(ROOT/'research/round66'));import audit66 as J
sys.path.insert(0,str(ROOT/'research/round58'));import independent58 as T
sys.path.insert(0,str(ROOT/'research/round65'));import audit_readout65 as M
from scipy.special import softmax
H=ROOT/'research/round69';p=read(H/'experiment/AUDIT_CPU_PLAN_LOCK_attempt002.json');pin(p);rows=read(ROOT/p['manifest']);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');initial=load(ROOT/p['initial_head']);torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter();fitcount=0;maxerr=0.;controlcount=0
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';ab=load(ROOT/p['AB_capture_root']/rel);e=load(ROOT/p['evidence_root']/rel);z={v:{k[2:]:val for k,val in ab.items() if k.startswith(v+'_')} for v in 'AB'};raw=np.load(r['pointpath'],allow_pickle=False) if Path(r['pointpath']).suffix=='.npy' else np.fromfile(r['pointpath'],np.float32).reshape(-1,4);xx=raw[np.unique(np.r_[z['A']['point_indices'],z['B']['point_indices']]),:3].astype(np.float32).copy()
 if r['platform']=='drone':xx[:,2]+=1.8
 af,_,_=I.af(z['A']['boxes'].astype(float),z['B']['boxes'].astype(float),xx.astype(float));W=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));base=I.probability(z['A']['query'],z['A']['tokens'],z['A']['token_mask']);pb=I.probability(z['B']['query'],z['B']['tokens'],z['B']['token_mask']);mass=float((W@pb).sum());assert np.array_equal(W,e['B_W']) and np.array_equal(base,e['base']) and mass==float(e['B_fixed_mass']);groups,pool=I.groups_and_pool(z['A']['boxes'].astype(float),xx.astype(float),(base+I.mapped(pb,W,mass,base))/2);members=np.array([np.isin(np.arange(256),groups[k]) for k in pool]);assert np.array_equal(members,e['members'])
 valid=[]
 for vi,v in enumerate('AB'):
  regions=np.array([independent_point_support(z[v]['xyz'][0].astype(float),z['A']['boxes'][m].astype(float)).any(0) for m in members]);assert np.array_equal(regions,e[v+'_region_masks']);controls=I.controls(z[v]['xyz'][0].astype(float),regions,z[v]['query_points_sample_inds']);row=[]
  for j,(mask,ok) in enumerate(controls):
   assert np.array_equal(mask,e[v+'_control_masks'][j]);row.append(ok and bool(ab['category_available']))
   if row[-1]:assert mask.sum()==regions[j].sum() and mask[z[v]['query_points_sample_inds']].sum()==regions[j][z[v]['query_points_sample_inds']].sum();controlcount+=1
  valid.append(row)
 assert np.array_equal(valid,e['valid']);dd=np.zeros_like(e['drops'])
 for vi in range(2):
  for ti in range(2):
   for ki in range(2):
    for j,m in enumerate(members):dd[vi,ti,ki,j]=np.log(max(float(e['baseline'][vi,ti,m].sum()),1e-300))-np.log(max(float(e['perturbed'][vi,ti,ki,j,m].sum()),1e-300))
 assert np.array_equal(dd,e['drops']);ee=(dd[:,0,0]-dd[:,0,1])-(dd[:,1,0]-dd[:,1,1]);pos=np.asarray(valid).all(0)&(ee>1e-6).all(0);neg=np.asarray(valid).all(0)&(ee< -1e-6).all(0);stable=pos|neg;S=members[pos].any(0) if pos.any() else np.zeros(256,bool);C=members[stable].any(0) if stable.any() else np.zeros(256,bool)
 for k,val in [('E_views',ee),('positive',pos),('negative',neg),('tested_query_mask',C),('positive_query_mask',S),('update_allowed',np.array(bool(pos.any() and neg.any())))]:assert np.array_equal(e[k],val),(i,k)
 actual=load(ROOT/p['fit_root']/rel);record=read(ROOT/p['fit_root']/'cases_PRIVATE'/(sid+'.json'))
 if bool(e['update_allowed']):
  with torch.enable_grad():params,rec=J.independently_fit(initial,J.abdata(ab,e),e)
  fitcount+=1
  for k,v in params.items():assert np.array_equal(v.detach().numpy(),actual[k]),('fit',i,k)
  for k in ['accepted','scale','before_hash','after_hash']:assert rec[k]==record[k]
  for k in ['loss_before','loss_after','gradient_norm','step_L2']:assert abs(rec[k]-record[k])<1e-10
 else:
  for k in initial:assert np.array_equal(actual[k],initial[k])
 if(i+1)%64==0:print('AUDIT_EVIDENCE_FITS69',i+1,len(rows),fitcount,flush=True)
# First C arrays are opened only after every AB fit was independently verified.
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];ab=load(ROOT/p['AB_capture_root']/rel);c=load(ROOT/p['C_random_root']/rel);e=load(ROOT/p['evidence_root']/rel);state=load(ROOT/p['fit_root']/rel);saved=load(pred/rel);g=load(pred/'association_PRIVATE'/(sid+'.npz'));raw=np.load(r['pointpath'],allow_pickle=False) if Path(r['pointpath']).suffix=='.npy' else np.fromfile(r['pointpath'],np.float32).reshape(-1,4);ids=np.unique(np.r_[ab['A_point_indices'],ab['B_point_indices'],c['point_indices']]);x=np.array(raw[ids,:3],np.float32,copy=True)
 if r['platform']=='drone':x[:,2]+=1.8
 zz={v+'_'+k:(c[k] if v=='C' else ab[v+'_'+k]) for v in 'ABC' for k in ['h','query','tokens','token_mask','boxes']};ga={'base':I.probability(ab['A_query'],ab['A_tokens'],ab['A_token_mask'])};assert np.array_equal(ga['base'],g['base'])
 for v in 'BC':
  af=M.affinity(zz['A_boxes'].astype(float),zz[v+'_boxes'].astype(float),x.astype(float));W=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));mass=float((W@I.probability(zz[v+'_query'],zz[v+'_tokens'],zz[v+'_token_mask'])).sum());assert np.array_equal(af,g[v+'_affinity']) and np.array_equal(W,g[v+'_W']) and mass==float(g[v+'_fixed_mass']);ga[v+'_affinity']=af;ga[v+'_W']=W;ga[v+'_fixed_mass']=mass
 expected={'source':ab['A_scores']};expected['frozen_ABC'],expected['R59']=M.readouts(zz,ga,initial);_,expected['R66_R59']=M.readouts(zz,ga,state);expected['R66_E_direct']=J.reweight(expected['R59'],e);expected['R66_set_direct']=J.project_set(expected['R59'],e);d=T.inputs(ab,c,ga);expected['C_source']=T.prediction(T.params(initial),d,'C').numpy();expected['C_R66']=T.prediction(T.params(state),d,'C').numpy()
 for repeat in range(3):
  rr={v+'_'+k:c['s'+str(repeat)+'_'+v+'_'+k] for v in 'ABC' for k in ['h','tokens']};rr.update({v+'_token_mask':zz[v+'_token_mask'] for v in 'ABC'});_,expected['random_s'+str(repeat)]=M.readouts(rr,ga,initial)
 assert np.array_equal(saved['source_boxes'],ab['A_boxes']) and np.array_equal(saved['selected_original_boxes'],ab['A_boxes'][saved['queries']])
 for j,a in enumerate(ARMS):
  err=float(abs(saved['scores'][j]-expected[a]).max());maxerr=max(maxerr,err);assert err<1e-11 and int(expected[a].argmax())==saved['queries'][j],(i,a,err)
 if(i+1)%64==0:print('AUDIT_READOUT69',i+1,len(rows),flush=True)
out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R69_ALL_INPUT_EVIDENCE_GROUPS_CONTROLS_FITS_CORRESPONDENCE_AND_READOUTS','successful_execution_attempt':2,'prior_import_failure_preserved':True,'n':len(rows),'arms':ARMS,'independent_fits':fitcount,'matched_seed_and_query_controls':controlcount,'max_probability_error':maxerr,'all_fits_independently_checked_before_C':True,'all_unknown_fallback_and_no_update_rows_retained':True,'canonical_source_256_boxes_and_readout_verified':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'GPU_sample_audit_sha256':sha(H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json'),'seconds':time.perf_counter()-start});print({'n':len(rows),'max_probability_error':maxerr})
