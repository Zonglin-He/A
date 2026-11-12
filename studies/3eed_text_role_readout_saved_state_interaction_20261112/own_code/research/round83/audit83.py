"""Independent replay of saved states and explicit float64 readout arithmetic."""
from core83 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H83/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H83/'experiment/AUDIT_CODE_LOCK.json'));P=B83/'prediction2304_attempt001';ss=read(P/'SEALED.json');T=B83/'roles2304_attempt001';ts=read(T/'SEALED.json')
for f,h in ss['files'].items():assert sha(P/f)==h
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);start=time.perf_counter();maxS=maxp=maxnew=0.;cases=0
def meanexp(x,mask):
 y=x[:,mask].astype(float);a=y.max(1);return a+np.log(np.exp(y-a[:,None]).mean(1))
def norm(x):
 ex=np.exp(x-x.max());return ex/ex.sum()
for i,r in enumerate(read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')):
 sid=r['id'];role=read(T/'cases_PRIVATE'/(sid+'.json'));assert sha(T/'cases_PRIVATE'/(sid+'.json'))==ts['files']['cases_PRIVATE/'+sid+'.json'];z0,z,d=prefix83(r);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));old=load(B82/'prediction2640_attempt001/samples_PRIVATE'/(sid+'.npz'));refs=dict(zip(old['arm_names'],old['scores']));name=np.array(role['name'],bool);attr=np.array(role['attribute'],bool);bg=np.array(role['background'],bool);sh=np.array(role['mismatch'],bool);assert np.array_equal(name,z0['token_mask']) and np.flatnonzero(bg).tolist()==[len(bg)-2,len(bg)-1] and attr.sum()==sh.sum() and not (attr&sh).any();assert not (name&attr).any();assert role['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest()
 for state in ['frozen','BT']:
  if state=='BT':put_patch(m,load(B82/'prediction2640_attempt001/states_PRIVATE'/(sid+'_bt_correct.npz')))
  with torch.no_grad():pp,q,e=m.main(base,z,d,'bidirectional');matrix=(q@e['projected_text'].T/.07).cpu().numpy()
  maxS=max(maxS,float(abs(matrix-saved[state+'_matrix']).max()));assert maxS==0.;native=refs['frozen_bi' if state=='frozen' else 'bt_correct'];maxp=max(maxp,float(abs(pp.cpu().numpy()-native).max()));assert maxp==0.
  if role['usable']:
   a=meanexp(matrix,name);b=meanexp(matrix,bg);out={'native':native,'background':norm(a-b),'role':norm(.5*a+.5*meanexp(matrix,attr)-b) if attr.any() else norm(a-b),'mismatch':norm(.5*a+.5*meanexp(matrix,sh)-b) if sh.any() else norm(a-b)}
  else:out={k:native for k in ['native','background','role','mismatch']}
  for k,v in out.items():
   stored=saved['scores'][ARMS83.index(state+'_'+k)];maxnew=max(maxnew,float(abs(v-stored).max()));assert maxnew<1e-12 and int(v.argmax())==int(stored.argmax())
  restore(m,S);cases+=1
 assert np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']])
 for a in ['source','frozen_direct3','frozen_ABC','R59']:assert np.array_equal(saved['scores'][ARMS83.index(a)],refs[a])
 if(i+1)%128==0:print('AUDIT83',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp and all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());O=H83/'independent_native_audit_attempt001';O.mkdir();result={'state':'PASS_R83_ALL_SAVED_STATES_QUERY_TOKEN_MATRICES_READOUTS_AND_ORIGINAL_BOXES','n':2304,'arm_names':ARMS83,'state_forwards':cases,'max_matrix_error':maxS,'max_native_probability_error':maxp,'max_new_readout_error':maxnew,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'new_parameter_updates':0,'GT_read':False,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',result);print(result)
