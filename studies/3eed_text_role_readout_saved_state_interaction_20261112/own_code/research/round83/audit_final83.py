"""All final masks/arithmetic plus unchanged already GPU-audited matrices."""
from core83 import *
from roles_fixed83 import roles as fixedroles
import collections,time
P=B83/'prediction2304_attempt002';oldP=B83/'prediction2304_attempt001';T=B83/'roles2304_attempt002';s=read(P/'SEALED.json');olds=read(oldP/'SEALED.json');prior=read(H83/'independent_native_audit_attempt001/AUDIT_RESULT.json');assert prior['state'].startswith('PASS_') and prior['prediction_seal_sha256']==sha(oldP/'SEALED.json');coh=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');start=time.perf_counter();cases=[];maxerror=0.
for i,r in enumerate(read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(P/rel)==s['files'][rel] and sha(oldP/rel)==olds['files'][rel];z=load(P/rel);old=load(oldP/rel);role=read(T/'cases_PRIVATE'/(sid+'.json'));again=fixedroles(r['caption'],role['document'],role['subject'],np.array(role['offsets']),np.array(role['name'],bool),sid)
 for k in ['name','attribute','background','mismatch']:assert np.array_equal(again[k],role[k])
 assert again['usable']==role['usable'];attr=np.array(role['attribute'],bool);sh=np.array(role['mismatch'],bool);assert attr.sum()==sh.sum() and not (attr&sh).any()
 for key in ['source_boxes','frozen_matrix','BT_matrix','frozen_query','BT_query','frozen_text','BT_text']:assert np.array_equal(z[key],old[key])
 for state in ['frozen','BT']:
  matrix=z[state+'_matrix'].astype(float);name=np.array(role['name'],bool);bg=np.array(role['background'],bool)
  def lm(mask):
   x=matrix[:,mask];m=x.max(1);return m+np.log(np.exp(x-m[:,None]).mean(1))
  def pp(v):x=np.exp(v-v.max());return x/x.sum()
  native=old['scores'][ARMS83.index(state+'_native')];out={'native':native}
  if role['usable']:
   a,b=lm(name),lm(bg);out.update(background=pp(a-b),role=pp(.5*a+.5*lm(attr)-b) if attr.any() else pp(a-b),mismatch=pp(.5*a+.5*lm(sh)-b) if sh.any() else pp(a-b))
  else:out.update({k:native for k in ['background','role','mismatch']})
  for k,v in out.items():q=z['scores'][ARMS83.index(state+'_'+k)];maxerror=max(maxerror,float(abs(v-q).max()));assert maxerror<1e-12 and int(v.argmax())==int(q.argmax())
 assert np.array_equal(z['queries'],z['scores'].argmax(1)) and np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);cases.append({'row':'E'+str(i+1).zfill(5),'cohort':coh[sid],'platform':r['platform'],'usable':role['usable'],'attribute_enabled':role['usable'] and role['attribute_count']>0,'attribute_subtokens':role['attribute_count'],'reason':role['reason']})
summary=[]
for c in ['VAL1280','TRAIN1024']:
 for pl in ['drone','quad','all']:
  rr=[r for r in cases if r['cohort']==c and (pl=='all' or r['platform']==pl)]
  for reason,n in collections.Counter(r['reason'] for r in rr).items():summary.append({'cohort':c,'platform':pl,'reason':reason,'n':n})
  summary.append({'cohort':c,'platform':pl,'n':len(rr),'usable':sum(r['usable'] for r in rr),'attribute_enabled':sum(r['attribute_enabled'] for r in rr),'UNKNOWN':sum(not r['usable'] for r in rr)})
dump(H83/'ROLE_COVERAGE_ROWS.json',cases);dump(H83/'ROLE_COVERAGE_SUMMARY.json',summary);O=H83/'independent_final_native_audit_attempt001';O.mkdir();v={**prior,'state':'PASS_R83_FINAL_ROLES_ARITHMETIC_AND_UNCHANGED_GPU_AUDITED_NATIVE_MATRICES','max_new_readout_error':maxerror,'prediction_seal_sha256':sha(P/'SEALED.json'),'original_neural_audit_seal_sha256':sha(oldP/'SEALED.json'),'final_CPU_audit_seconds':time.perf_counter()-start,'seconds':prior['seconds']+time.perf_counter()-start,'all_neural_matrices_bit_identical_to_prior_GPU_audit':True,'GT_read':False};dump(O/'AUDIT_RESULT.json',v);print(v)
