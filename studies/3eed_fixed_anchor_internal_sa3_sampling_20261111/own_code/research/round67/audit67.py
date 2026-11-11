"""Independent same-information readouts and all original-box selections."""
import sys,time,torch
from common67 import *
sys.path.insert(0,str(ROOT/'research/round65'));import audit_readout65 as indep
H=ROOT/'research/round67';p=read(H/'experiment/AUDIT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['capture_root'];pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');cert=read(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['capture_seal_sha256']==sha(cap/'SEALED.json');w=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');torch.set_grad_enabled(False);torch.set_num_threads(4);start=time.perf_counter();error=0.;statuses=read(pred/'READOUT_STATUS_PRIVATE.json')
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];z=load(cap/rel);actual=load(pred/rel);g=load(ROOT/p['association_root']/rel);prior=load(ROOT/p['r59_root']/rel);old66=load(ROOT/p['r66_prediction_root']/rel);expected={a:prior['scores'][prior['arm_names'].tolist().index(b)] for a,b in [('source','source'),('frozen_ABC','frozen_ABC'),('R59','direct_all')]};expected['R66_R59_tta1']=old66['scores'][old66['arm_names'].tolist().index('R59_tta1')]
 for mode in ['native','random','feature']:
  zz={v+'_'+k:z[mode+'_'+v+'_'+k] for v in 'ABC' for k in ['h','tokens']};zz.update({v+'_token_mask':z[v+'_token_mask'] for v in 'ABC'});_,expected[mode+'_R59']=indep.readouts(zz,g,w)
 assert actual['arm_names'].tolist()==ARMS and np.array_equal(actual['source_boxes'],z['A_boxes']) and np.array_equal(actual['source_scores'],z['A_scores']) and np.array_equal(actual['selected_original_boxes'],z['A_boxes'][actual['queries']])
 for j,a in enumerate(ARMS):
  e=float(abs(expected[a]-actual['scores'][j]).max());error=max(error,e);assert e<1e-11 and expected[a].argmax()==actual['queries'][j],(i,a,e)
 assert all(statuses[i]['arm_statuses'][a]['no_parameter_update'] for a in ARMS if a!='R66_R59_tta1')
out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R67_INDEPENDENT_SAMPLING_PROVENANCE_AND_ALL_FIXED_A256_READOUTS','n':336,'arms':len(ARMS),'all_original_final_boxes_preserved':True,'max_probability_error':error,'all_unknown_fallback_and_no_update_rows_retained':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'capture_audit_sha256':sha(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start});print({'n':336,'max_probability_error':error})
