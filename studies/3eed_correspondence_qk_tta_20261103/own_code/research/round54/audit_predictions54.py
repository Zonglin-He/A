"""All-case original boxes, fitted-variable scope and effective-update audit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def exact(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 pred=ROOT/p['prediction_root'];seal=read(pred/'SEALED.json');rows=read(ROOT/p['manifest']);status={r['id']:r['arm_statuses'] for r in read(pred/'READOUT_STATUS_PRIVATE.json')};main=ROOT/p['main_root'];ms=read(main/'SEALED.json');strong=ROOT/p['strong_root'];ss=read(strong/'SEALED.json');canon=ROOT/p['canonical_root'];cs=read(canon/'SEALED.json');start=time.perf_counter();arms=seal['arm_names'];updates={a:0 for a in arms};changes={a:0 for a in arms};zero_exact=0;maxnorm_error=0.;nfit=0
 assert read(ROOT/p['observation_audit'])['state'].startswith('PASS_');assert read(ROOT/p['method_audit'])['state'].startswith('PASS_')
 for rel,h in seal['files'].items():assert sha(pred/rel)==h
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';fr='fitted_PRIVATE/'+sid+'.npz';assert sha(canon/rel)==cs['files'][rel] and sha(main/rel)==ms['files'][rel] and sha(strong/rel)==ss['files'][rel] and sha(main/fr)==ms['files'][fr] and sha(strong/fr)==ss['files'][fr]
  record=read(main/'cases_PRIVATE'/(sid+'.json'));details=record['statistics'];extra=read(strong/'cases_PRIVATE'/(sid+'.json'))['details'];assert details['source_bridge_bitexact'] and details['direct_zero_Top1_preserved']
  with np.load(pred/rel,allow_pickle=False) as z,np.load(canon/rel,allow_pickle=False) as c,np.load(main/rel,allow_pickle=False) as m,np.load(strong/rel,allow_pickle=False) as s,np.load(main/fr,allow_pickle=False) as fit,np.load(strong/fr,allow_pickle=False) as f20:
   exact(z['source_boxes'],c['boxes']);exact(z['source_scores'],c['scores']);exact(z['scores'][:11],m['scores']);exact(z['scores'][11:],s['scores']);assert z['arm_names'].tolist()==arms;zero_exact+=bool(np.array_equal(z['scores'][0],z['scores'][1]));assert int(z['queries'][0])==int(z['queries'][1])
   for j,arm in enumerate(arms):
    assert np.isfinite(z['scores'][j]).all() and z['queries'][j]==z['scores'][j].argmax();exact(z['selected_original_boxes'][j],c['boxes'][int(z['queries'][j])]);st=status[sid][arm];assert set(st)=={'unknown','fallback','no_parameter_update'} and all(type(x) is bool for x in st.values()) and not st['fallback']
    if arm in ['source','zero_step']:assert st['no_parameter_update'];continue
    if arm.startswith('free'):
     value=f20[arm] if arm.endswith('_20') else fit[arm];assert value.shape==(256,1024) and np.isfinite(value).all();norm=float(np.linalg.norm(value.astype(float)));expected=extra[arm]['final_diagnostics']['parameter_delta_L2'] if arm.endswith('_20') else details['trajectories'][arm.rsplit('_',1)[0]]['steps'][int(arm.rsplit('_',1)[1])-1]['parameter_delta_L2'];assert st['no_parameter_update']
    else:
     sizes={'A_q':(4,288),'B_q':(288,4),'A_k':(4,288),'B_k':(288,4)};square=0.
     for name,shape in sizes.items():
      v=fit[arm+'__'+name];assert v.shape==shape and np.isfinite(v).all();origin=fit['qk_1__'+name] if name.startswith('A_') else np.zeros_like(v);square+=float(np.sum((v.astype(float)-origin.astype(float))**2))
     norm=float(np.sqrt(square));expected=details['trajectories'][arm.rsplit('_',1)[0]]['steps'][int(arm.rsplit('_',1)[1])-1]['parameter_delta_L2'];assert st['no_parameter_update']==(expected==0.)
     if arm in ['qk_1','qk_RGBshuffle_1']:
      assert np.array_equal(fit[arm+'__A_q'],fit['qk_1__A_q']) and np.array_equal(fit[arm+'__A_k'],fit['qk_1__A_k'])
     updates[arm]+=expected>0
    error=abs(norm-expected);assert error<2e-4,(arm,error);maxnorm_error=max(maxnorm_error,error);nfit+=1;changes[arm]+=int(z['queries'][j])!=int(z['queries'][0])
 result={'state':'PASS_R54_INDEPENDENT_ALL_CASE_FITTED_SCOPE_AND_BOX_READOUT_AUDIT','n':len(rows),'arm_names':arms,'prediction_seal_sha256':sha(pred/'SEALED.json'),'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'zero_step_scores_bitexact_cases':zero_exact,'zero_step_Top1_same_all':True,'fitted_arm_states_checked':nfit,'QK_actual_nonzero_update_cases':updates,'query_changes_vs_source':changes,'maximum_saved_parameter_norm_error':maxnorm_error,'no_GT_or_model_calls':0,'seconds':time.perf_counter()-start,'observation_audit_sha256':sha(ROOT/p['observation_audit']),'method_audit_sha256':sha(ROOT/p['method_audit'])};out=ROOT/'research/round54/prediction_audit_attempt001';assert not out.exists();out.mkdir();(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
