"""Resolve immutable LOG providers and reconstruct all first-three saved steps, no GT."""
from common133 import *
import numpy as np,collections
def main():
 basecheck();reg=rd(R133/'research/round131/INPUT_REGISTRY_PRIVATE.json');out=[];maximum=0.;exact=0;active=0;engineering=[]
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R133/'crossplatform/round132/formal'/name;meta=rd(D/'records_PRIVATE'/(sid+'.json'))['LOG'];ref=meta['reference_original'];provider=R133/'crossplatform/round131/formal'/name if ref else D;record=rd(provider/'records_PRIVATE'/(sid+'.json')) if ref else meta;isactive=record['partial_eligible'];sp=provider/'states_PRIVATE'/(sid+'.npz' if ref else sid+'_LOG.npz');fp=provider/'features_PRIVATE'/(sid+'.npz' if ref else sid+'_LOG.npz');item=dict(e,active=isactive,reference_original_R131=ref,provider_states=str(sp.relative_to(R133)),provider_features=str(fp.relative_to(R133)),provider_states_sha256=sha(sp) if isactive else None,provider_features_sha256=sha(fp) if isactive else None,provider_record=record)
  if isactive:
   active+=1;states=np.load(sp);rr=record['parameter_radius'];delta1={};delta3={}
   for label in ['partial','wrong']:
    key='wrong_base' if ref and label=='wrong' else label;gkey='partial' if label=='wrong' and record['identity_control'] else key;trace=record['traces'][key];delta=np.zeros(4896,np.float32)
    for i,t in enumerate(trace):
     g=states[gkey+'_step'+str(i+1)+'_gradient']
     if t['accepted']:delta=(delta+(-np.float32(rr)*g/np.float32(t['gradient_L2']))*np.float32(t['fraction'])).astype(np.float32)
     if i==0:delta1[label]=delta.copy()
    error=float(abs(delta-states[key]).max());maximum=max(maximum,error);assert error<1e-7;(exact_count:=np.array_equal(delta,states[key]));exact+=exact_count;delta3[label]=states[key].copy()
   P=B133/'reconstructed_PRIVATE'/name;P.mkdir(exist_ok=True,parents=True);np.savez_compressed(P/(sid+'.npz'),**{label+str(step):val for step,dd in [(1,delta1),(3,delta3)] for label,val in dd.items()});item['reconstructed']=str((P/(sid+'.npz')).relative_to(R133));item['reconstructed_sha256']=sha(P/(sid+'.npz'))
  out.append(item)
 # Exercise cached-B and reconstructed-original-B provider paths without task labels.
 for name in sorted({e['source']+'_to_'+e['target'] for e in out}):
  rr=[e for e in out if e['source']+'_to_'+e['target']==name and e['active']]
  for reference in [False,True]:
   ss=[e for e in rr if e['reference_original_R131']==reference]
   if ss:engineering.append({'source':ss[0]['source'],'target':ss[0]['target'],'id_PRIVATE':ss[0]['row']['id']})
 dump(H133/'INPUT_REGISTRY_PRIVATE.json',out);dump(H133/'ENGINEERING_INPUTS_PRIVATE.json',engineering);summary={'n':len(out),'active':active,'inactive':len(out)-active,'active_provider_counts':dict(collections.Counter('R131' if e['reference_original_R131'] else 'R132' for e in out if e['active'])),'three_step_reassembly_max_error':maximum,'bitexact_reassembled_states':exact,'reassembled_states':active*2,'engineering_inputs':len(engineering),'GT_read':False,'FINAL_access':False};dump(H133/'PREPARATION_SUMMARY.json',summary)
 plan={'utc':utc(),'question':'At fixed R132 LOG supervision/interface/rho, what do1/3/5steps change in grounding versus same-information direct and matched wrong labels?','primary':'partial3_F reused exactly; no targetGT-based replacement/step selection','sensitivity':[1,5],'no_parameter_selection_in_this_round':True,'fixed':'same1798/sixdirections/A256/originalB/source3/R91states/kappa2.742430260548657/S,C,wrongC/PointUp4896/rho0/featurekeep1/Armijo9/episodicreset/fullFAB','trajectory':'derive endpoint1 from saved first gradient+accepted fraction; reuse actual endpoint3; continue original endpoint3 for steps4,5 with unchanged objective/line search; wrongcontrol does same','cost':'same per-step proposal radius, so nominal path budget grows with steps; report L2 displacement,accepted pathlength,backward/forward/trial counts and incremental measured timing; do not call cached decode latency full TTA cost','reuse':'R132original4LOG outputs and646 actual delta3s reused; no baseline/backbone/calibration rerun, except originalB alignment58active without saved B_vis; all unsupported retained','analysis':'all49better27worse relativezero inR132 revisited after new prediction seal; classify C exclusion, withinC competition, geometricduplicate switches, unknown; IoU/groups not identity labels; all new endpoint harms retained','evaluation':'existing R131 all256IoUs only after every new prediction/state sealed; all1798 full denominator; same content/frame/recording clusters and inverse sampling weights; raw/sixdirections/targets/scenes','protocol':'exposed development and prior R129/R132 pooledsource calibration; not new independent confirmation or strict direction-isolated label selection','no_source_training':True,'no_B3':True,'no_new_loss_or_guard':True,'FINAL_access':False,'resource':'one GPU exclusive,one bounded target matrix<=10800seconds,guardedCPU scoring, no further step/rho/kappa search','files':{p.name:sha(p) for p in [H133/'INPUT_REGISTRY_PRIVATE.json',H133/'ENGINEERING_INPUTS_PRIVATE.json',H133/'PREPARATION_SUMMARY.json']}}
 dump(H133/'experiment/DESIGN_LOCK.json',plan);print(summary)
if __name__=='__main__':main()
