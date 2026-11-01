"""Independent same-GPU cached2x2 arithmetic, then exact old-IoU join. No geometry/model."""
import json,hashlib,time,collections,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=ROOT/'research/round41/experiment';NEW=ROOT/'crossplatform/round41/cachedU128_attempt001';OLD=ROOT/'crossplatform/round35/predictionsU128_attempt001';SRC=ROOT/'crossplatform/round33/sourceU128_attempt001';ARMS=['source_native_replay','source_token_logodds','R30_native','R30_token_logodds','source_historical_stored_probability']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(p):
 seal=read(p/'SEALED.json')
 for f,h in seal['files'].items():assert sha(p/f)==h
 return seal

def main():
 start=time.perf_counter();auth=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_CACHED_CPU_DIAGNOSTIC'
 for f,h in auth['refs_sha256'].items():assert sha(ROOT/f)==h
 impl=read(EXP/'IMPLEMENTATION_LOCK.json');assert impl['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 for f,h in {**impl['program_sha256'],**impl['input_sha256']}.items():assert sha(ROOT/f)==h,f
 ns=verify(NEW);verify(OLD);verify(SRC);assert ns['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json') and ns['original_R35_seal_sha256']==sha(OLD/'SEALED.json');assert ns['within_weight_readout_query_unchanged_all128']
 for pl in ['drone','quad']:
  a=load(ROOT/'crossplatform/round35/partial_keep_attempt001/states_PRIVATE'/(pl+'_R30_U_rows.npz'));b=load(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
 rr=read(ROOT/'research/NIGHT_20261101/input/U128_PRIVATE.json');assert len(rr)==128 and all(r['stage']=='U' for r in rr);diag=[];queries={};nativechange=collections.Counter();pairerror=0.
 for r in rr:
  c=load(OLD/'predictions_PRIVATE'/(r['id']+'.npz'));src=load(SRC/'samples_PRIVATE'/(r['id']+'.npz'));p=load(NEW/'predictions_PRIVATE'/(r['id']+'.npz'));d=read(NEW/'diagnostics_PRIVATE'/(r['id']+'.json'));assert c['arm_names'].tolist()==['source','R30_U_rows','U_rows_matched_denominator','partial_output_keep'];assert c['probability_scores'].shape==c['exact_token_logodds'].shape==(4,256);assert c['source_probability_projection_replay'].shape==(256,)
  for v in [c['probability_scores'],c['exact_token_logodds'],c['source_probability_projection_replay']]:assert v.dtype==np.float32 and np.isfinite(v).all()
  assert np.array_equal(c['probability_scores'][0],src['scores']);assert sha(ROOT/d['original_R35_file'])==d['original_R35_sha256'] and sha(ROOT/d['original_source_file'])==d['original_source_sha256'];assert p['arm_names'].tolist()==ARMS
  expected=np.stack([c['source_probability_projection_replay'],c['exact_token_logodds'][0],c['probability_scores'][1],c['exact_token_logodds'][1],c['probability_scores'][0]]).astype(np.float64);q=np.argmax(expected,axis=-1);assert np.array_equal(p['scores'],expected) and np.array_equal(p['queries'],q);assert np.array_equal(p['boxes'],src['boxes'][q]);assert q[0]==q[1]==q[4] and q[2]==q[3] and q[4]==c['queries'][0] and q[2]==c['queries'][1]
  lm=-np.logaddexp(0.,-c['exact_token_logodds'][:2].astype(float));assert np.array_equal(lm,p['stable_logmass']);local=[]
  for weight,pval,zval,im in [('source',c['source_probability_projection_replay'],c['exact_token_logodds'][0],0),('R30',c['probability_scores'][1],c['exact_token_logodds'][1],1)]:
   topP=np.flatnonzero(pval==pval.max()).tolist();topZ=np.flatnonzero(zval==zval.max()).tolist();topL=np.flatnonzero(lm[im]==lm[im].max()).tolist();assert topZ==topL and np.all(np.diff(lm[im][np.argsort(zval,kind='stable')])>=0);err=float(np.max(abs(pval.astype(float)-np.exp(lm[im]))));pairerror=max(pairerror,err);assert err<2e-5
   out={'weight':weight,'native_query':int(pval.argmax()),'stable_query':int(zval.argmax()),'native_topset':topP,'stable_topset':topZ,'native_exact_top_tie_n':len(topP),'stable_exact_top_tie_n':len(topZ),'native_values_exactly_one':int((pval==1).sum()),'native_values_above_one':int((pval>1).sum()),'changed_by_readout':int(pval.argmax())!=int(zval.argmax()),'stable_winner_in_native_topset':int(zval.argmax()) in topP,'change_is_native_top_tie_break':int(pval.argmax())!=int(zval.argmax()) and int(zval.argmax()) in topP,'change_is_non_tie_numerical_reversal':int(pval.argmax())!=int(zval.argmax()) and int(zval.argmax()) not in topP,'logmass_topset_equal_logodds':True,'logmass_sorted_monotone':True,'logmass_not_subtracted_near_equal_LSEs':True,'probability_sigmoid_logodds_maxerror':err};local.append(out)
  assert local==d['numerical_diagnostics'];bridge={'source_replay_query':int(q[0]),'source_historical_query':int(q[4]),'query_changed':False,'max_probability_abs_difference':float(abs(c['source_probability_projection_replay'].astype(float)-c['probability_scores'][0].astype(float)).max()),'not_TTA_or_readout_factor_gain':True};assert bridge==d['source_arithmetic_bridge'];assert d['no_readout_query_change_for_either_weight']
  nativechange[r['platform']]+=int(q[0]!=q[2]);queries[r['alias']]=(r,dict(zip(ARMS,map(int,q))),p);diag.append({'alias':r['alias'],'platform':r['platform'],'source_bridge':bridge,'numerical_diagnostics':local})
 assert nativechange=={'drone':38,'quad':11};assert all(d['native_exact_top_tie_n']==d['stable_exact_top_tie_n']==1 and not d['native_values_exactly_one'] and not d['changed_by_readout'] for r in diag for d in r['numerical_diagnostics'])
 # No new geometry scoring: only after arithmetic audit has passed, join exact
 # presealed old query->IoU values and previously validated frozen selected boxes.
 oldpath=ROOT/'research/round35/experiment/U128_ALL_ROWS_PRIVATE.json';oldrows=read(oldpath);oldmap={r['alias']:r for r in oldrows};assert len(oldmap)==128 and set(oldmap)==set(queries);allrows=[]
 for alias,(r,q,p) in queries.items():
  old=oldmap[alias];c=load(OLD/'predictions_PRIVATE'/(r['id']+'.npz'));assert old['platform']==r['platform'];assert q['source_native_replay']==old['queries']['source'] and q['R30_native']==old['queries']['R30_U_rows'];assert np.array_equal(p['boxes'][0],c['boxes'][0]) and np.array_equal(p['boxes'][2],c['boxes'][1])
  vals={a:old['IoU']['R30_U_rows'] if a in ['R30_native','R30_token_logodds'] else old['IoU']['source'] for a in ARMS};allrows.append({'alias':alias,'platform':r['platform'],'scene_PRIVATE':old['scene_PRIVATE'],'queries':q,'IoU':vals,'old_IoU_source':'R35 sealed source/R30 samequery samebox exact join'})
 summary={};scenes=[];negative=[];refs=['source_native_replay','R30_native','source_historical_stored_probability']
 for pl in ['drone','quad']:
  ss=[r for r in allrows if r['platform']==pl];summary[pl]={}
  for a in ARMS:
   v=np.array([r['IoU'][a] for r in ss]);d=v-np.array([r['IoU']['source_native_replay'] for r in ss]);summary[pl][a]={'n':64,'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_sameforward_source_native_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source_native_replay'] for r in ss),'scene_clusters':len({r['scene_PRIVATE'] for r in ss})}
  for i,sc in enumerate(sorted({r['scene_PRIVATE'] for r in ss})):
   tt=[r for r in ss if r['scene_PRIVATE']==sc];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(i),'n':len(tt),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in tt])*100) for a in ARMS}})
 for r in allrows:
  for a in ARMS:
   for ref in refs:
    if a!=ref and r['IoU'][a]<r['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(r['IoU'][a]-r['IoU'][ref])})
 summary['balanced_mIoU_percent']={a:float(np.mean([summary[p][a]['mIoU_percent'] for p in ['drone','quad']])) for a in ARMS};balance=summary['balanced_mIoU_percent'];interaction=(balance['R30_token_logodds']-balance['source_token_logodds'])-(balance['R30_native']-balance['source_native_replay']);assert interaction==0
 output={'state':'PASS_R41_CACHED_ARITHMETIC_AND_EXACT_OLD_IOU_JOIN','n':128,'arms':5,'full_score_vectors_reconstructed':640,'frozen_selected_boxes_exact':640,'within_weight_query_effect':0,'source_historical_bridge_query_effect':0,'native_highest_tie_rows':0,'native_exact1_values':0,'stable_highest_tie_rows':0,'native_R30_vs_source_query_changes':dict(nativechange),'probability_sigmoid_logodds_maxerror':pairerror,'original_R35_R30_states_bitexact':True,'old_IoU_samequery_samebox_values_joined':640,'new_geometry_or_GT_container_reads':0,'old_IoU_file_sha256':sha(oldpath),'negative_method_reference_rows':len(negative),'state_STOP_correct_not_TTA_nochange':True,'no_new_model_GPU_optimizer_or_FINAL':True,'no_memory_or_fullmodel_adoption_claim':True,'prediction_seal_sha256':sha(NEW/'SEALED.json'),'implementation_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'root_authorization_sha256':sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),'seconds':time.perf_counter()-start}
 dump(HERE/'AUDIT_RESULT.json',output);dump(HERE/'ALL128_NUMERICAL_DIAGNOSTICS.json',diag);dump(HERE/'ALL128_TASK_ROWS_PRIVATE.json',allrows);dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'SCENE_ROWS.json',scenes);dump(EXP/'ALL_NEGATIVES.json',negative);dump(EXP/'ALL128_TASK_ROWS_PRIVATE.json',allrows);dump(EXP/'TASK_JOIN_RECEIPT.json',{'state':'PASS_EXACT_OLD_IOU_JOIN_NO_NEW_GEOMETRY','new_prediction_seal_sha256':sha(NEW/'SEALED.json'),'old_R35_IoU_sha256':sha(oldpath),'independent_arithmetic_audit_sha256':sha(HERE/'AUDIT_RESULT.json'),'joined_values':640,'negative_reference_set':refs,'balanced_mIoU_percent':balance,'native_TTA_delta_pp':balance['R30_native']-balance['source_native_replay'],'stable_TTA_delta_pp':balance['R30_token_logodds']-balance['source_token_logodds'],'interaction_pp':interaction,'all_after_new_prediction_seal':True,'new_GT_or_geometry_score_calls':0});print(json.dumps(output,ensure_ascii=False));print('JOINED_SUMMARY',balance)
if __name__=='__main__':main()
