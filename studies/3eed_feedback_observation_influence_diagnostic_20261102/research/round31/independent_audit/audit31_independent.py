"""Independent CPU audit of saved R31 states, predictions and influence reductions.
No torch, optimizer, GPU or source-of-truth label filter is invoked.
"""
import sys,json,hashlib,collections,time,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'crossplatform/round31'),str(ROOT/'research/round5/audit')]
from common31 import read,dump,sha,seal,verify,load_npz,rows,feature
from independent_core_v2 import candidate_ious,transform_gt
BASE=ROOT/'crossplatform/round31/loo19_attempt002';OUT=ROOT/'research/round31/independent_audit'
COUNTS=collections.Counter()
def lse(x):
 m=x.max(-1);return m+np.log(np.exp(x-m[:,None]).sum(-1))
def head(state,z):
 h=z['h'].astype(np.float64)
 for k in ['0','2','4']:
  h=h@state[k+'.weight'].astype(np.float64).T+state[k+'.bias'].astype(np.float64)
  if k!='4':h=np.maximum(h,0)
 h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-12);v=h@z['tokens'].astype(np.float64).T/.07;mask=z['token_mask'].astype(bool);lo=lse(v[:,mask])-lse(v[:,~mask]);return lo,1/(1+np.exp(-lo))
def group_q(r,s,z):
 if r['stage'] in ['Q','R','S']:return int(s.argmax())
 g=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();options=[]
 for inds in g.values():
  used.update(inds);rep=max(inds,key=lambda i:(float(z['scores'][i]),-i));options.append((float(s[inds].max()),-rep,rep))
 for i in range(256):
  if i not in used:options.append((float(s[i]),-i,i))
 return max(options)[2]
def numclose(a,b,tol=1e-9):
 assert abs(float(a)-float(b))<tol,(a,b)
def main():
 start=time.perf_counter();verify(BASE);verify(ROOT/'crossplatform/round30/screen_attempt001');verify(ROOT/'crossplatform/round29/direct208_attempt001')
 specs=read(BASE/'MODEL_SPECS_PRIVATE.json');receipts={r['name']:r for r in read(BASE/'FIT_RECEIPTS_PRIVATE.json')};assert len(specs)==len(receipts)==23
 ff=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={f['id_PRIVATE'] for f in ff};assert len(ff)==22 and len(fids)==19
 allrows=[dict(r,split=split) for split in ['A208','DEV64','R64','S64'] for r in rows(split)];assert len(allrows)==400 and len({r['id'] for r in allrows})==400;lookup={r['id']:r for r in allrows};zmap={r['id']:feature(r) for r in allrows}
 initial=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');initial={k[6:]:v for k,v in initial.items() if k.startswith('image.')};initial_lo={};initial_err=0
 for r in rows():
  lo,pr=head(initial,zmap[r['id']]);initial_lo[r['id']]=lo;initial_err=max(initial_err,float(abs(pr-zmap[r['id']]['scores']).max()))
 assert initial_err<2e-5;pred={};state_errors={};pair_errors=[];first_loss_errors=[];keep_contract={}
 for pl in ['drone','quad']:
  feedback=[f for f in ff if f['platform']==pl];N=len(feedback);A=[r for r in rows() if r['platform']==pl];U=[r['id'] for r in A if r['id'] not in fids];assert N=={'drone':10,'quad':12}[pl] and len(U)=={'drone':96,'quad':93}[pl]
  keep_contract[pl]={'denominator':N,'fixed_keep_n':len(U),'keep_sorted_id_sha256_PRIVATE':hashlib.sha256('\n'.join(sorted(U)).encode()).hexdigest(),'feedback_observations':len({f['id_PRIVATE'] for f in feedback})}
  platform_specs=[x for x in specs if x['platform']==pl];assert sum(x['kind']=='leave_one_observation' for x in platform_specs)=={'drone':8,'quad':11}[pl];assert sum(x['kind']=='all' for x in platform_specs)==sum(x['kind']=='zero' for x in platform_specs)==1
  omitted={x['omitted_id_PRIVATE'] for x in platform_specs if x['kind']=='leave_one_observation'};assert omitted=={f['id_PRIVATE'] for f in feedback}
  gap0=np.array([initial_lo[f['id_PRIVATE']][f['positive_group_PRIVATE']].max()-initial_lo[f['id_PRIVATE']][f['negative_group_PRIVATE']].max() for f in feedback]);loss0=np.logaddexp(0,-gap0/.1)
  for spec in platform_specs:
   rec=receipts[spec['name']];state=load_npz(BASE/'states_PRIVATE'/(spec['name']+'.npz'));p=load_npz(BASE/'predictions_PRIVATE'/(spec['name']+'.npz'));pred[spec['name']]=p;rr=[r for r in allrows if r['platform']==pl];assert p['row_ids_PRIVATE'].tolist()==[r['id'] for r in rr];assert p['aliases'].tolist()==[r['alias'] for r in rr];assert p['scores'].shape==(200,256) and p['queries'].shape==(200,)
   assert rec['original_N']==spec['original_N']==N and rec['keep_n']==spec['keep_n']==len(U);assert rec['steps']==16 and rec['source_initialization'] and rec['removed_added_to_keep']==False
   weights=np.ones(N)
   if spec['kind']=='zero':weights[:]=0
   elif spec['kind']=='leave_one_observation':weights=np.array([f['id_PRIVATE']!=spec['omitted_id_PRIVATE'] for f in feedback],float);assert spec['omitted_id_PRIVATE'] not in U;assert spec['omitted_alias']==lookup[spec['omitted_id_PRIVATE']]['alias']
   assert spec['active_preferences']==int(weights.sum());expected_init_pref=float((loss0*weights).sum()/N);first=rec['trace'][0];assert first['step']==1;err=abs(expected_init_pref-first['pref']);assert err<3e-4,(spec['name'],expected_init_pref,first['pref']);assert first['keep']==0;first_loss_errors.append(err);COUNTS['fixed_denominator_initial_objectives_recomputed']+=1
   if spec['kind']=='all':
    r30=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));assert state.keys()==r30.keys();assert all(np.array_equal(state[n],r30[n]) for n in state);state_errors[spec['name']]=0.;assert rec['baseline_R30_state_maxerror']==0
   if spec['kind']=='zero':
    assert all(np.array_equal(state[n],initial[n]) for n in state);assert rec['parameter_delta_L2']==0;assert all(t['pref']==t['keep']==t['loss']==0 for t in rec['trace'])
   norm=float(np.sqrt(sum(np.square(state[n].astype(float)-initial[n].astype(float)).sum() for n in state)));assert abs(norm-rec['parameter_delta_L2'])<2e-5
   cpu_lo={};maxscore=0.
   for i,r in enumerate(rr):
    z=zmap[r['id']];stored_q=int(p['queries'][i]);assert stored_q==group_q(r,p['scores'][i],z);lo,score=head(state,z);maxscore=max(maxscore,float(abs(score-p['scores'][i]).max()));assert maxscore<2e-5,(spec['name'],r['alias'],maxscore);cpu_lo[r['id']]=lo;COUNTS['state_score_vectors_recomputed']+=1;COUNTS['group_readouts_recomputed']+=1
    if spec['kind']=='zero':assert stored_q==int(z['scores'].argmax());COUNTS['zero_source_choices_reproduced']+=1
   gaps=np.array([cpu_lo[f['id_PRIVATE']][f['positive_group_PRIVATE']].max()-cpu_lo[f['id_PRIVATE']][f['negative_group_PRIVATE']].max() for f in feedback]);er=float(abs(gaps-np.array(rec['all_original_pair_gaps'])).max());assert er<5e-4,(spec['name'],er);pair_errors.append(er);state_errors[spec['name']]=maxscore
 # A direct CPU recomputation of every selected frozen box, with metadata bindings.
 idx={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};task={};source={}
 for r in allrows:
  ii=idx[r['id']];path=ii['meta_path']
  if path not in containers:assert sha(path)==ii['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ii['object_index']]['bbox_3d'],r['platform'],ii.get('waymo_view_suffix'));z=zmap[r['id']];q0=int(z['scores'].argmax());choices={'source':q0}
  for spec in specs:
   if spec['platform']==r['platform']:
    p=pred[spec['name']];j=p['row_ids_PRIVATE'].tolist().index(r['id']);choices[spec['name']]=int(p['queries'][j])
  b=z['boxes'][np.array(list(choices.values()))].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vals=candidate_ious(b,gt)[0]
  for name,value in zip(choices,vals):
   if name=='source':source[r['id']]=float(value)
   else:task.setdefault(name,{})[r['id']]=float(value);COUNTS['selected_box_task_values_recomputed']+=1
 # Rebuild all 3800 influence entries independently and compare raw task deltas.
 original=read(ROOT/'research/round31/experiment/ALL_INFLUENCE_ROWS_PRIVATE.json');original={(r['model'],r['target_id_PRIVATE']):r for r in original};reconstructed=[]
 for spec in specs:
  if spec['kind']!='leave_one_observation':continue
  name=spec['name'];pl=spec['platform'];baseline=task[pl+'_all'];p=pred[name];pb=pred[pl+'_all'];assert p['row_ids_PRIVATE'].tolist()==pb['row_ids_PRIVATE'].tolist()
  for i,rid in enumerate(p['row_ids_PRIVATE'].tolist()):
   r=lookup[rid];delta=100*(task[name][rid]-baseline[rid]);q=int(p['queries'][i]);qb=int(pb['queries'][i]);o=original[name,rid]
   for k,val in [('source_iou',source[rid]),('all_iou',baseline[rid]),('minus_iou',task[name][rid]),('delta_remove_pp',delta),('baseline_gain_vs_source_pp',100*(baseline[rid]-source[rid])),('minus_gain_vs_source_pp',100*(task[name][rid]-source[rid]))]:numclose(o[k],val,1e-10)
   assert o['query_changed']==(q!=qb);assert o['all_query_PRIVATE']==qb and o['minus_query_PRIVATE']==q;assert o['split']==r['split'] and o['platform']==pl and o['target_alias']==r['alias'];assert o['is_removed_observation']==(rid==spec['omitted_id_PRIVATE']);reconstructed.append(dict(o));COUNTS['influence_rows_recomputed']+=1
 assert len(original)==len(reconstructed)==3800
 # R30 reference choices and task scores, not only state values.
 r30={r['id_PRIVATE']:r for r in read(ROOT/'research/round30/experiment/ALL336_TASK_ROWS_PRIVATE.json')};shared=0
 for pl in ['drone','quad']:
  p=pred[pl+'_all']
  for i,rid in enumerate(p['row_ids_PRIVATE'].tolist()):
   if rid in r30:assert int(p['queries'][i])==r30[rid]['q']['keep_selected'];numclose(task[pl+'_all'][rid],r30[rid]['ious']['keep_selected'],1e-12);shared+=1
 assert shared==336
 # Reconstruct all aggregate subset means/counts and nonadditive signed tails.
 summaries={r['feedback_alias']:r for r in read(ROOT/'research/round31/experiment/INFLUENCE19_SUMMARY.json')};independent_summary=[]
 for spec in specs:
  if spec['kind']!='leave_one_observation':continue
  rs=[r for r in reconstructed if r['model']==spec['name']];sm=summaries[spec['omitted_alias']];expected_removed=spec['original_N']-spec['active_preferences'];assert sm['removed_preferences']==expected_removed;result={'feedback_alias':spec['omitted_alias'],'platform':spec['platform'],'removed_preferences':expected_removed,'subsets':{}}
  for split in ['A208','DEV64','R64','S64','R64_S64','A_feedback_others','A_no_feedback']:
   rr=[r for r in rs if not r['is_removed_observation'] and (r['split'] in ['R64','S64'] if split=='R64_S64' else r['split']=='A208' and ((r['target_id_PRIVATE'] in fids) if split=='A_feedback_others' else r['target_id_PRIVATE'] not in fids) if split in ['A_feedback_others','A_no_feedback'] else r['split']==split)];d=np.array([r['delta_remove_pp'] for r in rr]);agg={'n':len(rr),'mean_delta_remove_pp':float(d.mean()),'improves_if_removed':int((d>1e-8).sum()),'worsens_if_removed':int((d< -1e-8).sum())};a=sm['subsets'][split];assert a['n']==len(rr);numclose(a['mean_delta_remove_pp'],agg['mean_delta_remove_pp'],1e-10);assert a['improves_if_removed']==agg['improves_if_removed'] and a['worsens_if_removed']==agg['worsens_if_removed']
   if split not in ['A_feedback_others','A_no_feedback']:
    fields={'mIoU_all_percent':float(np.mean([r['all_iou'] for r in rr])*100),'mIoU_minus_percent':float(np.mean([r['minus_iou'] for r in rr])*100),'changed_query':sum(r['query_changed'] for r in rr),'baseline_harms_recovered_pp_sum':float(sum(max(r['minus_iou']-r['all_iou'],0)*100 for r in rr if r['baseline_gain_vs_source_pp']< -1e-8)),'baseline_gains_lost_pp_sum':float(sum(max(r['all_iou']-r['minus_iou'],0)*100 for r in rr if r['baseline_gain_vs_source_pp']>1e-8))}
    for k,v in fields.items():numclose(a[k],v,1e-10)
    agg.update(fields)
   result['subsets'][split]=agg;COUNTS['aggregate_subsets_recomputed']+=1
  assert result['subsets']['A208']['n']==103;assert result['subsets']['A_no_feedback']['n']=={'drone':96,'quad':93}[spec['platform']];independent_summary.append(result)
 # Engineering failure trace is retained, not misclassified as a scientific run.
 repair=read(ROOT/'research/round31/experiment/ENGINEERING_REPAIR.json');assert repair['state']=='failed_before_any_optimizer_step';failed=ROOT/'crossplatform/round31/loo19_attempt001';assert failed.exists() and not (failed/'SEALED.json').exists();assert not list(failed.rglob('*.npz'))
 # Explicitly distinguish structural cross-platform isolation from measured new outputs.
 cross_platform={'only_own_platform_rows_materialized':True,'other_platform_state_uses_unchanged_all_reference':True,'cross_platform_delta_is_structural_zero_not_3800_new_scored_predictions':True}
 dump(OUT/'INFLUENCE19_INDEPENDENT_SUMMARY.json',independent_summary);dump(OUT/'ALL3800_RECONSTRUCTED_ROWS_PRIVATE.json',reconstructed);dump(OUT/'FIXED_CONTRACT_PRIVATE.json',keep_contract)
 result={'state':'PASS','reviewer':'independent_cpu_subagent','counts':dict(COUNTS),'models':23,'model_kinds':dict(collections.Counter(x['kind'] for x in specs)),'original_pair_denominators':{'drone':10,'quad':12},'original_keep_counts':{'drone':96,'quad':93},'all_omitted_observations_outside_keep':True,'initial_fixed_denominator_objective_max_error':max(first_loss_errors),'source_score_replay_max_error':initial_err,'state_score_replay_max_error':max(state_errors.values()),'final_pair_gap_CPU_GPU_max_error':max(pair_errors),'all_feedback_state_vs_R30_bitexact':True,'zero_states_vs_source_bitexact':True,'R30_shared_prediction_and_task_exact':shared,'cross_platform':cross_platform,'Aothers_excludes_own_observation':True,'all_19_LOO_x200_equals3800':True,'raw_GTs_used_only_in_diagnostic_CPU_taskscore':True,'attempt001_failure_preserved_no_optimizer_states':True,'no_GPU_or_optimizer_or_Qwen_called':True,'all_positive_and_negative_influence_rows_retained':True,'limitations':['Code/receipts plus initial loss reconstruction establish denominator and fixed-U contract; no independent training rerun was performed.','Numerical scores and task geometries are reconstructed independently from saved states/feature banks and bound metadata, using approved independent_core_v2 geometry.','R64/S64 andDEV labels are exposed diagnostic evidence, not new confirmation or runtime feedback filtering.','Single-observation removal effects under nonlinear optimization are not additive; cannot turn influence rankings into deployable label quality gates.','Only corresponding-platform counterfactuals are materialized; opposite platform is fixed all-state by experimental construction.'],'seconds':time.perf_counter()-start};dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
