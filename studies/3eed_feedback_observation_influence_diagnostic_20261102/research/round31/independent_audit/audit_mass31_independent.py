"""Independent CPU audit of uniform nominal-preference-mass controls."""
import sys,json,collections,time,numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE),str(ROOT/'crossplatform/round31'),str(ROOT/'research/round5/audit')]
from audit31_independent import head,group_q,numclose
from common31 import read,dump,sha,verify,load_npz,rows,feature
from independent_core_v2 import candidate_ious,transform_gt
OUT=HERE;BASE=ROOT/'crossplatform/round31/masscontrol_attempt001'
def main():
 start=time.perf_counter();verify(BASE);assert read(OUT/'AUDIT_RESULT.json')['state']=='PASS';specs=read(BASE/'MODEL_SPECS_PRIVATE.json');receipts={x['name']:x for x in read(BASE/'FIT_RECEIPTS_PRIVATE.json')};assert len(specs)==len(receipts)==4;counts=collections.Counter();feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fids={f['id_PRIVATE'] for f in feedback};rr=[dict(r,split=s) for s in ['A208','DEV64','R64','S64'] for r in rows(s)];lookup={r['id']:r for r in rr};aliaslookup={r['alias']:r for r in rr};assert len(aliaslookup)==400
 zmap={r['id']:feature(r) for r in rr};initial=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');initial={k[6:]:v for k,v in initial.items() if k.startswith('image.')};source_lo={r['id']:head(initial,zmap[r['id']])[0] for r in rows()};initial_errors=[];score_errors=[];gap_errors=[];masspred={};nominal=[]
 for spec in specs:
  pl=spec['platform'];N={'drone':10,'quad':12}[pl];k=spec['removed_count_matched'];assert k in [1,2];c=(N-k)/N;assert spec['original_N']==N;numclose(spec['uniform_weight'],c,1e-14);assert spec['kind']=='uniform_mass_control';assert spec['omitted_alias'] is None and spec['omitted_id_PRIVATE'] is None;assert spec['keep_n']=={'drone':96,'quad':93}[pl] and not spec['removed_added_to_keep'];assert spec['active_preferences']==N-k
  f=[x for x in feedback if x['platform']==pl];assert len(f)==N;U=[r for r in rows() if r['platform']==pl and r['id'] not in fids];assert len(U)==spec['keep_n'];rec=receipts[spec['name']];assert rec['steps']==16 and rec['source_initialization'];assert rec['active_preferences']==N-k
  # All terms survive; each is multiplied by c, still divided by original N.
  gap0=np.array([source_lo[x['id_PRIVATE']][x['positive_group_PRIVATE']].max()-source_lo[x['id_PRIVATE']][x['negative_group_PRIVATE']].max() for x in f]);expected=float((np.logaddexp(0,-gap0/.1)*c).sum()/N);err=abs(expected-rec['trace'][0]['pref']);assert err<3e-4;assert rec['trace'][0]['keep']==0;initial_errors.append(err);counts['uniform_initial_objectives_recomputed']+=1
  state=load_npz(BASE/'states_PRIVATE'/(spec['name']+'.npz'));norm=float(np.sqrt(sum(np.square(state[n].astype(float)-initial[n].astype(float)).sum() for n in state)));assert norm>0 and abs(norm-rec['parameter_delta_L2'])<2e-5;p=load_npz(BASE/'predictions_PRIVATE'/(spec['name']+'.npz'));masspred[spec['name']]=p;targets=[r for r in rr if r['platform']==pl];assert p['row_ids_PRIVATE'].tolist()==[r['id'] for r in targets];assert p['aliases'].tolist()==[r['alias'] for r in targets];assert p['scores'].shape==(200,256);los={}
  for i,r in enumerate(targets):
   lo,sc=head(state,zmap[r['id']]);los[r['id']]=lo;er=float(abs(sc-p['scores'][i]).max());assert er<2e-5;score_errors.append(er);assert int(p['queries'][i])==group_q(r,p['scores'][i],zmap[r['id']]);counts['mass_state_score_vectors_recomputed']+=1;counts['mass_group_choices_recomputed']+=1
  gap=np.array([los[x['id_PRIVATE']][x['positive_group_PRIVATE']].max()-los[x['id_PRIVATE']][x['negative_group_PRIVATE']].max() for x in f]);er=float(abs(gap-rec['all_original_pair_gaps']).max());assert er<5e-4;gap_errors.append(er);nominal.append({'platform':pl,'removed_count_matched':k,'original_pair_count_nonzero':N,'uniform_coefficient':c,'nominal_weight_sum':N-k,'historical_active_preferences_is_nominal_mass_not_nonzero_count':True})
 assert sorted((s['platform'],s['removed_count_matched']) for s in specs)==[('drone',1),('drone',2),('quad',1),('quad',2)]
 # New raw GT scoring for all800selectedboxes, independent of mass main scorer.
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};mass={};originalmass={(r['model'],r['target_id_PRIVATE']):r for r in read(ROOT/'research/round31/experiment/MASS_CONTROL_ALL_ROWS_PRIVATE.json')};assert len(originalmass)==800
 for spec in specs:
  p=masspred[spec['name']]
  for i,rid in enumerate(p['row_ids_PRIVATE'].tolist()):
   r=lookup[rid];j=idx[rid];path=j['meta_path']
   if path not in containers:assert sha(path)==j['metadata_container_sha256'];containers[path]=read(path)
   gt=transform_gt(containers[path]['ground_info'][j['object_index']]['bbox_3d'],r['platform'],j.get('waymo_view_suffix'));q=int(p['queries'][i]);b=zmap[rid]['boxes'][q:q+1].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=float(candidate_ious(b,gt)[0][0]);mass[r['platform'],spec['removed_count_matched'],rid]=v;o=originalmass[spec['name'],rid];numclose(o['iou'],v,1e-12);assert o['query_PRIVATE']==q and o['target_alias']==r['alias'] and o['split']==r['split'];counts['mass_selected_box_task_values_recomputed']+=1
 # independently audited3800rows supply all/LOO comparisons and source values.
 loo=read(OUT/'ALL3800_RECONSTRUCTED_ROWS_PRIVATE.json');assert len(loo)==3800;raw=read(ROOT/'research/round31/experiment/LOO_VS_MASS_ROWS.json');raw={(r['feedback_alias'],r['target_alias']):r for r in raw};assert len(raw)==3800;loo_specs=read(ROOT/'crossplatform/round31/loo19_attempt002/MODEL_SPECS_PRIVATE.json');lometa={x['omitted_alias']:x for x in loo_specs if x['kind']=='leave_one_observation'};residual=[]
 for x in loo:
  spec=lometa[x['feedback_alias']];k=spec['original_N']-spec['active_preferences'];v=mass[x['platform'],k,x['target_id_PRIVATE']];massdelta=100*(v-x['all_iou']);resid=100*(x['minus_iou']-v);o=raw[x['feedback_alias'],x['target_alias']];numclose(o['loo_delta_remove_pp'],x['delta_remove_pp'],1e-10);numclose(o['uniform_mass_delta_pp'],massdelta,1e-10);numclose(o['loo_minus_uniform_mass_pp'],resid,1e-10);numclose(o['loo_delta_remove_pp']-o['uniform_mass_delta_pp'],resid,1e-10);assert o['is_removed_observation']==x['is_removed_observation'];residual.append(dict(o));counts['LOO_mass_residual_rows_recomputed']+=1
 summaries={x['feedback_alias']:x for x in read(ROOT/'research/round31/experiment/LOO_VS_MASS_SUMMARY.json')};resummary=[]
 for alias,spec in lometa.items():
  ss=summaries[alias];N=spec['original_N'];k=N-spec['active_preferences'];numclose(ss['uniform_coefficient'],(N-k)/N,1e-14);rs=[r for r in residual if r['feedback_alias']==alias];out={'feedback_alias':alias,'platform':spec['platform'],'removed_preferences':k,'uniform_coefficient':(N-k)/N,'subsets':{}}
  for split in ['A208','DEV64','R64','S64','R64_S64']:
   targets=[r for r in rs if not r['is_removed_observation'] and (r['split'] in ['R64','S64'] if split=='R64_S64' else r['split']==split)];ag={'n':len(targets)}
   for key in ['loo_delta_remove_pp','uniform_mass_delta_pp','loo_minus_uniform_mass_pp']:ag[key]=float(np.mean([r[key] for r in targets]));numclose(ss['subsets'][split][key],ag[key],1e-10)
   assert ss['subsets'][split]['n']==len(targets);out['subsets'][split]=ag;counts['LOO_mass_aggregate_subsets_recomputed']+=1
  resummary.append(out)
 # Cohort-level mass effects, with fixed allmodel reference and no oracle selection.
 alldict={}
 for x in loo:alldict[x['target_id_PRIVATE']]=x['all_iou']
 cohort=[];witness=[]
 for spec in specs:
  p=masspred[spec['name']];pl=spec['platform'];k=spec['removed_count_matched']
  for split in ['A208','DEV64','R64','S64','R64_S64']:
   targets=[r for r in rr if r['platform']==pl and (r['split'] in ['R64','S64'] if split=='R64_S64' else r['split']==split)];d=np.array([100*(mass[pl,k,r['id']]-alldict[r['id']]) for r in targets]);cohort.append({'model':spec['name'],'platform':pl,'coefficient':spec['uniform_weight'],'split':split,'n':len(targets),'mass_mIoU_percent':float(np.mean([mass[pl,k,r['id']] for r in targets])*100),'all_mIoU_percent':float(np.mean([alldict[r['id']] for r in targets])*100),'same_platform_delta_pp':float(d.mean()),'balanced_two_platform_delta_pp':float(d.mean()/2),'improves':int((d>1e-8).sum()),'worsens':int((d< -1e-8).sum())})
  for alias in ['S_drone_005','S_quad_016']:
   r=aliaslookup[alias]
   if r['platform']==pl:witness.append({'model':spec['name'],'target_alias':alias,'all_iou':alldict[r['id']],'mass_iou':mass[pl,k,r['id']],'delta_pp':100*(mass[pl,k,r['id']]-alldict[r['id']])})
 assert all(abs(x['same_platform_delta_pp'])<1e-12 for x in cohort if x['platform']=='drone' and x['split']=='S64')
 q2=next(x for x in cohort if x['model']=='quad_mass_2' and x['split']=='S64');numclose(q2['same_platform_delta_pp'],2.143912488767659,1e-10)
 q1=next(x for x in cohort if x['model']=='quad_mass_1' and x['split']=='S64');numclose(q1['same_platform_delta_pp'],0,1e-12)
 wd=next(x for x in resummary if x['feedback_alias']=='W_drone_012');numclose(wd['subsets']['S64']['loo_minus_uniform_mass_pp'],3.787801580556434,1e-10)
 dump(OUT/'MASS_CONTROL_NOMINAL_CONTRACT.json',nominal);dump(OUT/'LOO_VS_MASS_INDEPENDENT_SUMMARY.json',resummary);dump(OUT/'MASS_COHORT_INDEPENDENT_SUMMARY.json',cohort);dump(OUT/'MASS_REQUESTED_WITNESSES.json',witness);dump(OUT/'ALL3800_MASS_RESIDUALS_PRIVATE.json',residual)
 result={'state':'PASS','reviewer':'independent_CPU_subagent','models':4,'counts':dict(counts),'coefficients':{'drone':[.9,.8],'quad':[11/12,10/12]},'all_feedback_terms_retained':True,'fixed_original_denominators_and_U_retained':True,'historical_active_preferences_field_is_int_sumweights_not_nonzero_terms':True,'initial_objective_max_error':max(initial_errors),'CPU_state_scores_max_error':max(score_errors),'CPU_final_pair_gaps_max_error':max(gap_errors),'Drone_S64_mass_control_task_delta_zero_both_coefficients':True,'Quad_S64_c11over12_delta_pp':q1['same_platform_delta_pp'],'Quad_S64_c10over12_delta_pp':q2['same_platform_delta_pp'],'W_drone012_S64_residual_vs_uniform_mass_pp':wd['subsets']['S64']['loo_minus_uniform_mass_pp'],'all800_mass_selected_task_values_recomputed':True,'all3800_LOO_mass_residuals_recomputed':True,'all95_LOO_mass_subset_aggregates_recomputed':True,'no_GPU_training_or_Qwen_invoked':True,'limits':['Uniform nominalweight mass does not equal matching gradientnorm/direction or effectiveAdam step.','Control registered after originalLOO identifies a posthoc confound; no new method or independent performance confirmation.','IndividualLOO residuals contain nonlinear optimization interactions and cannot be treated as linearly additive causal feedback values.','Per-feedback summaries affect oneplatform; two-platformbalanced impact is half same-platformmean with the otherstate fixed.','No coefficient is selected as a deployment rule from R/S outcomes.'],'seconds':time.perf_counter()-start};dump(OUT/'MASS_AUDIT_RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
