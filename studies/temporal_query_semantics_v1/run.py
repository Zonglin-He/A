from common import *
import argparse,csv,ast,shutil

def prepare():
 assert not(D/'plan.json').exists();torch.set_num_threads(1);inv=read(D/'inventory_v3.json');pairs=inv['selected'];assert len(pairs)==32 and len({x['a']['video']for x in pairs})==32
 allrows=old.selected.records('vidstg');keys={x[s]['id']for x in pairs for s in ['a','b']};records=[r for r in allrows if r['id']in keys];assert len(records)==64;atomic(D/'records.json',records)
 cohort=[]
 for i,x in enumerate(pairs,1):
  a,b=x['a'],x['b'];assert a['target']==b['target']==a['relation']['subject_tid']==b['relation']['subject_tid']and a['relation']['object_tid']==b['relation']['object_tid']and a['video_sha']==b['video_sha']and a['frame_ids']==b['frame_ids']and a['relation']['predicate']!=b['relation']['predicate']and x['GT_event_tIoU']<=.2
  cohort.append(dict(pair=f'P{i:03}',a=a['id'],b=b['id'],video_sha256=a['video_sha'],selection_order_sha256=x['order_sha256']))
 atomic(D/'cohort.json',cohort);cases=[]
 for x in cohort[:2]:
  a,pa=source_inputs(x['a']);b,pb=source_inputs(x['b']);z=derive(a,b);again=derive(a,b);assert all(np.array_equal(v,again[k])for k,v in z.items());cases.append(dict(pair=x['pair'],actual_frozen_cache_own_L_posterior_interval_exact=True,all_counterfactual_repeat_exact=True))
 atomic(D/'preflight.json',dict(status='passed_two_real_pairs_four_cached_queries',cases=cases,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_GT_score=0))
 inputs=[D/'inventory_v3.json',D/'records.json',D/'cohort.json',D/'preflight.json',R/'data/annotations/vidstg_test.json',R/'data/annotations/vidstg_tastvg_eval.json',R/'data/annotations/hc2_tastvg_eval.json',OLD/'temporal/vidstg/sealed.json',OLD/'temporal/vidstg/metrics.json',OLD/'temporal/vidstg/reporting_receipt.json',old.selected.PREV/'spatial/vidstg_sealed.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 code=list(S.glob('*.py'))+[R/'stvg/prototype.py',R/'stvg/current_shared_roi.py',R/'studies/reviewer_key_controls_v1/common_v1.py']
 p=dict(schema='same_target_different_action_temporal_diagnostic_v1',authorization='2026-11-04 direct user execute experiment to address temporal event semantics reviewer concern',cohort=dict(dataset='VidSTG',pairs=32,queries=64,independent_videos=32,same_target_actor_track=True,same_interaction_object_track=True,exact_same_physical_video_SHA_and_native_frame_grid=True,different_official_nonstatic_action_predicates=True,GT_event_tIoU_max=.2,source_catalog_eligible_pairs=inv['summary']['eligible_pairs'],source_catalog_eligible_videos=inv['summary']['eligible_videos'],ambiguous_relation_queries_excluded=34,duplicate_identical_catalog_records=442,selection='seed20261104 SHA-first onepair/video,max32,prior to model outcome access',GT_selected_offline_diagnostic_not_inference_or_unseen_confirmation=True,HC_scope='no reliable explicit cross-query target-track identity in HC catalog, not asserted same-person from video alone'),arms=ARMS,comparison='Natural query pairs have same actor+object but other wording, Source outputs, E regions and planned eight positions can differ. Their exact equality rates are recorded; no same-observation causal claim across queries. SourceNative/JointPrior/SourceROI/CorrectedROI reuse original inputs and rederive current temporal output exactly; feature/prior cross2x2, crossEvidence and common symmetric prior are mechanism interventions, not deployable baselines.',text_control='one_step(features,q0,kappa,beta) has no text argument; metadata-only action text substitution with upstream fixed is an explicit structural invariance control, not endtoend Source text swapping',neutral_control='allzero features must exactly return q0',params=dict(kappa=0,beta=1,rounds=1,context=1.5,eight_observations_per_original_query_unchanged=True),selection_criteria_no_model_outcome=True,separate_all64_seal_before_offline_GT_score=True,stats='query means within eachpair then equal32independentvideo pairs;10000pairedvideo bootstrap seed20261104; pointwise no familywise correction; tIoU improvement/harm/severe>5pp/rescue; correct-vs-swapped GT assignment margin, exclusive-action-frame contrast and both-events tIoU>=.3',new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0,no_method_or_paper_change=True,resource='max2CPU14-15nice19idleIO1threadCUDAhidden,RAM8GiB/disk80GiB/private160GiB/72h',code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'],flush=True)

def predict():
 plan=validate();guard();torch.set_num_threads(1);start=time.perf_counter();entries={}
 for i,pair in enumerate(read(D/'cohort.json'),1):
  resources(start);a,pa=source_inputs(pair['a']);b,pb=source_inputs(pair['b'])
  for side,x,y,proof in [('a',a,b,pa),('b',b,a,pb)]:
   key=pair[side];z=derive(x,y);path=D/'predictions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
   with path.open('wb')as f:np.savez_compressed(f,**z)
   with np.load(path)as zz:assert set(zz.files)==set(z)and all(np.array_equal(v,zz[k])for k,v in z.items())
   rec=dict(path=str(path.relative_to(D)),sha256=sha(path),pair=pair['pair'],side=side,input_SHA=proof,GT_read=False,worker=identity());rp=D/'receipts'/(key+'.json');atomic(rp,rec);entries[key]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
  atomic(D/'progress.json',dict(pairs=i,total_pairs=32,queries=len(entries),GT_read=False,worker=identity()))
 atomic(D/'sealed.json',dict(status='all64cached_query_intervention_arrays_sealed',files=entries,queries=64,pairs=32,GT_read=False,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,plan_fingerprint=plan['fingerprint']));print('sealed64',flush=True)

def evaluate():
 validate();torch.set_num_threads(1);seal=read(D/'sealed.json');assert seal['queries']==64 and not seal['GT_read'];truth=old.selected.labels('vidstg');prior={r['id']:r for r in read(OLD/'temporal/vidstg/metrics.json')['private_rows']};rows=[];queryrows=[];scorable=0;err=0.
 for pair in read(D/'cohort.json'):
  a,b=pair['a'],pair['b'];entries=[seal['files'][a],seal['files'][b]]
  for e in entries:assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
  with np.load(D/entries[0]['path'])as za,np.load(D/entries[1]['path'])as zb:
   assert np.array_equal(za['ids'],zb['ids']);gta,gtb=truth[a]['interval'],truth[b]['interval'];assert tiou(gta,gtb)<=.21
   r=dict(pair=pair['pair']);vals={}
   for arm in ARMS:
    iva,ivb=za[arm+'_interval'],zb[arm+'_interval'];ta,tb=tiou(iva,gta),tiou(ivb,gtb);cross=.5*(tiou(iva,gtb)+tiou(ivb,gta));mean=.5*(ta+tb);vals[arm]=[ta,tb];margin=100*(mean-cross)
    r[arm+'_changed_vsCorrected_percent']=50.*(int(not np.array_equal(iva,za['CorrectedROI_interval']))+int(not np.array_equal(ivb,zb['CorrectedROI_interval'])))
    r.update({arm+'_t_percent':100*mean,arm+'_assignment_margin_pp':margin,arm+'_correct_assignment_percent':100.*(margin>1e-10),arm+'_both_events_t03_percent':100.*(ta>=.3 and tb>=.3),arm+'_identical_intervals_percent':100.*np.array_equal(iva,ivb),arm+'_interval_overlap':tiou(iva,ivb)})
    for key,t in [(a,ta),(b,tb)]:
     if arm in ['SourceNative','JointPrior','SourceROI','CorrectedROI']:
      oldarm={'SourceNative':'Source','JointPrior':'SourceJointMAP','SourceROI':'SourceTR','CorrectedROI':'ExpertInitTR'}[arm];err=max(err,abs(t-prior[key]['values'][oldarm]['t']));assert abs(t-prior[key]['values'][oldarm]['t'])<1e-12
   for name,aa,bb in [('visual_gain','CorrectedROI','JointPrior'),('total_gain','CorrectedROI','SourceNative'),('own_feature_gain','CorrectedROI','CrossFeatures'),('own_evidence_gain','CorrectedROI','CrossEvidence')]:
    delta=100*(np.asarray(vals[aa])-vals[bb]);r[name+'_t_pp']=float(delta.mean());r[name+'_harmed_percent']=float(100*np.mean(delta< -1e-10));r[name+'_improved_percent']=float(100*np.mean(delta>1e-10));r[name+'_severe_percent']=float(100*np.mean(delta< -5));r[name+'_rescue_percent']=float(100*np.mean(delta>5));r[name+'_assignment_delta_pp']=r[aa+'_assignment_margin_pp']-r[bb+'_assignment_margin_pp']
   ids=za['ids'];ma=(ids>=gta[0])&(ids<gta[1]);mb=(ids>=gtb[0])&(ids<gtb[1]);ea=ma&~mb;eb=mb&~ma;r['exclusive_native_support']=int(ea.any()and eb.any())
   for arm in ['CorrectedROI','CrossFeatures','CrossPrior','CommonPrior']:
    r[arm+'_exclusive_action_L_contrast']=float(.5*((za[arm+'_L'][ea].mean()-za[arm+'_L'][eb].mean())+(zb[arm+'_L'][eb].mean()-zb[arm+'_L'][ea].mean())))if ea.any()and eb.any()else None
   r['same_eight_positions_percent']=100.*np.array_equal(za['positions8'],zb['positions8']);r['E_features_relative_L2']=float(np.linalg.norm(za['input_F']-zb['input_F'])/max(np.linalg.norm(za['input_F'])+np.linalg.norm(zb['input_F']),1e-12));r['feature_swap_L_mean_abs']=float(.5*(np.abs(za['CrossFeatures_L']-za['CorrectedROI_L']).mean()+np.abs(zb['CrossFeatures_L']-zb['CorrectedROI_L']).mean()));r['prior_swap_L_mean_abs']=float(.5*(np.abs(za['CrossPrior_L']-za['CorrectedROI_L']).mean()+np.abs(zb['CrossPrior_L']-zb['CorrectedROI_L']).mean()));r['q0_total_variation']=float(.5*np.abs(za['input_q0']-zb['input_q0']).sum());r['E_features_exact_percent']=100.*np.array_equal(za['input_F'],zb['input_F']);r['text_metadata_q_exact_percent']=100.*(np.array_equal(za['text_metadata_control_q'],za['CorrectedROI_q'])and np.array_equal(zb['text_metadata_control_q'],zb['CorrectedROI_q']));r['text_metadata_L_exact_percent']=100.*(np.array_equal(za['text_metadata_control_L'],za['CorrectedROI_L'])and np.array_equal(zb['text_metadata_control_L'],zb['CorrectedROI_L']));rows.append(r)
   for side,key,z,gt in [('a',a,za,gta),('b',b,zb,gtb)]:queryrows.append(dict(pair=pair['pair'],side=side,**{arm+'_t_percent':100*tiou(z[arm+'_interval'],gt)for arm in ARMS}))
 assert err<1e-12;fields=[k for k in rows[0]if k not in ['pair','exclusive_native_support']];report=dict(status='offline_pair_semantic_temporal_statistics_PENDING_independent_audit',summary=bootstrap(rows,fields),exclusive_action_frame_scorable_pairs=sum(r['exclusive_native_support']for r in rows),reused_official_tIoU_checks=64*4,max_reused_official_discrepancy=err,cohort_selected_using_catalog_only=True,GT_for_offline_diagnostic_only=True,pointwise_video_bootstrap_not_unknown_test=True);atomic(P/'report.json',report);atomic(D/'private_pair_rows.json',rows)
 for name,data in [('all32_pair_signed.tsv',rows),('all64_query_tIoU.tsv',queryrows)]:
  with(P/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(data)
 atomic(D/'evaluation_receipt.json',dict(status='complete_offline_temporal_interval_analysis_PENDING_audit',seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),rows_sha256=sha(D/'private_pair_rows.json'),reused_official_checks=256,new_GT_tube_scoring=0,new_model=0,worker=identity()));print('evaluated32pairs',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','evaluate']);a=p.parse_args();globals()[a.stage]()
