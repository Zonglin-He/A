"""R51 selected GROUPED full5912 FIVE readout assembler; no execution on import.

New-method scores/statuses are copied only from pure single-factor safe and
no-association seals. R50 cached scores supply source/external baselines ONLY;
no old visual arrays, crops, ownership or feedback are materialized.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
ARMS=['source','external_G','replace_all','no_relation','no_instance_association']
FACTOR_NAMES={.5:('grouped_replace_all_c05','no_instance_association_c05')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p);return p
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def npz(p,fields):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in fields}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_SELECTED_GROUPED_FULL_FIVE_READOUT_ASSEMBLY_NO_GT'
 assert auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_SELECTED_GROUPED_FULL_FIVE_READOUT_ASSEMBLY' and plan['n']==5912 and plan['manifest']==MANIFEST
 assert plan['arm_names']==ARMS and plan['method_selected_before_full_run'] is True and plan['new_full_calibration'] is False
 factor=plan['selected_factor'];assert factor in FACTOR_NAMES and plan['selected_method']=='grouped_replace_all_c05'
 safe_arm,na_arm=FACTOR_NAMES[factor]
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():assert sha(path(rel))==h,('pin',rel)
 assert plan['programs_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]==sha(__file__)
 assert plan['method_freeze'] in plan['inputs_sha256'] and auth['method_freeze_sha256']==sha(path(plan['method_freeze']))
 roots={key:path(plan[key]) for key in ('safe_root','noassociation_root','r50_prediction_root')};seals={key:read(p/'SEALED.json') for key,p in roots.items()}
 for key,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 assert seals['safe_root']['state']=='R51_FULL_GROUP_MEDOID_G_PREDICTIONS_NO_GT' and seals['safe_root']['single_registered_mechanism_repair'] is True and seals['safe_root']['method_selected_before_full_run'] is True
 assert seals['noassociation_root']['state']=='R51_RAW_SAFE_NO_INSTANCE_ASSOCIATION_ABLATION_COMPLETE_NO_GT'
 for key in ('safe_root','noassociation_root'):
  assert seals[key]['n']==5912 and seals[key]['appearance_factor_values']==[factor] and seals[key]['external_vision_outputs_or_mask_derived_caches_read'] is False
 assert seals['r50_prediction_root']['state']=='R50_FULL5912_THREE_FIXED_BOX_READOUTS_BEFORE_TASK_SCORE' and seals['r50_prediction_root']['n']==5912
 def bound(key,rel):
  p=roots[key]/rel;assert sha(p)==seals[key]['files'][rel];return p
 assert MANIFEST in plan['inputs_sha256'];rows=read(ROOT/MANIFEST)
 assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 assert sum(r['platform']=='drone' for r in rows)==2984 and sum(r['platform']=='quad' for r in rows)==2928
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir(parents=True)
 for name in ('samples_PRIVATE','cases_PRIVATE'):(out/name).mkdir()
 dump(out/'RUN_BINDINGS.json',{'plan_sha256':sha(args.plan),'method_freeze_sha256':sha(path(plan['method_freeze'])),'selected_factor':factor,'arm_names':ARMS,'new_full_calibration':False,'TTA_not_included_or_claimed':True})
 done=[];statuses=[];started=time.perf_counter()
 try:
  for i,row in enumerate(rows):
   sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';sp=bound('safe_root',rel);nap=bound('noassociation_root',rel);op=bound('r50_prediction_root',rel)
   new=npz(sp,['arm_names','scores','queries','selected_original_boxes','source_boxes','source_scores','appearance_factor_values','no_relation_scores','no_relation_queries','no_relation_selected_original_boxes','cue_query_known'])
   noassoc=npz(nap,['arm_names','scores','queries','selected_original_boxes','source_boxes','source_scores','appearance_factor_values'])
   old=npz(op,['arm_names','scores','queries','selected_original_boxes','source_boxes','source_scores','wide_pool'])
   assert new['arm_names'].tolist()==['source',safe_arm,'grouped_no_relation_c05'] and noassoc['arm_names'].tolist()==[na_arm]
   assert new['appearance_factor_values'].tolist()==noassoc['appearance_factor_values'].tolist()==[factor]
   assert old['arm_names'].tolist()==['source','old_G','wideentry_oldW']
   boxes=old['source_boxes'];source=old['source_scores'];assert boxes.shape==(256,6) and source.shape==(256,)
   for z in (new,noassoc):assert exact(z['source_boxes'],boxes) and exact(z['source_scores'],source)
   assert exact(new['scores'][0],old['scores'][0]) and int(new['queries'][0])==int(old['queries'][0])
   fs=[old['scores'][0],old['scores'][2],new['scores'][1],new['no_relation_scores'][0],noassoc['scores'][0]]
   scores=np.stack(fs);queries=scores.argmax(1).astype(np.int64);selected=boxes[queries]
   expected_q=np.asarray([old['queries'][0],old['queries'][2],new['queries'][1],new['no_relation_queries'][0],noassoc['queries'][0]],np.int64)
   assert np.array_equal(queries,expected_q)
   expected_boxes=np.stack([old['selected_original_boxes'][0],old['selected_original_boxes'][2],new['selected_original_boxes'][1],new['no_relation_selected_original_boxes'][0],noassoc['selected_original_boxes'][0]])
   assert exact(selected,expected_boxes) and np.isfinite(scores).all()
   sc=read(bound('safe_root','cases_PRIVATE/'+sid+'.json'));nc=read(bound('noassociation_root','cases_PRIVATE/'+sid+'.json'))
   assert sc['external_vision_outputs_or_mask_derived_caches_read'] is False and nc['external_vision_outputs_or_mask_derived_caches_read'] is False
   unavailable=not bool(new['cue_query_known'][0,:2].any());ext_empty=len(old['wide_pool'])==0
   status={'source':{'unknown':False,'fallback':False,'no_parameter_update':True},'external_G':{'unknown':ext_empty,'fallback':ext_empty,'no_parameter_update':True},'replace_all':sc['arm_statuses'][safe_arm],'no_relation':sc['arm_statuses']['grouped_no_relation_c05'],'no_instance_association':nc['arm_statuses'][na_arm]}
   for ss in status.values():assert set(ss)=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in ss.values())
   np_save={'arm_names':np.asarray(ARMS),'scores':scores,'queries':queries,'selected_original_boxes':selected,'source_boxes':boxes,'source_scores':source}
   # Avoid any feedback/transformed array save: only fixed readout/canonical data.
   with (out/'samples_PRIVATE'/(sid+'.npz')).open('wb') as stream:np.savez_compressed(stream,**np_save)
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'selected_factor':factor,'selected_method':plan['selected_method'],'source_external_bridge_bitexact':True,'new_method_scores_copied_not_recomputed':True,'old_visual_data_used_only_by_external_baseline':True,'TTA_included':False,'input_sample_sha256':{'safe':sha(sp),'noassociation':sha(nap),'R50_baseline':sha(op)}})
   statuses.append({'id':sid,'arm_statuses':status});done.append(sid)
   if (i+1)%128==0:print('R51_GROUPED_FULL_FIVE_FRAMES',i+1,5912,flush=True)
  dump(out/'READOUT_STATUS_PRIVATE.json',statuses)
  dump(out/'COSTS.json',{'seconds':time.perf_counter()-started,'new_model_forwards':0,'optimizer_steps':0,'cache_only_assembly_not_end_to_end_method_latency':True})
  dump(out/'SEALED.json',{'state':'R51_SELECTED_GROUPED_FULL5912_FIVE_READOUTS_ASSEMBLED_NO_GT','n':len(done),'arm_names':ARMS,'selected_factor':factor,'selected_method':plan['selected_method'],'plan_sha256':sha(args.plan),'method_freeze_sha256':sha(path(plan['method_freeze'])),'no_GT_or_models_or_new_boxes':True,'TTA_included':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry_or_GT_score':True});raise

if __name__=='__main__':main()
