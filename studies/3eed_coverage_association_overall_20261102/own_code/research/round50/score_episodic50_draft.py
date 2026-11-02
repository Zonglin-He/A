"""UNEXECUTED single post-seal DEV336 six-arm episodic score.

No model, optimizer or author algorithm import. This author-prepared scorer
requires independent review plus a separate root GT-scoring grant. All 336,
including rejection and zero-update rows, remain in the overall denominator.
"""
from pathlib import Path
import sys,json,hashlib,time,collections,itertools,argparse,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}
def csv_rows(path,fields,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def verify(folder):
 folder=Path(folder);s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(folder/rel)==h
 return s
def scoring_index(rows):
 import ijson
 path=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json';receipt=read(ROOT/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json')
 assert sha(path)==receipt['scoring_index_sha256'];wanted={r['id']:r for r in rows};out={}
 with path.open('rb') as f:
  for e in ijson.items(f,'item'):
   if e['id'] not in wanted:continue
   r=wanted[e['id']];assert e['platform']==r['platform'] and e['scene']==r['scene'] and Path(e['meta_path']).resolve()==Path(r['metapath']).resolve()
   assert type(e['object_index']) is int and e['object_index']>=0 and e['id'] not in out
   out[e['id']]={k:e[k] for k in ('meta_path','object_index','metadata_container_sha256')}
 assert set(out)==set(wanted);return out
def summary_of(tasks):
 summary={};scenes=[]
 for split in ('ALL336','D208','V64','M64'):
  rr=tasks if split=='ALL336' else [r for r in tasks if r['split']==split];part={}
  for platform in ('drone','quad'):
   pr=[r for r in rr if r['platform']==platform];assert pr;part[platform]={}
   for arm in ARMS:
    v=np.array([r['IoU'][arm] for r in pr]);d0=v-np.array([r['IoU']['sameinfo_consensus0'] for r in pr]);dg=v-np.array([r['IoU']['literal_A_wideG'] for r in pr])
    part[platform][arm]={'n':len(pr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),
     'gain_vs_consensus0_pp':float(d0.mean()*100),'gain_vs_literal_A_wideG_pp':float(dg.mean()*100),
     'improved_vs_consensus0':int((d0>1e-10).sum()),'worsened_vs_consensus0':int((d0< -1e-10).sum()),
     'improved_vs_literal_A_wideG':int((dg>1e-10).sum()),'worsened_vs_literal_A_wideG':int((dg< -1e-10).sum()),
     'query_changes_vs_consensus0':sum(r['queries'][arm]!=r['queries']['sameinfo_consensus0'] for r in pr),
     'accepted_episode_n':sum(r['accepted'] for r in pr),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for scene in sorted({r['scene_PRIVATE'] for r in pr}):
    sr=[r for r in pr if r['scene_PRIVATE']==scene];scenes.append({'split':split,'platform':platform,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{arm:float(np.mean([r['IoU'][arm] for r in sr])*100) for arm in ARMS}})
  part['balanced_mIoU_percent']={arm:float(np.mean([part[p][arm]['mIoU_percent'] for p in ('drone','quad')])) for arm in ARMS};summary[split]=part
 return summary,scenes
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 a=p.parse_args();start=time.perf_counter();plan=read(a.plan);grant=read(a.authorization)
 assert grant['state']=='ROOT_AUTHORIZED_R50_SINGLE_DEV_EPISODIC_POSTSEAL_TASK_SCORE'
 assert grant['plan_sha256']==sha(a.plan) and grant['program_sha256']==sha(__file__)
 assert plan['arms']==ARMS and plan['n']==336 and plan['task_values']==2016
 assert plan['selection_rule']=='ALL336_balanced_max_step1_or3_exact_tie1_strictgain_vs_consensus0_and_literalAwideG'
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'FINAL' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel and sha(ROOT/rel)==h
 pred=ROOT/plan['prediction_root'];assert pred.is_relative_to(ROOT/'crossplatform/round50')
 assert grant['prediction_seal_sha256']==sha(pred/'SEALED.json')
 auditpath=ROOT/plan['independent_audit_result'];assert sha(auditpath)==grant['independent_audit_sha256'];audit=read(auditpath)
 assert audit['state']=='PASS_R50_INDEPENDENT_EPISODIC_NUMERIC_AND_CONTRACT_AUDIT_NO_GT' and audit['n']==336 and audit['arms']==6 and audit['no_GT_read'] is True
 assert audit['prediction_seal_sha256']==sha(pred/'SEALED.json')
 changed=audit['nonzero_parameter_element_count'];assert type(changed) is int and changed>=0
 if changed>0:
  assert audit['installed_closure_requirement_satisfied'] is True
  measurement_mode='ACTUAL_NONZERO_EPISODIC_STATES_WITH_INSTALLED_CLOSURE'
 else:
  # Root registered this exception before any TTA GT access. It can establish
  # a pure no-update comparison, never a fictitious adapted closure or gain.
  assert plan.get('allow_verified_no_parameter_update_branch') is True
  assert audit['all_saved_heads_source_bytes_identical'] is True and audit['all_no_update_outputs_equal_s0'] is True
  assert audit['installed_closure_state']=='NOT_APPLICABLE_VERIFIED_NO_PARAMETER_UPDATE'
  assert audit['installed_closure_requirement_satisfied'] is False
  measurement_mode='VERIFIED_NO_PARAMETER_UPDATE_TTA_NOT_ACTUALLY_TESTED'
 ps=verify(pred);assert ps['n']==336 and ps['arm_names']==ARMS
 rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');assert len(rows)==336 and len({r['id'] for r in rows})==336
 split={r['id']:s for s in ('D208','V64','M64') for r in read(ROOT/'research/round24/input'/(s+'_PRIVATE.json'))};assert set(split)=={r['id'] for r in rows}
 oldroot=ROOT/'research/round50/independent_score/dev336_attempt001';oldseal=verify(oldroot)
 old={r['id_PRIVATE']:r for r in read(oldroot/'ALL336_TASK_ROWS_PRIVATE.json')}
 assert (Path(plan['prediction_root'])/'SEALED.json').as_posix() in plan['inputs_sha256']
 out=ROOT/plan['output_root'];assert out.is_relative_to(ROOT/'research/round50') and not out.exists();out.mkdir(parents=True)
 # GT machinery/index are first entered only after all grant/seal/audit gates.
 sys.path.insert(0,str(ROOT/'research/round5/audit'));from independent_core_v2 import candidate_ious,transform_gt
 index=scoring_index(rows);containers={};tasks=[]
 try:
  for i,r in enumerate(rows):
   z=load(pred/'samples_PRIVATE'/(r['id']+'.npz'));case=read(pred/'cases_PRIVATE'/(r['id']+'.json'))
   assert z['arm_names'].tolist()==ARMS and z['queries'].shape==(6,) and z['selected_original_boxes'].shape==(6,6)
   assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']])
   assert int(z['queries'][2])==int(z['queries'][1]) and z['selected_original_boxes'][2].tobytes()==z['selected_original_boxes'][1].tobytes(),'group_A_G_must_preserve_literal_A_G'
   ix=index[r['id']];mp=ix['meta_path']
   if mp not in containers:assert sha(mp)==ix['metadata_container_sha256'];containers[mp]=read(mp)
   obj=containers[mp]['ground_info'][ix['object_index']];assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption']
   gt=transform_gt(obj['bbox_3d'],r['platform']);boxes=z['selected_original_boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);vals=candidate_ious(boxes,gt)[0]
   assert abs(float(vals[0])-old[r['id']]['IoU']['source'])<1e-12 and abs(float(vals[1])-old[r['id']]['IoU']['wideentry_oldW'])<1e-12
   tasks.append({'row':f'E{i+1:04d}','id_PRIVATE':r['id'],'platform':r['platform'],'split':split[r['id']],'scene_PRIVATE':r['scene'],
    'accepted':bool(case['accepted']),'nonzero_parameter_updates':case['nonzero_parameter_updates'],'reason':case['reason'],
    'queries':dict(zip(ARMS,map(int,z['queries']))),'IoU':dict(zip(ARMS,map(float,vals)))})
 except BaseException as e:
  dump(out/'PARTIAL_ROWS_PRIVATE.json',tasks);dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(tasks),'no_automatic_GT_rerun':True});raise
 summary,scenes=summary_of(tasks);neg=[]
 for row in tasks:
  for arm,ref in itertools.product(ARMS,repeat=2):
   d=row['IoU'][arm]-row['IoU'][ref]
   if arm!=ref and d< -1e-10:neg.append({'row':row['row'],'platform':row['platform'],'split':row['split'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
 m=summary['ALL336']['balanced_mIoU_percent'];step=1 if m['sameinfo_consensus1']>=m['sameinfo_consensus3'] else 3
 arm='sameinfo_consensus'+str(step);d0=m[arm]-m['sameinfo_consensus0'];dg=m[arm]-m['literal_A_wideG'];E=sum(r['accepted'] for r in tasks)
 gate={'state':'R50_EXPOSED_DEV_EPISODIC_SHARED_STEP_SELECTION_ALL_TRIALS_RETAINED','selected_step':step,'exact_tie_priority':1,
  'selected_arm':arm,'accepted_episodes':E,'all336_denominator':336,'gain_vs_sameinfo_consensus0_pp':d0,'gain_vs_literal_A_wideG_pp':dg,
  'gate_PASS':bool(E>0 and changed>0 and d0>0 and dg>0),'nonzero_parameter_element_count':changed,'measurement_mode':measurement_mode,
  'all_zero_parameters_force_STOP':bool(changed==0),'all_six_arms_and_negatives_retained':True,'selection_is_exposed_DEV_label_based':True,
  'no_GT_classspan_or_feedback_filtering_or_R8_replacement':True,'full_TTA_not_authorized_by_this_receipt':True}
 dump(out/'ALL336_TASK_ROWS_PRIVATE.json',tasks);dump(out/'TASK_SUMMARY.json',summary);dump(out/'SCENE_SUMMARIES_PRIVATE.json',scenes)
 dump(out/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',neg);dump(out/'DEV_STEP_SELECTION_AND_GATE.json',gate)
 scalar=[{'row':r['row'],'platform':r['platform'],'split':r['split'],'arm':arm,'IoU':r['IoU'][arm],
   'accepted':r['accepted'],'nonzero_s1':r['nonzero_parameter_updates']['1'],'nonzero_s3':r['nonzero_parameter_updates']['3']} for r in tasks for arm in ARMS]
 csv_rows(out/'ALL2016_ANONYMOUS_TASK_SCALARS.csv',['row','platform','split','arm','IoU','accepted','nonzero_s1','nonzero_s3'],scalar)
 csv_rows(out/'ALL_ANONYMOUS_ORDERED_NEGATIVES.csv',['row','platform','split','arm','reference','delta_pp'],neg)
 counts={'n':336,'rejected_n':336-E,'accepted_n':E,'nonzero_s1_n':sum(r['nonzero_parameter_updates']['1'] for r in tasks),'nonzero_s3_n':sum(r['nonzero_parameter_updates']['3'] for r in tasks)}
 dump(out/'CONDITION_COUNTS.json',counts)
 receipt={'state':'PASS_R50_SINGLE_DEV336_EPISODIC_TASK_SCORE','n':336,'arms':6,'task_values':2016,'historical_source_and_literalG_replayed_n':336,
  'ordered_negative_comparisons':len(neg),'gate_PASS':gate['gate_PASS'],'program_sha256':sha(__file__),'prediction_seal_sha256':sha(pred/'SEALED.json'),
  'independent_audit_sha256':sha(auditpath),'measurement_mode':measurement_mode,'nonzero_parameter_element_count':changed,
  'seconds':time.perf_counter()-start,'single_GT_scoring_pass':True,'no_method_or_model_changes':True}
 dump(out/'SCORING_RECEIPT.json',receipt);dump(out/'SEALED.json',{'state':receipt['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
 print(json.dumps({'receipt':receipt,'gate':gate,'balanced_mIoU_percent':m}),flush=True)
if __name__=='__main__':main()
