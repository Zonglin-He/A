"""Local public export only; no network mutation, assets, weights or opaque IDs."""
import sys,io,csv,shutil,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round31'))
from common31 import *

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def main():
 ia=EXP.parent/'independent_audit';assert read(ia/'AUDIT_RESULT.json')['state']==read(ia/'MASS_AUDIT_RESULT.json')['state']=='PASS';assert read(EXP/'DECISION.json')['run']=='completed'
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(src,dst):
  p=out/dst;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
 def write_csv(dst,records,fields=None):
  assert records;fields=fields or list(records[0]);buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(records);p=out/dst;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue())
 for folder in [BASE,EXP.parent,ia,EXP.parent/'independent_identity']:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 cp(EXP/'REPORT.md','REPORT.md')
 for n in ['PLAN_LOCK','MASS_CONTROL_LOCK','ENGINEERING_REPAIR','FINAL_CODE_SNAPSHOT']:cp(EXP/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['DECISION','BASELINE_REPRODUCTION','INFLUENCE19_SUMMARY','LOO_VS_MASS_SUMMARY','REQUESTED_WITNESSES']:cp(EXP/(n+'.json'),Path('results')/(n+'.json'))
 for n in ['AUDIT_RESULT','MASS_AUDIT_RESULT','BALANCED_SYSTEM_EFFECTS','INFLUENCE19_INDEPENDENT_SUMMARY','LOO_VS_MASS_INDEPENDENT_SUMMARY','MASS_COHORT_INDEPENDENT_SUMMARY','MASS_CONTROL_NOMINAL_CONTRACT','MASS_REQUESTED_WITNESSES','SCENE_INFLUENCE_INDEPENDENT_SUMMARY','ZERO_TASK_EFFECT_DELETIONS']:dump(out/'audit'/(n+'.json'),clean(read(ia/(n+'.json'))))
 for folder,label in [('loo19_attempt002','MAIN23'),('masscontrol_attempt001','MASS4')]:
  dump(out/'results'/(label+'_FIT_RECEIPTS.json'),clean(read(BASE/folder/'FIT_RECEIPTS_PRIVATE.json')));dump(out/'results'/(label+'_MODEL_SPECS.json'),clean(read(BASE/folder/'MODEL_SPECS_PRIVATE.json')))
 identity=read(EXP.parent/'independent_identity/IDENTITY_REVIEW.json');identity.pop('viewed_files',None);dump(out/'review/A_DRONE034_IDENTITY_REVIEW.json',identity)
 cp(EXP.parent/'literature/RESEARCH.md','literature/RESEARCH.md');src=read(EXP.parent/'literature/SOURCES.json')
 for x in src.get('sources',[]):x.pop('evidence_snapshot',None)
 dump(out/'literature/SOURCES.json',src)
 rr=read(EXP/'ALL_INFLUENCE_ROWS_PRIVATE.json');assert len(rr)==3800;scene_values={p:sorted({r['target_scene_PRIVATE'] for r in rr if r['platform']==p}) for p in ['drone','quad']};export=[]
 for r in rr:
  x=clean(r);x['scene_alias']=r['platform']+'_s'+str(scene_values[r['platform']].index(r['target_scene_PRIVATE']));export.append(x)
 write_csv('results/ALL3800_INFLUENCE_ROWS.csv',export)
 write_csv('results/ALL3800_LOO_MASS_RESIDUAL_ROWS.csv',read(EXP/'LOO_VS_MASS_ROWS.json'))
 write_csv('results/ALL800_MASS_TASK_ROWS.csv',[clean(r) for r in read(EXP/'MASS_CONTROL_ALL_ROWS_PRIVATE.json')])
 # Baseline/zero rows are shared and retained explicitly rather than inventing cross-platform runs.
 baseline={}
 for r in rr:
  key=r['platform'],r['target_alias']
  val={'platform':r['platform'],'target_alias':r['target_alias'],'split':r['split'],'scene_alias':r['platform']+'_s'+str(scene_values[r['platform']].index(r['target_scene_PRIVATE'])),'source_iou':r['source_iou'],'all_iou':r['all_iou']}
  if key in baseline:assert baseline[key]==val
  else:baseline[key]=val
 assert len(baseline)==400
 write_csv('results/ALL400_BASELINE_TASK_ROWS.csv',list(baseline.values()))
 task=[]
 for r in baseline.values():
  for kind,key in [('all','all_iou'),('zero','source_iou')]:task.append({'model':r['platform']+'_'+kind,'platform':r['platform'],'target_alias':r['target_alias'],'split':r['split'],'scene_alias':r['scene_alias'],'iou':r[key]})
 for r in export:task.append({'model':r['model'],'platform':r['platform'],'target_alias':r['target_alias'],'split':r['split'],'scene_alias':r['scene_alias'],'iou':r['minus_iou']})
 assert len(task)==4600;write_csv('results/ALL4600_MAIN_MODEL_TASK_ROWS.csv',task)
 mass={r['feedback_alias']:r for r in read(EXP/'LOO_VS_MASS_SUMMARY.json')};summ=[]
 for r in read(EXP/'INFLUENCE19_SUMMARY.json'):
  s=r['subsets']['S64'];summ.append({'feedback_alias':r['feedback_alias'],'platform':r['platform'],'removed_preferences':r['removed_preferences'],'A_other_n':r['subsets']['A208']['n'],'A_other_delta_pp':r['subsets']['A208']['mean_delta_remove_pp'],'A_no_feedback_delta_pp':r['subsets']['A_no_feedback']['mean_delta_remove_pp'],'R64_delta_same_platform_pp':r['subsets']['R64']['mean_delta_remove_pp'],'S64_delta_same_platform_pp':s['mean_delta_remove_pp'],'S64_balanced_system_delta_pp':s['mean_delta_remove_pp']/2,'R64_S64_delta_same_platform_pp':r['subsets']['R64_S64']['mean_delta_remove_pp'],'S64_improves_if_removed':s['improves_if_removed'],'S64_worsens_if_removed':s['worsens_if_removed'],'S64_query_changes':s['changed_query'],'S64_original_harm_subset_improvement_sum_pp':s['baseline_harms_recovered_pp_sum'],'S64_original_gain_subset_decline_sum_pp':s['baseline_gains_lost_pp_sum'],'S64_uniform_mass_delta_pp':mass[r['feedback_alias']]['subsets']['S64']['uniform_mass_delta_pp'],'S64_LOO_minus_uniform_mass_pp':mass[r['feedback_alias']]['subsets']['S64']['loo_minus_uniform_mass_pp']})
 write_csv('results/FEEDBACK19_INFLUENCE_TABLE.csv',summ)
 (out/'README.md').write_text('''# R31 — leave-one-feedback-observation-out influence diagnosis

Read REPORT.md. Same R30 beta1/source/MLP/16steps. All19feedback observations
removed one at a time, with original platform denominators Drone10/Quad12 and
fixed original U96/93. Removed observation is not moved into keep. Original
instance groups and source representatives are unchanged. No GT-based filter.

23states:2all+19LOO+2zero. All-state bitexact R30b1; zero bitexact source.
400exposedobservations A208/DEV64/R64/S64; ownplatform200perintervention,
3800influence rows. Counterpart platform is structurally unchanged reference,
not an extra set of newly scored predictions. Same-platform S32 mean impact
must be divided by2 for two-platform balanced impact. Aothers excludes itself.

Four all-feedback uniform nominalmass controls: Drone.9/.8, Quad11/12,10/12.
They match number-of-term mass, not gradientnorm/direction/Adam displacement.
Historical active_preferences field in MASS4 receipts is int(sum(weights)),
NOT the count of nonzero terms; all original terms are present.

Correct W_drone_012 removal improves S sameplatform+3.787802pp beyond matched
uniformmass task effect0. Wrong W_quad_010 removal restores a harmed target
but loses more gains and S mean-2.605951pp. Multiple removals and Quad10/12mass
can recover the same S_quad_016; do not add these effects or make a GTblacklist.
12/19remove runs have taskIoU unchanged, not necessarily states/query unchanged.

A_drone034 newgroup is visibly dark M0 with109/109assignedforegroundpoints;
source chiefly white M1 with170/184. Full-caption direction has ambiguity and
mask/camera support is not independent worldID gold. M0 was compared but U;
only M1>M5 accepted. No new negative was assigned. Rawimages/points remain local.

Independent CPU audit reconstructs23contracts/4600main taskvalues/3800effects,
plus4mass contracts/800taskvalues/3800residuals. No newQwen/sourcefulltraining,
no new confirmation or model adoption. All positive/negative/zero effects kept.
The 19-summary is diagnostic, not a deployment ranking of pseudo-label quality.

Requires licensedprivateassets and parent ccff83b. No rawcaptions, RGB, pointclouds,
coordinates, opaque taskIDs, featurecaches, states or thirdparty source snapshots
redistributed. Own literature audit links primary papers/official code; proposed
scope controls are adaptations, not claimed reproductions. Use fresh output
directories and do not overwrite seals. Main workflow:influence31 ->score31;
posthoc registered mass_control31 ->score_mass31; independent audit/identity;
finalize31 ->publish31. failedattempt001 preserved, repaired before firststep.
''')
 dump(out/'DEPENDENCIES.json',{'parent_commit':'ccff83b5383aca03fbb65c879ccb27dd7d06f4d4','student_input':'XYZ+text','external_feedback':'R29 cached Qwen RGB+fulltext preferences unchanged','new_Qwen_calls':0,'thirdparty_assets_or_source_snapshots_redistributed':False,'cross_platform_counters':'opposite platform structurally fixed; no extra synthetic scored samples'})
 forbidden={str(r[k]) for split in ['A208','DEV64','R64','S64'] for r in rows(split) for k in ['id','caption','imagepath','pointpath','metapath','scene']};files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:
  text=p.read_text();assert not any(t and t in text for t in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True,'thirdparty_source_snapshots_excluded':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_feedback_observation_influence_diagnostic_20261102';dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'influence_rows':3800,'LOO_mass_residual_rows':3800,'main_model_task_scalar_rows':4600,'mass_control_task_scalar_rows':800,'original_feedback_observations':19,'private_payload_scan_passed':True,'all_signs_and_zero_effects_retained':True,'prefix':prefix});print('EXPORT31',len(files),3800,4600,800)
if __name__=='__main__':main()
