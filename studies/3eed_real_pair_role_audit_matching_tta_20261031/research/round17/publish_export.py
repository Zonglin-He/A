"""Explicit code/scalar export with all negative rows; no sensor payloads."""
from pathlib import Path
import sys,json,csv,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round17'))
from common import EXP,read,dump,sha
STUDY='3eed_real_pair_role_audit_matching_tta_20261031'
def main():
 out=EXP.parent/'publication/public_export_attempt001';out.mkdir(parents=True,exist_ok=False)
 for p in list((ROOT/'crossplatform/round17').glob('*.py'))+list((ROOT/'research/round17').glob('*.py')):
  dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 protocol=['AUTHORIZATION.json','PLAN_LOCK.json','FIT_SPEC_LOCK.json','FIT_SPEC_PRE_ROLE_POLICY_PRESERVED.json','ROLE_GATE_PRELABEL_ADDENDUM.json','CALIBRATION_PROBE_PRELABEL_LOCK.json','INPUT_FILENAME_COMPATIBILITY_REPAIR.json']
 for n in protocol:dump(out/'protocol'/n,read(EXP/n))
 dump(out/'protocol/EXPOSURE.json',read(EXP.parent/'input/EXPOSURE.json'));shutil.copyfile(EXP/'REPORT.md',out/'REPORT.md')
 for n in ['DECISION.json','CPU_REPLAY_AUDIT.json','MECHANISM_DIAGNOSIS.json','ACTIONABLE_PREFERENCE_DIAGNOSIS.json','B_ORACLE_POSTSCORE_DIAGNOSIS.json','B_CLOSURE.json']:
  dump(out/'results'/n,read(EXP/n))
 for stage in ('A','B'):dump(out/'results'/(stage+'_SUMMARY.json'),read(EXP/(stage+'_scoring_attempt001/SUMMARY.json')))
 for n in ('GATE.json','PAIR_GATE_AUDIT.json','SUBSET_SUMMARIES.json'):dump(out/'results'/('A_'+n),read(EXP/'A_scoring_attempt001'/n))
 for platform in ('drone','quad'):dump(out/'results'/(platform+'_FIT.json'),read(ROOT/'crossplatform/round17/B_predictions_attempt001'/(platform+'_FIT.json')))
 rows=[];pref=[];scenes={}
 for stage in ('A','B'):
  for r in read(EXP/(stage+'_scoring_attempt001/ROWS_PRIVATE.json')):
   key=(r['platform'],r['scene_PRIVATE'])
   if key not in scenes:scenes[key]=f'{r["platform"]}_scene{1+sum(x[0]==r["platform"] for x in scenes):02d}'
   for arm,v in r['ious'].items():rows.append({'case_alias':r['alias'],'split':stage,'platform':r['platform'],'scene_alias':scenes[key],'subset':r['subset'],'arm':arm,'iou':v,'Acc25':v>.25,'Acc50':v>.5,'gain_pp':100*(v-r['ious']['source']),'query_changed':r['queries'][arm]!=r['queries']['source']})
   if stage=='A':
    for policy,d in r['details'].items():pref.append({'case_alias':r['alias'],'platform':r['platform'],'subset':r['subset'],'policy':policy,**d})
 def csvwrite(path,rs):
  path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rs for k in r))
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rs)
 csvwrite(out/'results/ALL96_SCALAR_ROWS.csv',rows);csvwrite(out/'results/ALL_A_POLICY_UTILITY.csv',pref)
 artifacts=[]
 for n in ('features96_attempt001','pairs64_attempt001','G64_attempt001','G_variants64_attempt001','G_selected64_attempt001','B_predictions_attempt001','calibration_probe4_attempt001'):
  p=ROOT/'crossplatform/round17'/n;s=read(p/'SEALED.json');artifacts.append({'artifact':n,'seal_sha256':sha(p/'SEALED.json'),**{k:s[k] for k in ('state','n','counts','calls','seconds','peak_bytes','accepted_observation_counts','source_unchanged','A64_boxes_scores_bitexact_R15','new_B_repeat_bitexact','model_unchanged','boxes_from_frozen_source_pool_only','no_B_G_or_fit') if k in s}})
 dump(out/'results/ARTIFACT_SCALARS.json',artifacts)
 dump(out/'DEPENDENCIES.json',{'prior_R15':'https://github.com/Zonglin-He/A/tree/be8610e7a08386d799d560949b7df3620446ec36/studies/3eed_unified_fulltext_relational_G_20261031','prior_R16':'https://github.com/Zonglin-He/A/tree/35a1a142c881c76810d9e9fc2e7b23d0c6446e20/studies/3eed_frozen_box_matching_preference_screen_20261031','official_3EED':'https://github.com/worldbench/3EED','contextual_calibration_source':'https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf','raw_inputs_and_weights_remain_local':True,'runtime':'Existing source/CPU torch2.7cu128 environment and existing frozen-Qwen environment; no installs or downloads.','reproduction':'Overlay this code on both prior bundles, provide the source100 checkpoint and own3EED inputs. Exact private registration/content inventory is not exported. Legacy fitter input filename A32_PRIVATE is an explicit alias of A64_PRIVATE, not a32-observation subset.'})
 (out/'README.md').write_text('# Actual-pair subject/reference verification and frozen-box matching TTA\n\n[Report](REPORT.md). Real Waymo-only→Drone/Quad experiments completed:64 A observations,32 newly reserved B observations,223 actual same-class candidate pairs,737 frozen VLM generations and32 calibration-probe forwards. Global exposed-development policy chose20 preference observations. Each target platform actually completed32projection updates (18496parameters), then32updated full-modelBforwards verified unchanged boxes and valid scores.\n\nNew B mIoU gains are0for both TTA and the same-information no-grounder-update preference-memory control. TTA changed all32scores but noTop1. No method adopted. All352anonymous task-arm rows,192policy-utility rows,223pair-audit records, negative outcomes and protocol revisions are preserved. Role/mapping gate utility and lack of actionable Top1 feedback are diagnosed without post-B tuning.\n\nAll target VAL has historical exposure; global policy/activation uses A task utility for development and is not an unlabeled deployment gate. B is excluded from G/fit and differs in decoded RGB/XYZ; no same-instance/multiview/time-independence claim. Only1Drone/2Quad scenes. Calibration4 is an engineering probe, not a validated role selector.\n\nThe executed scorer is research/round17/score_policies.py; research/round17/score.py is an earlier unexecuted prototype retained for history. The current fit is crossplatform/round17/fit_projection.py and the full-model check is closure.py.\n\nNo raw RGB, point clouds, captions, coordinates, instance/query IDs, annotations, feature arrays or weights are exported. See DEPENDENCIES.json for prior code and required local assets.\n')
 forbidden={'id_PRIVATE','scene_PRIVATE','pointpath','metapath','imagepath','caption','raw_PRIVATE','raw_answer_PRIVATE','query_indices','source_pool','tokens','weight','bias','mask_indices_PRIVATE'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&forbidden,set(x)&forbidden
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 for p in out.rglob('*.json'):check(read(p))
 assert len(rows)==352 and len(pref)==192
 assert len(read(out/'results/A_PAIR_GATE_AUDIT.json')['rows'])==223
 fs=[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'study':STUDY,'files':fs,'actual_TTA_executed':True,'no_raw_data_or_weights':True})
 dump(EXP.parent/'publication/LOCAL_EXPORT_AUDIT.json',{'all_explicit_sources_copied':True,'structured_private_key_check_passed':True,'task_arm_rows':352,'A_policy_rows':192,'pair_audit_rows':223,'files':len(fs)+1,'manifest_sha256':sha(out/'MANIFEST.json'),'report_sha256':sha(out/'REPORT.md'),'all_negative_task_and_pair_rows_retained':True})
 print('R17_EXPORT_READY',len(fs)+1)
if __name__=='__main__':main()
