"""Public source/scalar allowlist; preserve zero coverage and all failures."""
from pathlib import Path
import sys,csv,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round19'))
from common import *
STUDY='3eed_top1_relational_evidence_pair_execution_20261031'
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE') and k not in ('image_array_sha256',)}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def main():
 out=EXP.parent/'publication/public_export_attempt001';out.mkdir(parents=True,exist_ok=False)
 for p in list((ROOT/'crossplatform/round19').glob('*.py'))+list((ROOT/'research/round19').glob('*.py')):
  dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 for p in (EXP.parent/'history').glob('*.py'):
  dest=out/'history'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 for p in sorted(EXP.glob('*LOCK*.json'))+[EXP/'AUTHORIZATION.json',EXP/'VISUAL_INTERFACE_REPAIR.json']:
  dump(out/'protocol'/p.name,clean(read(p)))
 dump(out/'protocol/EXPOSURE.json',read(EXP.parent/'input/EXPOSURE.json'))
 shutil.copyfile(EXP/'REPORT.md',out/'REPORT.md')
 files=['DECISION.json','CPU_REPLAY_AUDIT.json','POSTSCORE_DIAGNOSTIC_AUDIT.json','SOURCE_REPEAT_PROBE_ERRATUM.json','CACHE_AUDIT_SUMMARY.json','CACHE_MEAN_VS_MAX_DIAGNOSIS.json','MECHANISM_FUNNEL.json','EVIDENCE_RESPONSE_DIAGNOSTICS.json','RELATION_REJECTION_DIAGNOSIS.json','REJECTED_PAIR_POSTSCORE_UTILITY.json','TEXT_BINDING_PATCH_AUDIT.json','QUALITATIVE_REVIEW.json']
 for n in files:dump(out/'results'/n,clean(read(EXP/n)))
 dump(out/'results/CACHE_AUDIT_CASES.json',clean(read(EXP/'CACHE_AUDIT_PRIVATE.json')))
 for name in ('SUMMARY.json','FEEDBACK_EXECUTION.json','PAIR_DIAGNOSTIC.json'):dump(out/'results'/name,read(EXP/'scored112_attempt001'/name))
 scenes={};rows=[]
 for r in read(EXP/'scored112_attempt001/ROWS_PRIVATE.json'):
  key=(r['platform'],r['scene_PRIVATE'])
  if key not in scenes:scenes[key]=f'{r["platform"]}_scene{1+sum(k[0]==r["platform"] for k in scenes):02d}'
  for arm,v in r['ious'].items():rows.append({'alias':r['alias'],'stage':r['stage'],'platform':r['platform'],'scene_alias':scenes[key],'arm':arm,'iou':v,'Acc25':v>.25,'Acc50':v>.5,'gain_pp':100*(v-r['ious']['source']),'accepted':r['accepted'],'steps':r['steps'],'query_changed':r['query_changed'][arm]})
 assert len(rows)==336
 with (out/'results/ALL112_TASK_ARM_SCALARS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 dump(out/'results/PAIR_FIT_TRACE.json',clean(read(ROOT/'crossplatform/round19/pair_fit112_attempt002/TRACE_PRIVATE.json')))
 dump(out/'results/HISTORICAL_PAIR_TRACE.json',clean(read(ROOT/'crossplatform/round19/historical_pair_interface_attempt001/TRACE_PRIVATE.json')))
 dump(out/'results/HISTORICAL_BUDGET_CONTROLS.json',read(ROOT/'crossplatform/round19/historical_controls_attempt001/RESULTS.json'))
 artifact_names=['sourceH32_attempt001','sourceV16_attempt001','pairs96_attempt001','pairs112_attempt002','G96_attempt001','G112_attempt002','G112_attempt003','G112_attempt004','pair_fit112_attempt002','historical_pair_interface_attempt001','closed112_attempt002','historical_controls_attempt001','historical_closure_attempt001'];artifacts=[]
 for n in artifact_names:
  p=ROOT/'crossplatform/round19'/n;s=verify(p);entry={'artifact':n,'seal_sha256':sha(p/'SEALED.json'),**{k:v for k,v in s.items() if k not in ('files','programs')}}
  if n=='sourceV16_attempt001':entry.pop('source_repeat_bitexact_two_probes',None);entry.update(actual_source_repeat_probes=1,original_two_probe_flag_metadata_incorrect=True,erratum='SOURCE_REPEAT_PROBE_ERRATUM.json')
  artifacts.append(entry)
 dump(out/'results/ARTIFACT_SCALARS.json',artifacts)
 grow=[]
 for r in read(EXP.parent/'input/ALL112_v2_PRIVATE.json'):
  g=read(ROOT/'crossplatform/round19/G112_attempt004/cases_PRIVATE'/(r['id']+'.json'));grow.append({'alias':r['alias'],'stage':r['stage'],'platform':r['platform'],'preparation_reason':g['preparation_reason'],'preference':clean(g['preference']),'pair_count':len(g['pairs']),'probes':clean(g['probes'])})
 dump(out/'results/G_ALL112_DIAGNOSTICS.json',grow)
 dump(out/'DEPENDENCIES.json',{'prior_R18':'https://github.com/Zonglin-He/A/tree/2ad6eb9ee51d211c15a70acf9c4ac9a95e3922dc/studies/3eed_external_tta_grounding_baselines_20261031','prior_R17':'https://github.com/Zonglin-He/A/tree/c3ea771a968713489446322eb0c491ff44fa8f9d/studies/3eed_real_pair_role_audit_matching_tta_20261031','official_3EED':'https://github.com/worldbench/3EED','references':{'EDA':'https://github.com/yanmin-wu/EDA','ZSVG3D':'https://arxiv.org/html/2311.15383v2','DeYO':'https://proceedings.iclr.cc/paper_files/paper/2024/hash/cd0665986c1e9c15f6569ac944bcf88a-Abstract-Conference.html'},'not_these_papers_reproduced':True,'executed_primary':'crossplatform/round19/short_G.py -> fit_pair_v2.py -> closure_v2.py; research/round19/score.py','retained_failed_interfaces':['relational_G.py completed96','relational_G_v2.py completed112','relational_G_v3.py partial stopped'],'actual_score_scope':'112currentobservations,noformalGfeedbackorupdates; historicalpair84stepcapabilityandfixedbudgetcontrolsseparate','postscore_text_patch':'text_binding_patch.py/binding_audit.py only preserve explicit subject appearance field; not integrated into scoredG or accuracy-tested','private_registration_required':True,'reproduction':'Overlay prior public study chain and local licensed3EED/source100/masks/frozenQwen assets. Provide private registrations and hashes; source/media/features/labels/weights not published. Code keeps exact executed local paths/hash checks. No full paper text or third-party weights redistributed.','runtime':'Source/projection/CPU torch2.7cu128 Python3.11; existing frozenVLM runtime at mllm/.venv. No environment installation or source retraining.'})
 (out/'README.md').write_text('# Current-Top1 relational evidence and pair execution\n\n[Report](REPORT.md). Real Waymo-only→Drone/Quad diagnostic completed, boxes frozen, existing18496-parameter matching projection only. All64historical cache cases audited;112currentobservations(A64/H32/V16)processed. Primary relationalG accepted0, so actual newG-TTA steps0 and no formal TTA gain is tested. All336task-arm rows and81rejectedpair audits are retained; no best-case selection.\n\nConcrete old loss failure: group-average log preference improves while the decisive negative sourceTop1 also rises. Historical q0-negative pair diagnostic actually fits after84steps; a real full-grounder forward verifies selectedB and all256boxes unchanged. Same maximum optimization budget with oldgroup objective still fails, but pairfit raises25box-query scores with AABB-IoU<.1 to bothA/B above.5. This one-case execution result is not newG reliability or TTA contribution.\n\nFailed text/visual interfaces and interruptedpartial attempt are preserved. Final localG had no explicit relation contradictingq0 and supportingcompetitor. Post-score rejectedpair IoU direction:22beneficial/43harmful/16ties, correlatedpairs, not identitygold. Occlusion/desaturation controls are diagnostics, not acceptance gates.\n\nImplementation limitation identified: a nounphrase-containment rule dropped18explicit Main appearance features subject fields. A separate text-only binding patch is checked on actualcaptions but NOT integrated or accuracy-tested. The full proposed mechanism is not thereby disproved. Current G/TTA not adopted; no geometryregression or crosssample transfer.\n\nAlltargetVAL historically exposed. A64/H32 diagnostic; V16 newly reserved by inputcontenthash with finalvisual rules fixed before its visual outputs. Not pristineblind or independent-scene confirmation. No rawRGB/pointcloud/caption/instanceID/coordinates/features/weights/annotations/credentials/paperfulltext published. See dependencies and exact protocol hashes.\n')
 forbidden={'id_PRIVATE','scene_PRIVATE','caption','pointpath','metapath','imagepath','raw_PRIVATE','arm_boxes_raw6','query_indices','source_pool','source_scores'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&forbidden,set(x)&forbidden
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 for p in out.rglob('*.json'):check(read(p))
 private=read(EXP.parent/'input/ALL112_v2_PRIVATE.json')
 for p in out.rglob('*'):
  if p.is_file():
   text=p.read_text()
   for r in private:assert r['id'] not in text and r['caption'] not in text,p
 fs=[{'path':p.relative_to(out).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'study':STUDY,'files':fs,'primary_observations':112,'primary_task_arm_rows':336,'rejected_pair_rows':81,'all_negative_rows_retained':True,'no_raw_assets_or_weights':True})
 dump(EXP.parent/'publication/LOCAL_EXPORT_AUDIT.json',{'structured_private_key_and_full_caption_ID_checks_passed':True,'files':len(fs)+1,'task_arm_rows':336,'rejected_pair_rows':81,'report_sha256':sha(out/'REPORT.md'),'manifest_sha256':sha(out/'MANIFEST.json')});print('R19_EXPORT_READY',len(fs)+1)
if __name__=='__main__':main()
