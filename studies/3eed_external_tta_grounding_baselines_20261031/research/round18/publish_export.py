"""Explicit public source/scalar allowlist, all negative cases, no raw assets."""
from pathlib import Path
import sys,csv,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round18'))
from common import EXP,read,dump,sha
STUDY='3eed_external_tta_grounding_baselines_20261031'
def main():
 out=EXP.parent/'publication/public_export_attempt001';out.mkdir(parents=True,exist_ok=False)
 for p in list((ROOT/'crossplatform/round18').glob('*.py'))+list((ROOT/'research/round18').glob('*.py')):
  dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 for n in ('common_engineering_compatibility.py','supplement_controls_failed_attempt001.py','supplement_controls_failed_attempt002.py'):
  p=EXP.parent/'history'/n;dest=out/'history'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 protocol=['AUTHORIZATION.json','COST_PROBE_LOCK.json','ENGINEERING_PASS_AND_FORMAL_ENTRY.json','ONLINE_SUPPLEMENT_LOCK.json','SCORE_PROGRAM_LOCK.json','SUPPLEMENT_LOCK.json','SUPPLEMENT_SCORE_LOCK.json','C_CONTROL_CAPTURE_LOCK.json','C_CONTROL_CAPTURE_LOCK_v2.json','C_CONTROL_CAPTURE_LOCK_v3.json','C_CONTROL_SCORE_LOCK.json']
 for n in protocol:dump(out/'protocol'/n,read(EXP/n))
 for n in ('PLAN_LOCK.json','PLAN_ENGINEERING_PRESERVED_PRIVATE.json'):
  d=read(EXP/n);ids=d.pop('A_ids_PRIVATE');d['A_id_count']=len(ids);d['original_runtime_plan_sha256']=sha(EXP/n);d['export_notice']='Public redacted template. Actual runtime private A IDs withheld; hash identifies original. Not an executable exact private registration.';dump(out/'protocol'/n.replace('_PRIVATE','').replace('.json','_PUBLIC_TEMPLATE.json'),d)
 shutil.copyfile(EXP/'REPORT.md',out/'REPORT.md')
 for n in ('DECISION.json','METHOD_PORT_AUDIT.json','CPU_REPLAY_AUDIT.json','SUPPLEMENT_CPU_AUDIT.json','COVERAGE_COST_AND_CONTROLS.json','C32_COMPARISON.json','CONTROL_REPLAY_FAILURE_HISTORY.json'):dump(out/'results'/n,read(EXP/n))
 for src,dest in [('scored96_attempt001/SUMMARY.json','PRIMARY_A64_B32_SUMMARY.json'),('scored96_attempt001/ONLINE_SUMMARY.json','ONLINE_B32_SUMMARY.json'),('scoredC32_attempt001/SUMMARY.json','SUPPLEMENT_C32_SUMMARY.json'),('scoredCcontrols32_attempt001/SUMMARY.json','SUPPLEMENT_C32_CONTROLS_SUMMARY.json')]:dump(out/'results'/dest,read(EXP/src))
 rows=[];scenes={}
 sources=[('scored96_attempt001/ROWS_PRIVATE.json',False),('scored96_attempt001/ONLINE_ROWS_PRIVATE.json',True),('scoredC32_attempt001/ROWS_PRIVATE.json',False),('scoredCcontrols32_attempt001/ROWS_PRIVATE.json',True)]
 for filename,skip_duplicate_source in sources:
  for r in read(EXP/filename):
   key=(r['platform'],r['scene_PRIVATE'])
   if key not in scenes:scenes[key]=f'{r["platform"]}_scene{1+sum(x[0]==r["platform"] for x in scenes):02d}'
   for arm,v in r['ious'].items():
    if skip_duplicate_source and arm=='source':continue
    rows.append({'case_alias':r['alias'],'split':r['split'],'platform':r['platform'],'scene_alias':scenes[key],'arm':arm,'iou':v,'Acc25':v>.25,'Acc50':v>.5,'gain_pp':100*(v-r['ious']['source']),'box_changed':r['box_changed'][arm]})
 p=out/'results/ALL_TASK_ARM_SCALARS.csv'
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 assert len(rows)==1472
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
  if isinstance(x,list):return [clean(v) for v in x]
  return x
 for artifact,name in [('formal96_attempt001','PRIMARY_TRACE.json'),('reg_online32_attempt001','ONLINE_TRACE.json'),('supplement32_attempt001','C32_TRACE.json')]:dump(out/'results'/name,clean(read(ROOT/'crossplatform/round18'/artifact/'TRACE_PRIVATE.json')))
 dump(out/'results/C_CONTROL_REPLAY.json',read(ROOT/'crossplatform/round18/supplement_controls32_attempt003/REPLAY.json'))
 dump(out/'results/COST_PROBE4.json',read(ROOT/'crossplatform/round18/cost_probe4_attempt001/RESULTS.json'))
 artifacts=[]
 for n in ('engineering2_attempt001','formal96_attempt001','reg_online32_attempt001','cost_probe4_attempt001','supplement32_attempt001','supplement_controls32_attempt001','supplement_controls32_attempt002','supplement_controls32_attempt003'):
  p=ROOT/'crossplatform/round18'/n;s=read(p/'SEALED.json');artifacts.append({'artifact':n,'seal_sha256':sha(p/'SEALED.json'),**{k:v for k,v in s.items() if k not in ('files','programs')}})
 dump(out/'results/ARTIFACT_SCALARS.json',artifacts)
 dump(out/'DEPENDENCIES.json',{'previous_R17':'https://github.com/Zonglin-He/A/tree/c3ea771a968713489446322eb0c491ff44fa8f9d/studies/3eed_real_pair_role_audit_matching_tta_20261031','source100_provenance':'https://github.com/Zonglin-He/A/tree/b5ab49aa8ae14e726a0e0f71f8f2ea676d52651c/studies/3eed_waymo_source100_20261030','official_3EED':'https://github.com/worldbench/3EED','official_repositories':read(EXP/'METHOD_PORT_AUDIT.json')['official_commits'],'source_checkpoint_sha256':read(EXP/'METHOD_PORT_AUDIT.json')['source_checkpoint_sha256'],'runtime':'Existing torch2.7cu128 Python3.11 environment, existing3EED CUDA ops. No installs or upstream changes.','private_assets_withheld':True,'reproduction':'Overlay prior public study chain. Clone Tent, SAR and MEMO repositories into crossplatform/round18/official/{tent,SAR,memo} at the listed immutable commits. SAR sam.py is loaded directly from that clone, not redistributed. Provide licensed3EED inputs, local source100 checkpoint and original3EED/RoBERTa code/assets. Generate own private registrations and hashes; public templates intentionally omit A IDs and cannot validate the original private registration. Exact same private cohort reproduction requires the local registration.','copyright':'No paper PDFs, paper extracted fulltexts, official repository files or third-party weights redistributed. Our task-port source and scalar results only.','executed_scorers':['research/round18/score.py','research/round18/score_supplement.py','research/round18/score_controls.py'],'primary_entry':'crossplatform/round18/runner.py; --engineering before formal','separate_entrypoints':['crossplatform/round18/reg_online.py','crossplatform/round18/cost_probe.py','crossplatform/round18/supplement.py','crossplatform/round18/supplement_controls.py']})
 (out/'README.md').write_text('# Real 3EED external grounding TTA baseline ports\n\n[Full report](REPORT.md). Waymo-only100epoch source→Drone/Quad. TENT, SAR and MEMO grounding ports completed real96input episodic inference/update; Reg-TTA3D-inspired preserved NPG/CRR/CBU but uses stated detector-interface substitutes. It is not an original Reg reproduction. Source checkpoint immutable, current predictions sealed before CPU task scoring, all negative cases retained.\n\nPrimary A64/B32 are current-input adaptation then clean prediction, separately reported; not prior A→B transfer. Separate32input Reg online diagnostic and32new-input official optimizer/LN-scope diagnostic completed. No continual-main protocol chosen, CoTTA/MOS not executed. Official MEMO RVT AdamW recipe is a post-primary fixed diagnostic, not a cherry-picked replacement. Quad sample-mean gains do not survive equal-scene weighting and do not beat same-information augmentation-only. No new method adopted.\n\n1472 anonymous task-arm rows preserve1184primary+96online+128Cdiagnostic+64omitted-control results; duplicate online-source and C-source-control rows omitted from CSV only. Per-case gradients, parameter changes, skipped updates, matching coverage, costs and anonymous scene grouping included. All128current observations retained. All targetVAL has historical exposure, only1Drone/2Quad scenes; not fullVAL or blind independent-scene evidence.\n\nC controls were omitted from initial storage and captured after C scores were exposed, using fixed identical operations. Two failed bitexact replay attempts retained; source32bitexact, adapted outputs repeated within9.536743164e-7m (declared1e-4m bound). Original C predictions unchanged. Controls are descriptive, not independent confirmation.\n\nOwn cached RGB/VLM G is also tested onA64: Drone episodic projection updates harm mIoU, Quad unchanged, same-information direct G improves Quad. These negatives are included; external point/text methods and own extra-RGB systems have different information resources.\n\nNo sensor data, captions, candidate coordinates, instance IDs, features, target annotations, weights, credentials or full paper text are published. Public plan files are explicitly redacted templates with original hashes. Dependencies and execution differences are in DEPENDENCIES.json and results/METHOD_PORT_AUDIT.json.\n')
 forbidden={'id_PRIVATE','scene_PRIVATE','A_ids_PRIVATE','caption','pointpath','metapath','imagepath','arm_boxes_raw6','source_pool','source_scores','query_indices'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&forbidden,set(x)&forbidden
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 for p in out.rglob('*.json'):check(read(p))
 # Exact raw IDs/captions must not accidentally enter any exported textual file.
 private=read(EXP.parent/'input/ALL96_PRIVATE.json')+read(EXP.parent/'input/C32_PRIVATE.json')
 for p in out.rglob('*'):
  if p.is_file():
   text=p.read_text()
   for r in private:assert r['id'] not in text and r['caption'] not in text,p
 fs=[{'path':p.relative_to(out).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'study':STUDY,'files':fs,'all_negative_rows_retained':True,'task_arm_rows':len(rows),'no_raw_assets_or_weights':True})
 dump(EXP.parent/'publication/LOCAL_EXPORT_AUDIT.json',{'structured_private_key_and_raw_caption_ID_checks_passed':True,'files':len(fs)+1,'task_arm_rows':len(rows),'manifest_sha256':sha(out/'MANIFEST.json'),'report_sha256':sha(out/'REPORT.md'),'old_studies_immutable':True})
 print('R18_EXPORT_READY',len(fs)+1,len(rows))
if __name__=='__main__':main()
