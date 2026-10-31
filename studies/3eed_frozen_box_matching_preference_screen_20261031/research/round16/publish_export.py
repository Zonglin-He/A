"""Explicit scalar/code export; never copies raw media, IDs, captions or weights."""
from pathlib import Path
import sys,json,csv,shutil,re,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round16'))
from common import EXP,read,dump,sha
STUDY='3eed_frozen_box_matching_preference_screen_20261031'
def main():
 out=EXP.parent/'publication/public_export_attempt001';out.mkdir(parents=True,exist_ok=False)
 code=list((ROOT/'crossplatform/round16').glob('*.py'))+list((ROOT/'research/round16').glob('*.py'))
 for p in code:
  target=out/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 protocol=['AUTHORIZATION.json','PLAN_LOCK.json','FIT_SPEC_LOCK.json','FIT_SPEC_PRE_BINARY_PRESERVED.json','FIT_SPEC_BINARY_PRESERVED.json','FIT_SPEC_RELATIVE_PRESERVED.json','FIT_SPEC_PRE_INDEX_PATH_FIX.json','G_BINARY_PRELABEL_ADDENDUM.json','RELATIVE_PRELABEL_LOCK.json','LOCAL_PREFERENCE_PRELABEL_LOCK.json','SCORER_PATH_FIX_PRELABEL.json','PROJECTION_CPU_CAPABILITY_CHECK.json']
 for name in protocol:dump(out/'protocol'/name,read(EXP/name))
 shutil.copyfile(EXP/'REPORT.md',out/'REPORT.md');dump(out/'results/DECISION.json',read(EXP/'DECISION.json'))
 dump(out/'protocol/SPLIT_DISCLOSURE.json',read(EXP.parent/'input/SPLIT_DISCLOSURE.json'))
 audit=read(EXP/'CPU_REPLAY_AUDIT.json');dump(out/'results/CPU_REPLAY_AUDIT.json',audit)
 failure=read(EXP/'FAILURE_STAGE_DIAGNOSIS.json');dump(out/'results/FAILURE_STAGE_DIAGNOSIS.json',failure)
 summaries={s:read(EXP/(s+'_scoring_attempt001/SUMMARY.json')) for s in ('A','B')};dump(out/'results/FALLBACK_SUMMARIES.json',{'scope':'Branch outputs only. No effective reranking or TTA executed; actual TTA effect is null, not zero.','actual_parameter_updates':0,'summaries':summaries})
 rows=[];scene_alias={}
 for stage in ('A','B'):
  rr=read(EXP/(stage+'_scoring_attempt001/ROWS_PRIVATE.json'))
  for row in rr:
   key=(row['platform'],row['scene_PRIVATE'])
   if key not in scene_alias:scene_alias[key]=f'{row["platform"]}_scene{1+sum(k[0]==row["platform"] for k in scene_alias):02d}'
   for arm,iou in row['ious'].items():rows.append({'case_alias':row['alias'],'split':stage,'platform':row['platform'],'scene_alias':scene_alias[key],'arm':arm,'iou':iou,'Acc25':iou>.25,'Acc50':iou>.5,'query_changed':row['queries'][arm]!=row['queries']['source'],'gain_pp':100*(iou-row['ious']['source']),'branch_active':arm=='source','status':row['details'].get('reason','no_admissible_A_preference_source_fallback')})
 p=out/'results/ALL64_SCALAR_ROWS.csv';p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 g_rows=[];seals=[]
 for name in ('features64_attempt001','G_A32_attempt001','G_A32_attempt002','G_A32_attempt003','G_A32_attempt004','B_predictions_attempt001'):
  folder=ROOT/'crossplatform/round16'/name;s=read(folder/'SEALED.json');seals.append({'artifact':name,'seal_sha256':sha(folder/'SEALED.json'),**{k:s[k] for k in ('state','n','calls','new_model_calls','reasons','seconds','peak_bytes','source_unchanged','boxes_and_scores_bitexact_R15','model_unchanged','boxes_from_frozen_source_pool_only','no_B_G_or_fit') if k in s}})
  if name.startswith('G_'):
   for f in sorted((folder/'cases_PRIVATE').glob('*.json')):
    x=read(f);q=x['preference'];g_rows.append({'variant':name,'case_alias':x['alias'],'platform':x['platform'],'status':q['status'],'reason':q['reason'],'positive_query_count':len(q['positive_queries']),'negative_query_count':len(q['negative_queries'])})
 dump(out/'results/ALL_G_FAILURES.json',g_rows);dump(out/'results/SEALED_ARTIFACT_SCALARS.json',seals)
 closure=read(EXP/'B_CLOSURE.json');dump(out/'results/B_CLOSURE.json',closure)
 dump(out/'DEPENDENCIES.json',{'overlay_on_prior_public_bundle':'https://github.com/Zonglin-He/A/tree/be8610e7a08386d799d560949b7df3620446ec36/studies/3eed_unified_fulltext_relational_G_20261031','official_model':'https://github.com/worldbench/3EED','source_checkpoint_sha256':read(EXP/'DECISION.json')['source_checkpoint_sha256'],'source_checkpoint_and_private_inputs_not_exported':True,'runtime':'Source/metrics: existing PyTorch2.7cu128 environment; frozen Qwen2.5VL3B and G: existing mllm environment; no installation performed','reproduction':'Overlay code onto the existing study/workspace; supply legally obtained 3EED data, public VLM weights, source checkpoint and independently registered input manifests. Historical raw observations remain local.'})
 (out/'README.md').write_text('# Frozen-box matching preference prerequisite screen\n\nSee [REPORT.md](REPORT.md). Actual Waymo-only→Drone/Quad inputs were executed. All64 source forwards and710 frozen VLM calls completed;32 A preference observations and32 held-out-from-G/fit B observations retained. No admissible preference under the locked final rule; no actual TTA update. Conditional no-update/TTA outputs are source fallbacks, not measured adaptation gains.\n\nThe source boxes, backbone, decoder and text projection remained frozen. Planned update scope is only the last3D query-matching projection linear layer. The supervised task-utility gate did not activate fitting. All G failures, anonymous scalar rows, code, protocol revisions, cost and limits are retained. The original global-winner readouts and the absolute-confidence failure were preserved before moving to native local pairs. This is exposed development evidence, not a blind or scene-independent confirmation.\n\nNo raw RGB, point clouds, captions, instance/query IDs, boxes, feature arrays, annotations, model weights or credentials are included. Use DEPENDENCIES.json for the prior immutable public-code bundle and runtime/input requirements.\n')
 # Structured public payloads exclude actual private rows and sensor values.
 forbidden={'id_PRIVATE','scene_PRIVATE','pointpath','metapath','imagepath','caption','raw_PRIVATE','raw_answer_PRIVATE','query_indices','source_pool','tokens','weight','bias'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&forbidden,set(x)&forbidden
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
 for f in out.rglob('*.json'):check(read(f))
 assert len(rows)==160 and len(g_rows)==128
 assert all(r['gain_pp']==0 and not r['query_changed'] for r in rows)
 files=[{'path':str(f.relative_to(out)),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file()]
 dump(out/'MANIFEST.json',{'study':STUDY,'files':files,'no_raw_data_or_weights':True,'actual_TTA_gain_measured':False})
 dump(EXP.parent/'publication/LOCAL_EXPORT_AUDIT.json',{'all_explicit_sources_copied':True,'structured_private_key_check_passed':True,'all_scalar_rows':160,'all_G_failure_rows':128,'files':len(files)+1,'manifest_sha256':sha(out/'MANIFEST.json'),'public_report_sha256':sha(out/'REPORT.md'),'scope':'No source checkpoints/media/captions/features/annotations; all negative scalar rows retained.'})
 print('EXPORT_READY',len(files)+1)
if __name__=='__main__':main()
