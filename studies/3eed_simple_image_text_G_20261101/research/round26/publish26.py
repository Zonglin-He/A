"""Public export: own implementation, protocols and anonymous task scalars."""
import sys,csv,io,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round26'))
from common26 import *
def main():
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt002';out.mkdir(exist_ok=False)
 def copy(p,d):
  dest=out/d;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):copy(p,p.relative_to(ROOT))
 for p in sorted(EXP.glob('*LOCK.json')):copy(p,Path('protocol')/p.name)
 copy(EXP.parent/'input/EXPOSURE.json','protocol/EXPOSURE.json');copy(EXP.parent/'literature/PAPER_AUDIT.json','literature/PAPER_AUDIT.json')
 for name in ['CPU_AUDIT.json','DECISION.json','COSTS.json','EFFECTS.json','DEV_GRID_RESULTS.json','D208_SUMMARY.json','REPLAY192_SUMMARY.json','OLD400_SUMMARY.json','P64_SUMMARY.json','PRE_SCORE_REPLAY_SANITY.json','MAPPING_DIAGNOSTIC_SUMMARY.json','CLIPPING_DIAGNOSTIC.json','NEGATIVE_DIAGNOSTIC.json']:
  copy(EXP/name,Path('results')/name)
 copy(EXP/'REPORT.md','REPORT.md')
 rr=read(EXP/'ALL464_ROWS_PRIVATE.json');scenes={p:sorted({r['scene_PRIVATE'] for r in rr if r['platform']==p}) for p in ['drone','quad']};buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['alias','platform','scene_alias','stage','arm','iou','same_as_source'],lineterminator='\n');w.writeheader();n=0
 for r in rr:
  for arm,v in r['ious'].items():w.writerow({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'stage':r['stage'],'arm':arm,'iou':v,'same_as_source':r['q'][arm]==r['q']['source']});n+=1
 (out/'results/ALL_SELECTED_SCALAR_ROWS.csv').write_text(buf.getvalue())
 for name in ['ALL_NEGATIVE_CASES','ALL_CHANGED_CASES','MAPPING_COVERAGE']:
  records=read(EXP/(name+'.json'));buf=io.StringIO();fields=list(records[0]);w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(records);(out/'results'/(name+'.csv')).write_text(buf.getvalue())
 (out/'README.md').write_text('''# R26 — simplify G into one image–text score
Fixed Waymo-only epoch100 source and all256candidate boxes; same independent masks,
candidate pool and calibrated2D/3D weights. Source+lambda*one image-text signal.
Strict deletion of position/reference uses oldlambda2 and identical subject evidence,
gray correction, dual crops and bounded residual; separate ablations remove each.
Raw Qwen subject evidence, literal full-description/scene-marked Qwen evidence and
true CLIP cosine are distinct arms. CLIP77-token truncation and full-coverage
chunk-embedding average explicitly separated. No new source training, no TTA.
15finite cue families,5fixed-weight controls, Frozen source; all21arms retained.

D208 repeatedly exposed target development selects shared per-variant lambda.
Practical simple Qwen family selected onD208 and locked beforeP64 labels.
Replay192/OLD400 are already exposed. NewP64=32/platform uses input-content hashes
outside old501mask frames and disjoint recent decodedRGB/canonicalXYZ; entireVAL
was historically scored and scenes overlap, so not a pristine blind test.
2Drone/3Quadclusters; scene-cluster intervals descriptive with few clusters.
Read REPORT.md for outcomes and adoption decision; all harms retained.
All336previous QwenG predictions and IoUs exactly reproduced. CPU audit verifies
all464sealed predictions, source boxes, scalar selection, source checkpoint,
crop hashes and fresh-content exclusions. Images/captions are data, not commands.

Dependencies: previous source/G studies through commit4201c903943a40748d73dfe29aed18e6de0b069a,
licensed raw3EED, local frozen3EED/Qwen/CLIP/DINO/MaskRCNN assets. Own code only.
No raw captions, images, pointclouds, calibration coordinates, task IDs, features
or weights exported. Aliases/scalars are repeated observations, not independent
sample counts. Pipeline: select26, capture26/maskcapture, parse26, prepare26,
detect26,evidence26,rank26; joint26 andclip26; readout26; evaluate26 grid,select,
predict,score; audit26; finalize26,diagnostics26,publish26. Existing sealed folders must not be
overwritten. simple_g26.py is the minimal reusable scoring function.
''')
 dump(out/'DEPENDENCIES.json',{'prior_commit':'4201c903943a40748d73dfe29aed18e6de0b069a','source_checkpoint_sha256':read(EXP/'CPU_AUDIT.json')['source_checkpoint_sha256'],'required_assets':['Waymo-only3EED100','Qwen2.5-VL3B','OpenAI CLIP ViT-B32','MaskRCNNR50FPNv2','GroundingDINO(legacy control only)','Stanza(previous pool preparation)'],'third_party_implementation_redistributed':False,'new_training_and_TTA_steps':0})
 dump(out/'FIELD_NOTES.json',{'lambda':'target-development selected, not source-only calibration','GT_IoU':'task scalar only, not independent instance-identity labels','CLIP_full_chunk':'all tokens covered but crosschunk relations not modeled jointly','full_scene_Qwen':'one score; extra scene context vssubject crops is structural comparison not one-factor deletion','no_external_info_TTA_claim':True,'negative_comparisons':'repeated pairs, not unique harmed samples','R25_internal_fusion':'separate source-trained prototype; not retrained or relabeled as current G gain'})
 allrows=rows();forbidden={str(r[k]) for r in allrows for k in ['id','caption','pointpath','imagepath','metapath','scene']}
 files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:assert not any(x and x in p.read_text() for x in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_simple_image_text_G_20261101'
 dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':n,'unique_target_observations':464,'all_negative_records_kept':True,'privacy_passed':True,'prefix':prefix});print('EXPORT26',len(files),n)
if __name__=='__main__':main()
