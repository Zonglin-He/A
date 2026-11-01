"""Publish own code, anonymous scalar evidence and all negative tails; no raw media."""
import sys,shutil,io,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round27'))
from common27 import *
def main():
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def copy(p,d):
  dest=out/d;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):copy(p,p.relative_to(ROOT))
 for p in sorted(EXP.glob('*LOCK.json')):copy(p,Path('protocol')/p.name)
 for n in ['PROCESSOR_PROBE.json','ENGINEERING_PROBE_AUDIT.json','INPUT_SCHEMA_REPAIR.json']:copy(EXP/n,Path('protocol')/n)
 copy(EXP.parent/'input/EXPOSURE.json','protocol/EXPOSURE.json');copy(EXP.parent/'literature/PAPER_AUDIT.json','literature/PAPER_AUDIT.json')
 for n in ['CPU_AUDIT.json','DECISION.json','KEY_CONTRASTS.json','COSTS.json','COVERAGE_SUMMARY.json','EFFECTS.json','DEV_GRID_RESULTS.json','OLD400_SUMMARY.json','OLD464_SUMMARY.json','REPLAY256_SUMMARY.json','Q64_SUMMARY.json','RESOLUTION_SUMMARY.json','RESOLUTION_EFFECT_SUMMARY.json','POOL_OPPORTUNITY.json','OPPORTUNITY_TASK_SUMMARY.json']:copy(EXP/n,Path('results')/n)
 copy(EXP/'REPORT.md','REPORT.md');rr=read(EXP/'ALL528_ROWS_PRIVATE.json');scenes={p:sorted({x['scene_PRIVATE'] for x in rr if x['platform']==p}) for p in ['drone','quad']};buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['alias','platform','scene_alias','stage','arm','iou','same_as_source'],lineterminator='\n');w.writeheader();n=0
 for r in rr:
  for arm,v in r['ious'].items():w.writerow({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'stage':r['stage'],'arm':arm,'iou':v,'same_as_source':r['q'][arm]==r['q']['source']});n+=1
 (out/'results/ALL_SELECTED_SCALAR_ROWS.csv').write_text(buf.getvalue())
 for name in ['ALL_CHANGED_CASES','ALL_NEGATIVE_CASES','RESOLUTION_EFFECT_ROWS']:
  records=read(EXP/(name+'.json'));buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records);(out/'results'/(name+'.csv')).write_text(buf.getvalue())
 (out/'README.md').write_text('''# R27 — one joint full-text score from target crop and scene
A real Waymo-only -> Drone/Quad frozen-candidate experiment. Four input controls:
same-prompt scene; higher-pixel-cap scene; two identical scenes; local RGB crop +
fullscene. A single frozen Qwen forward jointly reads both images and the complete
literal description, then gives one Yes/No logodds. No hand-added appearance,
position, reference, gray-image or source-relative residual components. Shared
2D/3D mapping and source+lambda*onecue. No parameter TTA or source-model training.

D208 exposed target-development selects one sharedlambda pervariant on a finite
equal grid. Primary crop_scene_joint fixed before outputs. Old400 and replay256
are exposed. NewQ64=32/platform, decodedRGB/XYZ disjoint recent history including
R26P64; allVAL historically scored and2Drone/3Quadscenes overlap. Not a pristine
blind or new-scene test. Fixed2 controls, FrozenSource, legacyG and R26old-prompt
scene retained. All negatives and aliases retained; comparisons not independent.

Read REPORT.md and CPU_AUDIT.json. Native processor grid/resize and local detail
are audited against actual model inputs, with prompt/caption/image hash binding.
38oldR26-regression winner-mask sizes are descriptive, not identity ground truth.
Upsampling cannot recover absent raw details. Higher input token budget and
repeated images are controlled, with actual tokens retained. No raw images,
captions, pointclouds, calibration coordinates, task IDs, features or weights
exported. Own gallery generator only, no generated raw-data panels exported.

Dependencies: previous studies through3386ef7e9fd7d4d4b05558dafdad9ad1a3389e55,
licensed3EED rawdata, fixed Waymo-only100, Qwen2.5-VL3B, MaskRCNN, Stanza and
GroundingDINO(legacyGcontrol only). No model upgrade following GitHub redirects.
Workflow: select27; capture27/maskcapture; parse27/prepare27/detect27/evidence27/
rank27; joint27 --probe thenjoint27; resolution27; readout27; evaluate27 grid,
select,predict,score; audit27; diagnostics27; finalize27; publish27.
Sealed output directories must not be overwritten. Initial CPU resolution run
missed the OLD400 manifest alias; repair changed no inference-input bytes.
''')
 dump(out/'FIELD_NOTES.json',{'score':'one jointly encoded IT logodds per instance,mapped to3Dcandidates','repeat_control':'same perimage cap,actualtokens not always identical to crop+scene','resolution_diagnosis':'nativebbox pixels do not establish visible fineattribute or instance identity','primary_protocol':'crop_scene_joint fixed;all other controls retained without winner switch','lambda':'targetD208labels used for development,not source-only calibration','cached_dependencies':'oldmodels/pools/results fromR26 reused;freshQsource/masks/referenceG actually executed','legacy_tags_in_cloned_helpers':'some older log/seal tag names retained;actualn/source/paths verified','negative_records':'all repeatedarm/ref comparison rows,not independent samples','new_source_training_and_TTA_steps':0})
 dump(out/'DEPENDENCIES.json',{'prior_commit':'3386ef7e9fd7d4d4b05558dafdad9ad1a3389e55','source_checkpoint_sha256':read(EXP/'CPU_AUDIT.json')['source_checkpoint_sha256'],'third_party_code_or_weights_redistributed':False})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];forbidden={str(r[k]) for r in rows() for k in ['id','caption','pointpath','imagepath','metapath','scene']}
 for p in files:assert not any(s and s in p.read_text() for s in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_joint_crop_scene_fulltext_G_20261101'
 dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':n,'unique_target_observations':528,'privacy_passed':True,'prefix':prefix,'all_negative_records_retained':True});print('EXPORT27',len(files),n)
if __name__=='__main__':main()
