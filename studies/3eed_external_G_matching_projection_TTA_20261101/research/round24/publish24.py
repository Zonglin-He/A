"""Public own-code/config/anonymous-scalar export; no models/media/private text."""
import sys,json,csv,io,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'research/round24')]
from common24 import *
from evaluate24 import score

def main():
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def copy(p,d):
  target=out/d;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):copy(p,p.relative_to(ROOT))
 for p in sorted(EXP.glob('*LOCK.json')):copy(p,Path('protocol')/p.name)
 for name in ['AUTHORIZATION.json']:copy(EXP/name,Path('protocol')/name)
 copy(EXP.parent/'input/EXPOSURE.json','protocol/EXPOSURE.json');copy(EXP.parent/'literature/PAPER_AUDIT.json','literature/PAPER_AUDIT.json')
 for name in ['CPU_AUDIT.json','DECISION.json','COSTS.json','MATCHED_MODALITY_COST.json','ALL_CHANGED_CASES.json','ALL_NEGATIVE_CASES.json','EFFECTS.json','MECHANISM_AUDIT.json','REVIEW_MANIFEST.json','M64_SUMMARY.json','V64_LAYER_SUMMARY.json','D208_LAYER_SUMMARY.json','D208_MODAL_SUMMARY.json','V64_MODAL_SUMMARY.json']:copy(EXP/name,Path('results')/name)
 copy(BASE/'screen_attempt001/FIT_RECEIPTS.json','results/FIT_RECEIPTS.json');copy(BASE/'closureM64_attempt001/CLOSURE_AUDIT.json','results/CLOSURE_AUDIT.json');copy(BASE/'memory_attempt001/CACHE_RECEIPT.json','results/CACHE_RECEIPT.json')
 scored=[('D208_layers',read(EXP/'D208_LAYER_ROWS_PRIVATE.json')),('V64_layers',read(EXP/'V64_LAYER_ROWS_PRIVATE.json')),('D208_modalities',read(EXP/'D208_MODAL_ROWS_PRIVATE.json')),('V64_modalities',read(EXP/'V64_MODAL_ROWS_PRIVATE.json')),('M64_fixed',read(EXP/'M64_ROWS_PRIVATE.json'))]
 for split in ['V64','M64']:scored.append((split+'_memory_grid',score(BASE/('memory_attempt001/'+split+'_predictions'),rows(split))))
 allrows=rows();manifest={r['id']:r for r in allrows};scenes={p:sorted({r['scene'] for r in allrows if r['platform']==p}) for p in ['drone','quad']};total=0
 for name,rr in scored:
  buf=io.StringIO();writer=csv.DictWriter(buf,lineterminator='\n',fieldnames=['alias','platform','scene_alias','arm','iou','same_as_source']);writer.writeheader()
  for r in rr:
   source=int(feature(manifest[r['id_PRIVATE']])['scores'].argmax())
   for arm,v in r['ious'].items():writer.writerow({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'iou':v,'same_as_source':r['q'][arm]==source});total+=1
  p=out/'results'/name/'ALL_SCALAR_ROWS.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue())
 assert total==7296,total
 copy(EXP/'REPORT.md','REPORT.md');(out/'README.md').write_text('''# R24 — frozen external G and ranking-projection adaptation

Waymo-only source100 → Drone/Quad. Keep Qwen; matched same-candidate/crop/text no-Qwen and CLIP controls; four real parameter interfaces; same-A no-gradient cache; complete-model replay on64 new observations.

New M64 (32/platform,2Drone/3Quadscenes): mean mIoU source22.639159%, no-Qwen23.317627%, CLIP24.556213%, Qwen direct26.330773%, full-query-MLP transfer23.105303%, adapted+same-current-G26.809724%. Extra gain vs direct G +0.478950pp,2 improvements/1 harm/61 ties. Four other query changes remain IoU0. Few scenes, cluster interval touches0, no certified instance-identity correction. Retain a research candidate, not robust validated production TTA.

Update `contrastive_align_projection_image` all three Linear layers (288-288-288-64,184960params), AdamW5e-4,16steps, frozen boxes/backbones/decoder. Loss is G-distribution KL +source KL anchor +actual competing Top1 margin +parameter anchor. D208 unlabeled G fit; exposed V64 labels select layer/LR/steps, exposed D208 labels select CLIP scale40. M64 content-disjoint from recent manifests, selected before R24 outputs; all target VAL historically scored. No M64 G/labels used to fit. Main protocol is block target adaptation then new-observation inference, not a complete continual-online reproduction. Current-G hybrid was a secondary readout added before M64 scoring.

Source actual input is XYZ+text; use_color config is not actual RGB input. Qwen is frozen external teacher-like feedback, no EMA copy; not teacher-free. No-Qwen still retains independent Mask R-CNN, GroundingDINO and point geometry; CLIP is a VLM replacement. RLCF/TTD/TDA are mechanism references, not reproduced baselines.

All7296anonymous scalar rows include development alternatives, negative tails, memory grid and fixed confirmation, reusing336unique observations rather than7296independent samples. Matching-cost replay: same722images, Qwen44.07s vsCLIP4.93s excluding model startup and common detector.256complete-model head-installation forwards reproduced cached matching scores and bitexact frozen boxes; all nonmatching parameters unchanged and source restored. No new downloads, environments or source training.

Read REPORT.md, protocol/*LOCK.json, results/CPU_AUDIT.json and results/DECISION.json. Execute own scripts in crossplatform/round24 and research/round24 with licensed data and dependencies from prior studies, especially R23 at7ecdfe7497e0242161ba59e293834d349b865b47. No raw media, captions, coordinates, features, task IDs, weights or third-party model code are redistributed. Report gives execution order and limits.
''')
 dump(out/'DEPENDENCIES.json',{'previous_R23_commit':'7ecdfe7497e0242161ba59e293834d349b865b47','prior_study':'studies/3eed_soft_evidence_expanded_instance_G_20261101','source_checkpoint_sha256':read(EXP/'CPU_AUDIT.json')['source_sha256'],'point_input':'XYZ only','frozen_external_models':['Qwen2.5VL3B','OpenAI CLIP ViTB32','GroundingDINO tiny','MaskRCNN R50 FPNv2 COCO'],'original_paper_replication_claimed':False})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];forbidden={str(r[k]) for r in allrows for k in ['id','caption','pointpath','imagepath','metapath','scene']}
 for p in files:
  text=p.read_text();assert not any(x and x in text for x in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True});files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_external_G_matching_projection_TTA_20261101';entries=[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];dump(pub/'TREE_PAYLOAD_LOCAL.json',entries);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':total,'unique_inputs':336,'all_dev_trials_and_negatives_kept':True,'privacy_passed':True,'prefix':prefix});print('EXPORT24',len(files),total)
if __name__=='__main__':main()
