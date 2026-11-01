"""Public own-code/protocol/anonymous scalar export, all negatives retained."""
import sys,csv,io,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round25'))
from common25 import *

def main():
 pub=EXP.parent/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt002';out.mkdir(exist_ok=False)
 def copy(p,d):
  dest=out/d;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 for folder in [BASE,EXP.parent]:
  for p in sorted(folder.glob('*.py')):copy(p,p.relative_to(ROOT))
 for p in sorted(EXP.glob('*LOCK.json')):copy(p,Path('protocol')/p.name)
 for name in ['AUTHORIZATION.json','COORDINATE_REPAIR.json','PILOT_AUDIT.json','PILOT_REPAIRED_AUDIT.json']:copy(EXP/name,Path('protocol')/name)
 copy(EXP.parent/'input/EXPOSURE.json','protocol/EXPOSURE.json');copy(EXP.parent/'literature/PAPER_AUDIT.json','literature/PAPER_AUDIT.json')
 camera=read(EXP/'SOURCE_CAMERA_AUDIT.json');camera['records']=[x for x in camera['records'] if x['GT_projection_2D_IoU'] is None or x['GT_projection_2D_IoU']<.5];camera['export_records']='all source calibration low-tail aliases only; full4194record ledger private';dump(out/'results/SOURCE_CAMERA_SUMMARY.json',camera)
 for name in ['CPU_AUDIT.json','DECISION.json','COSTS.json','EFFECTS.json','ALL_CHANGED_CASES.json','ALL_NEGATIVE_CASES.json','CLASS_DIAGNOSTIC.json','FULL_MODEL_CLOSURE.json','REVIEW_MANIFEST.json','SOURCE_CURVES_AUDITED.json','ALL400_SUMMARY.json','OLD336_SUMMARY.json','NEW64_SUMMARY.json','ALTERNATIVE_ALL400_SUMMARY.json','ALTERNATIVE_OLD336_SUMMARY.json','ALTERNATIVE_NEW64_SUMMARY.json']:
  copy(EXP/name,Path('results')/name)
 for folder in ['training_attempt001','training_rank_attempt001']:copy(BASE/folder/'HISTORY.json',Path('results')/folder/'SOURCE_HISTORY.json')
 for name in ['source_labels_attempt001','source_labels_attempt002']:copy(EXP/name/'SUMMARY.json',Path('results')/('INVALID_'+name if name.endswith('001') else name)/'SOURCE_SUMMARY.json')
 allrows=rows();manifest={r['id']:r for r in allrows};scenes={p:sorted({r['scene'] for r in allrows if r['platform']==p}) for p in ['waymo','drone','quad']};total=0
 fields=['alias','platform','scene_alias','stage','arm','iou','same_as_source']
 for family,file in [('primary','T400_ROWS_PRIVATE.json'),('alternative','ALTERNATIVE_T400_ROWS_PRIVATE.json')]:
  rr=read(EXP/file);buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader()
  for r in rr:
   row=manifest[r['id_PRIVATE']]
   for arm,v in r['ious'].items():w.writerow({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_s'+str(scenes[r['platform']].index(row['scene'])),'stage':r['stage'],'arm':arm,'iou':v,'same_as_source':r['q'][arm]==r['q']['source']});total+=1
  p=out/'results'/family/'ALL_TARGET_SCALAR_ROWS.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue())
 # Selected source checkpoint row projections; all56curves retained separately.
 sv=rows('SV512');buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader()
 for family,folder,lock in [('token_mass','training_attempt001','SOURCE_SELECTED_LOCK.json'),('residual_rank','training_rank_attempt001','SOURCE_RANK_SELECTED_LOCK.json')]:
  for c in read(EXP/lock)['selected']:
   q=load_npz(BASE/folder/'validation_PRIVATE'/(c['run']+f'_e{c["epoch"]:02d}.npz'))['query_indices']
   for r,j in zip(sv,q):
    z=feature(r);v=float(load_npz(EXP/'source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'].astype('float32')[j]);w.writerow({'alias':r['alias'],'platform':'waymo','scene_alias':'waymo_s'+str(scenes['waymo'].index(r['scene'])),'stage':'source_VAL','arm':family+'_'+c['run'],'iou':v,'same_as_source':int(j)==int(z['scores'].argmax())});total+=1
 p=out/'results/SOURCE_SELECTED_SCALAR_ROWS.csv';p.write_text(buf.getvalue())
 q=read(EXP/'OLD336_QWEN_ROWS.json');buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['alias','platform','Qwen_G_iou','source_iou','RGB_seed0_iou','RGB_seed1_iou'],lineterminator='\n');w.writeheader()
 for r in q:w.writerow({'alias':r['alias'],'platform':r['platform'],'Qwen_G_iou':r['iou'],'source_iou':r['source_iou'],'RGB_seed0_iou':r['RGB_seed0_iou'],'RGB_seed1_iou':r['RGB_seed1_iou']});total+=1
 (out/'results/OLD336_QWEN_CONTROL.csv').write_text(buf.getvalue())
 assert total==13232,total
 copy(EXP/'REPORT.md','REPORT.md')
 (out/'README.md').write_text('''# R25 — source-supervised RGB/point/text fusion

A real candidate-level trimodal source model: frozen Waymo-only point/text backbone and boxes, frozen ImageNet ResNet34 layer2/3, calibrated local RGB ROI tokens, language-conditioned cross-attention, learned feature gate and matching/rank head. No Qwen in model training or inference. This is source-model construction, NOT a TTA result (zero target updates).

Waymo TRAIN3682, sourceVAL512, two finite readouts × RGB/noRGB × two fixed seeds,30epochs each. All source curves retained. SourceVAL chooses residual-rank family before any target scores;136930new parameters, original matching projection frozen. Point backbone was not retrained.

Target400 expressions (200/platform), including64 new current-input content-disjoint observations,2Drone/3Quadclusters; allVAL historically scored. Main seed mean mIoU Source22.566555%, noRGB22.512782%, RGB23.025109%. RGB vs matched noRGB +0.512327pp (second seed +0.106908). On new64, RGB vs noRGB +1.792572/+0.903738pp, but vs originalSource -0.641229/-0.005278pp. Quad decline is retained; source-trained noRGB ranking also harms that split. RGB benefit is small and platform/seed dependent. Keep research source prototype, not replacement for the stronger Qwen precision baseline and not robust cross-platform superiority.

Both source-readout families were predicted/sealed before current target scoring; no target-driven winner switch. RGB-zero, within-observation query/RGB permutation, global-only, source-supervised noRGB and previousQwen old336 control retained. 24raw-input complete-model closure probes reproduce cached scores and bitexact boxes without inference GT/cache dependency.

An early R25 source-cache coordinate bug omitted official Waymo view rotation, detected BEFORE any fusion training. Invalid001 caches/labels/partialRGB and code are retained and clearly excluded; corrected002 caches used throughout. Earlier R24 target experiments/source checkpoint unaffected. All5source camera views audited; calibration negative tails retained.

13232anonymous scalar rows cover both target families, selected source checkpoints and oldQwen336; they reuse4594unique current inputs, not13232independent samples. All56source curves and421negative comparison records retained. Own code/protocol/scalars only; no raw captions/media/pointclouds/calibration coordinates/task IDs/features/model weights or third-party implementation.

Read REPORT.md and results/CPU_AUDIT.json, DECISION.json, protocol/*LOCK.json. Prior R24 dependency commit86ad9bbbe0b9c6ddb0dabfdd443f52f8b96f8466 and earlier licensed data/source assets are required. Valid workflow: select25; corrected capture25; rgb25; source_labels25; train25 and train_rank25; select_model25; evaluate25 predict; infer25 closure; evaluate_alternative25 predict; score both; audit25; render25; finalize25. Repair/attempt001 scripts are historical engineering evidence, not steps to rerun over valid outputs.
''')
 dump(out/'FIELD_NOTES.json',{'target_TTA_steps':0,'source_supervised_training_not_TTA':True,'fit_receipt_parameter_L2_delta_and_optimizer_steps':'end of complete30epoch trajectory, not necessarily selected checkpoint','selected_optimizer_steps':'epoch*116 in FINAL_SOURCE_SELECTED_LOCK','source_VAL_scalar_precision':'float32 IoU values exactly as source-selection tensors; target metrics use independent float64 geometry','zero_RGB_and_retrained_noRGB':'distinct controls, not interchangeable','metadata_calibration_outliers':'source offline diagnostics, not training exclusions','N64':'current input-content holdout, historical fullVAL exposure','Qwen_control':'available onlyold336, not fake fallback onnew64'})
 dump(out/'DEPENDENCIES.json',{'R24_commit':'86ad9bbbe0b9c6ddb0dabfdd443f52f8b96f8466','source_checkpoint_sha256':read(EXP/'CPU_AUDIT.json')['source_checkpoint_sha256'],'RGB_asset':read(BASE/'assets/RESNET_ASSET_RECEIPT.json'),'third_party_source_code_redistributed':False,'weights_and_raw_data_public':False})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];forbidden={str(r[k]) for r in allrows for k in ['id','caption','pointpath','imagepath','metapath','scene']}
 for p in files:
  text=p.read_text();assert not any(x and x in text for x in forbidden),p
 dump(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];prefix='studies/3eed_source_supervised_trimodal_fusion_20261101'
 dump(pub/'TREE_PAYLOAD_LOCAL.json',[{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files]);dump(pub/'LOCAL_EXPORT_AUDIT.json',{'n':len(files),'scalar_rows':total,'unique_inputs':4594,'unique_target_inputs':400,'source_train':3682,'source_VAL':512,'all_valid_negative_records_and_invalid_attempt_disclosure_kept':True,'privacy_passed':True,'prefix':prefix});print('EXPORT25',len(files),total)
if __name__=='__main__':main()
