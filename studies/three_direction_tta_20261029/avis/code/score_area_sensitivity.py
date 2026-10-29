"""Separate evaluator sensitivity; never a claimed complete FSLA repair.
FSLA: replace only two hardcoded empty-RLE tests by area==0; retain native mapping.
HOTA: remove only area-zero predicted detections from per-frame tracking fields,
subset corresponding similarity columns, then use unchanged native HOTA pipeline.
Native predictions, native scores, GT, and training/inference code stay untouched.
"""
import pathlib,json,argparse,sys,copy,hashlib,numpy as np
from pycocotools import mask as mu
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'upstream/avism/data'))
import aviseval
from av_loc_area_only import compute_av_loc,combine_av_loc_sequences
p=argparse.ArgumentParser();p.add_argument('--study',required=True);args=p.parse_args();study=pathlib.Path(args.study);plan=json.loads((study/'PLAN_LOCK.json').read_text());rows=[]
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
 if isinstance(v,np.generic):return v.item()
 return v
def hota_summary(res):return {k:float(100*np.mean(res[k])) for k in ['HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA']}
def fa_summary(res):
 out={}
 for k in ['FA','FAn','FAs','FAm']:
  values=list(res[k]);defined=[x is not None and np.isfinite(x) for x in values]
  if not any(defined):out[k]=None
  else:
   assert all(defined),('partially undefined alpha values',k,values)
   out[k]=float(100*np.mean(values))
 return out
for arm in plan['arms']:
 name=arm['method']+'_'+str(arm['steps']);run=study/'runs'/name;proto=json.loads((run/'protocol.json').read_text());assert proto['state']=='sealed' and hashlib.sha256((run/'predictions.json').read_bytes()).hexdigest()==proto['prediction_sha256'];scoring=run/'scoring';native=json.loads((scoring/'metrics.json').read_text())['metrics']['AVIS']['candidate']
 config=aviseval.datasets.AVIS.get_default_dataset_config();config.update({'GT_FOLDER':str(scoring),'GT_File':'gt.json','TRACKERS_FOLDER':str(scoring/'trackers'),'TRACKERS_TO_EVAL':['candidate']});dataset=aviseval.datasets.AVIS(config);metric=aviseval.metrics.HOTA();hota_byseq={};fa_byseq={};per={};removed_total=0
 for sequence in dataset.seq_list:
  raw=dataset.get_raw_seq_data('candidate',sequence)
  # Independent FSLA compatibility variant never removes/drops raw arrays.
  fa=compute_av_loc(copy.deepcopy(raw));fa_byseq[sequence]=fa
  filtered=copy.deepcopy(raw);removed=0
  for t,detections in enumerate(raw['tracker_dets']):
   keep=np.array([int(mu.area(mask))>0 for mask in detections],dtype=bool);removed+=int((~keep).sum())
   for key in ['tracker_ids','tracker_classes','tracker_confidences']:
    if key in filtered:filtered[key][t]=np.asarray(raw[key][t])[keep]
   filtered['tracker_dets'][t]=[mask for mask,k in zip(detections,keep) if k]
   filtered['similarity_scores'][t]=raw['similarity_scores'][t][:,keep]
  removed_total+=removed;classes={}
  for cls in dataset.class_list:
   processed=dataset.get_preprocessed_seq_data(filtered,cls)
   assert processed['num_tracker_dets']==sum(len(ids) for ids in processed['tracker_ids'])
   assert processed['num_tracker_ids']==len(set(int(i) for ids in processed['tracker_ids'] for i in ids))
   classes[cls]=metric.eval_sequence(processed)
  hota_byseq[sequence]=classes;per[sequence]={'HOTA_area_zero_pred_dets':hota_summary(metric.combine_classes_class_averaged(classes)),'FSLA_area_tests_only':fa_summary(fa),'empty_predicted_detections_removed_for_HOTA':removed,'GT_frame_support':{'silent':int(fa['frame_num_n_all'][0]),'single':int(fa['frame_num_s_all'][0]),'multi':int(fa['frame_num_m_all'][0])}}
 combined_classes={cls:metric.combine_sequences({seq:hota_byseq[seq][cls] for seq in dataset.seq_list}) for cls in dataset.class_list};hota=hota_summary(metric.combine_classes_class_averaged(combined_classes));fa=fa_summary(combine_av_loc_sequences(fa_byseq));rows.append({'arm':name,'native_HOTA':native['HOTA'],'native_FSLA':native['FA'],'HOTA_area_zero_pred_dets':hota,'FSLA_area_tests_only':fa,'empty_predicted_detections_removed_for_HOTA':removed_total,'per_video':per})
 (study/'EVALUATOR_AREA_SENSITIVITY_PROGRESS.json').write_text(json.dumps(clean({'completed_rows':rows}),indent=2,allow_nan=False))
(study/'EVALUATOR_AREA_SENSITIVITY_PROGRESS.json').write_text(json.dumps(clean({'completed_rows':rows}),indent=2,allow_nan=False))
result={'scope':'post-seal evaluator compatibility sensitivity on all arms; not method scores or complete FSLA repair','native_scores_unchanged':True,'prediction_files_unchanged':True,'HOTA_change':'only area-zero predicted per-frame detections omitted; indices/classes/confidence/similarity columns stay aligned; native HOTA matching/aggregation and all26-class macro remain','FSLA_change':'only two empty-RLE literal tests replaced by area tests; original single-source GTrow→predcol mapping and all other logic intentionally retained','remaining_FSLA_caveat':'native single-source mapping may be wrong for some ID permutations; area-only values cannot be called fully corrected official FSLA','scorer_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'FSLA_variant_sha256':hashlib.sha256((R/'code/av_loc_area_only.py').read_bytes()).hexdigest(),'rows':rows}
out=study/'EVALUATOR_AREA_SENSITIVITY.json';assert not out.exists();out.write_text(json.dumps(clean(result),indent=2,allow_nan=False));print(json.dumps(clean([{k:v for k,v in x.items() if k!='per_video'} for x in rows]),indent=2,allow_nan=False))
