"""All-arm seal gate precedes annotation opening; confirmation requires final plan."""
import pathlib,sys,json,hashlib,argparse,copy,datetime
R=pathlib.Path(__file__).resolve().parents[1];A=R.parent
sys.path[:0]=[str(A/'upstream/avism/data'),str(A/'scripts')]
import numpy as np
from pycocotools import mask as mu
from frame_metric import accuracy
from av_loc_area_only import compute_av_loc,combine_av_loc_sequences
import aviseval

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
 if isinstance(v,np.generic):return v.item()
 return v

def hota_summary(res):return {k:float(100*np.mean(res[k])) for k in ['HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','LocA']}
def fa_summary(res):return {k:float(100*np.mean(res[k])) if all(x is not None and np.isfinite(x) for x in res[k]) else None for k in ['FA','FAn','FAs','FAm']}

def score(args):
 run=pathlib.Path(args.run);protocol=json.loads((run/'protocol.json').read_text());planpath=pathlib.Path(protocol['args']['plan']);plan=json.loads(planpath.read_text())
 assert protocol['state']=='sealed' and sha(planpath)==protocol['plan_sha256']
 manifestpath=pathlib.Path(plan['manifest']);assert sha(manifestpath)==protocol['manifest_sha256'];spec=json.loads(manifestpath.read_text())
 if 'confirmation' in manifestpath.name:
  assert plan['state']=='confirmation_final_locked' and (R/'CONFIRMATION_APPROVAL_v2.json').exists()
  masterpath=R/'MASTER_CONFIRMATION_LOCK_v2.json';master=json.loads(masterpath.read_text());approval=json.loads((R/'CONFIRMATION_APPROVAL_v2.json').read_text());assert sha(masterpath)==approval['master_plan_sha256']
  verified={};protos={}
  for study in master['studies']:
   study=pathlib.Path(study);cp=json.loads((study/'protocol.json').read_text());assert cp['state']=='sealed';protos[str(study)]=cp;verified[str(study)]={}
   for arm,h in cp['arms'].items():
    pp=study/arm/'predictions.json';ss=json.loads((study/arm/'seal.json').read_text());assert sha(pp)==h==ss['prediction_sha256'] and ss['state']=='sealed';verified[str(study)][arm]=h
  e=protos[master['studies'][0]];assert len(e['arms'])==6
  for activity_study in master['studies'][1:]:
   a=protos[activity_study];assert len(a['arms'])==7
   assert e['arms']['source']==a['arms']['source'] and e['arms']['latent_identity']==a['arms']['early_latent_v2']
  gate=R/'CONFIRMATION_ALL_SEALS_VERIFIED_v2.json'
  if not gate.exists():gate.write_text(json.dumps({'state':'all20armoutputsverifiedbeforeGTaccess','master_sha256':sha(masterpath),'same_source_early_hashes_shared':True,'verified':verified},indent=2))
 for arm in plan['arms']:
  seal=json.loads((run/arm/'seal.json').read_text());assert seal['state']=='sealed' and seal['plan_sha256']==sha(planpath) and sha(run/arm/'predictions.json')==seal['prediction_sha256']==protocol['arms'][arm]
 out=run/'scoring_v5';out.mkdir(exist_ok=False)
 (out/'GT_ACCESS_RECEIPT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_declared_arms_verified_sealed':len(plan['arms']),'GT_sha256':sha(A/'datasets/test.json'),'scope':manifestpath.name,'code_sha256':sha(__file__),'ground_truth_use':'scoring only, no method output mutation'},indent=2))
 gt=json.loads((A/'datasets/test.json').read_text());by_name={pathlib.PurePosixPath(v['file_names'][0]).parent.name:v for v in gt['videos']};idmap={};spec_byid={}
 for v in spec['videos']:
  ref=by_name[v['video_id']];assert (v['height'],v['width'],v['original_length'])==(ref['height'],ref['width'],ref['length'])
  assert [pathlib.Path(p).name for p in v['image_paths']]==[pathlib.Path(ref['file_names'][i]).name for i in v['frame_indices']]
  idmap[v['video_id']]=ref['id'];spec_byid[ref['id']]=v
 selected=set(idmap.values());gt['videos']=[v for v in gt['videos'] if v['id'] in selected];gt['annotations']=[a for a in gt['annotations'] if a['video_id'] in selected]
 for v in gt['videos']:
  idx=spec_byid[v['id']]['frame_indices'];v['file_names']=[v['file_names'][i] for i in idx];v['length']=len(idx)
 for a in gt['annotations']:
  idx=spec_byid[a['video_id']]['frame_indices']
  for k in ['segmentations','areas','bboxes']:
   if k in a:a[k]=[a[k][i] for i in idx]
 (out/'gt.json').write_text(json.dumps(gt));(out/'id_remap.json').write_text(json.dumps(idmap,indent=2));results=[]
 for arm in plan['arms']:
  native_pred=json.loads((run/arm/'predictions.json').read_text());exact=[]
  for v in spec['videos']:
   truth=[a for a in gt['annotations'] if a['video_id']==idmap[v['video_id']]];pred=[a for a in native_pred if a['video_id']==v['video_id']]
   metric=accuracy(truth,pred,list(range(v['length'])),v['height'],v['width']);exact.append({'video_id':v['video_id'],**metric})
  converted=copy.deepcopy(native_pred)
  for p in converted:p['video_id']=idmap[p['video_id']]
  scoring=out/arm;scoring.mkdir();(scoring/'gt.json').write_text(json.dumps(gt));tracker=scoring/'trackers/candidate';tracker.mkdir(parents=True);(tracker/'results.json').write_text(json.dumps(converted))
  (scoring/'CONVERSION_RECEIPT.json').write_text(json.dumps({'mapping_sha256':sha(out/'id_remap.json'),'sealed_prediction_sha256':sha(run/arm/'predictions.json'),'converted_prediction_sha256':sha(tracker/'results.json'),'policy':'only mechanical filename→official numeric video IDs; no labels/masks/confidence changed'},indent=2))
  cfg=aviseval.datasets.AVIS.get_default_dataset_config();cfg.update({'GT_FOLDER':str(scoring),'GT_File':'gt.json','TRACKERS_FOLDER':str(scoring/'trackers'),'TRACKERS_TO_EVAL':['candidate']});dataset=aviseval.datasets.AVIS(cfg)
  mc=aviseval.metrics.TrackMAP.get_default_metric_config();mc['USE_TIME_RANGES']=False;mc['AREA_RANGES']=[[0,128**2],[128**2,256**2],[256**2,1e10]]
  ec=aviseval.Evaluator.get_default_eval_config();ec['LOG_ON_ERROR']=str(scoring/'native_errors.log');evaluator=aviseval.Evaluator(ec);res,msg=evaluator.evaluate([dataset],[aviseval.metrics.TrackMAP(mc),aviseval.metrics.HOTA()])
  H=aviseval.metrics.HOTA();byseq={};fabs={};per={}
  for sequence in dataset.seq_list:
   raw=dataset.get_raw_seq_data('candidate',sequence);fabs[sequence]=compute_av_loc(copy.deepcopy(raw));filtered=copy.deepcopy(raw);removed=0
   for t,dets in enumerate(raw['tracker_dets']):
    keep=np.array([int(mu.area(m))>0 for m in dets],dtype=bool);removed+=int((~keep).sum())
    for key in ['tracker_ids','tracker_classes','tracker_confidences']:filtered[key][t]=np.asarray(raw[key][t])[keep]
    filtered['tracker_dets'][t]=[m for m,k in zip(dets,keep) if k];filtered['similarity_scores'][t]=raw['similarity_scores'][t][:,keep]
   cs={};native_cs={}
   for cls in dataset.class_list:
    processed=dataset.get_preprocessed_seq_data(filtered,cls)
    assert processed['num_tracker_dets']==sum(len(x) for x in processed['tracker_ids'])
    assert processed['num_tracker_ids']==len(set(int(i) for ids in processed['tracker_ids'] for i in ids))
    cs[cls]=H.eval_sequence(processed);native_cs[cls]=H.eval_sequence(dataset.get_preprocessed_seq_data(raw,cls))
   byseq[sequence]=cs;per[sequence]={'hota_area0':hota_summary(H.combine_classes_class_averaged(cs)),'hota_native':hota_summary(H.combine_classes_class_averaged(native_cs)),'fa_area_only':fa_summary(fabs[sequence]),'empty_prediction_dets_removed':removed}
  combined={cls:H.combine_sequences({seq:byseq[seq][cls] for seq in dataset.seq_list}) for cls in dataset.class_list}
  counts=np.sum([x['correct_counts_per_threshold'] for x in exact],axis=0);nf=sum(x['n_frames'] for x in exact)
  result={'arm':arm,'exact_set':{'frame_weighted_percent':float(counts.mean()/nf*100),'video_macro_percent':float(np.mean([x['mean_threshold_accuracy'] for x in exact])*100),'n_frames':nf,'per_video':exact},'native':res['AVIS']['candidate'],'hota_area0':hota_summary(H.combine_classes_class_averaged(combined)),'fa_area_only_compatibility':fa_summary(combine_av_loc_sequences(fabs)),'per_video_tracking':per,'native_message':msg}
  results.append(result);(scoring/'metrics.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False));(out/'PROGRESS.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False));print('SCORED',arm,float(counts.mean()/nf*100),flush=True)
 final={'state':'completed','scope':manifestpath.name,'plan_sha256':sha(planpath),'GT_sha256':sha(A/'datasets/test.json'),'scorer_sha256':sha(__file__),'primary_exact_set_definition':'area0 absent, exact same category multiplicity and per-threshold feasible class-compatible perfect matching; strictIoU>.05:.05:.95; notnativeFSLA','native_FSLA':'compatibility historical only; knownhardcodedempty/assignment issues','results':results}
 (out/'RESULTS.json').write_text(json.dumps(clean(final),indent=2,allow_nan=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);score(p.parse_args())
