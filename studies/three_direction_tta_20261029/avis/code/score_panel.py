"""Independent CPU scorer. Must be invoked after all predictions are sealed."""
import pathlib,sys,argparse,json,hashlib,copy,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'upstream/avism/data'))
import aviseval
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--run',required=True);p.add_argument('--gt',required=True);args=p.parse_args()
run=pathlib.Path(args.run);protocol=json.loads((run/'protocol.json').read_text());pred=run/'predictions.json'
assert protocol['state']=='sealed' and hashlib.sha256(pred.read_bytes()).hexdigest()==protocol['prediction_sha256']
plan=json.loads(pathlib.Path(protocol['args']['plan']).read_text());expected={(x['method'],x['steps']) for x in plan['arms']};seen=set()
for candidate in run.parent.glob('*/protocol.json'):
 item=json.loads(candidate.read_text())
 if item.get('plan_sha256')!=protocol['plan_sha256']:continue
 assert item['state']=='sealed' and hashlib.sha256((candidate.parent/'predictions.json').read_bytes()).hexdigest()==item['prediction_sha256']
 seen.add((item['args']['method'],item['args']['steps']))
assert expected<=seen, f'All arms must be sealed before opening GT; missing {expected-seen}'
manifest=json.loads(pathlib.Path(args.manifest).read_text());assert hashlib.sha256(pathlib.Path(args.manifest).read_bytes()).hexdigest()==protocol['manifest_sha256']
gt=json.loads(pathlib.Path(args.gt).read_text());predictions=json.loads(pred.read_text())
id_remap={}
if manifest.get('id_kind')=='source_filename':
 by_name={v['file_names'][0].split('/')[0]:v for v in gt['videos']};assert len(by_name)==len(gt['videos'])
 for v in manifest['videos']:
  official=by_name[v['video_id']];assert (v['height'],v['width'])==(official['height'],official['width'])
  assert [pathlib.Path(x).name for x in v['image_paths']]==[pathlib.PurePosixPath(official['file_names'][i]).name for i in v['frame_indices']]
  id_remap[v['video_id']]=official['id']
 for item in predictions:item['video_id']=id_remap[item['video_id']]
 selected={id_remap[v['video_id']]:v for v in manifest['videos']}
else:selected={v['video_id']:v for v in manifest['videos']}
gt['videos']=[v for v in gt['videos'] if v['id'] in selected]
for v in gt['videos']:
 idx=selected[v['id']]['frame_indices'];v['file_names']=[v['file_names'][i] for i in idx];v['length']=len(idx)
anns=[]
for a in gt['annotations']:
 if a['video_id'] not in selected:continue
 a=copy.deepcopy(a);idx=selected[a['video_id']]['frame_indices']
 for k in ['segmentations','bboxes','areas']:
  if k in a:a[k]=[a[k][i] for i in idx]
 if not any(a['segmentations']):continue
 anns.append(a)
gt['annotations']=anns;scoring=run/'scoring';scoring.mkdir(exist_ok=False);(scoring/'gt.json').write_text(json.dumps(gt));tracker=scoring/'trackers'/'candidate';tracker.mkdir(parents=True);(tracker/'results.json').write_text(json.dumps(predictions));(scoring/'id_remap.json').write_text(json.dumps({'mapping':id_remap,'sealed_prediction_sha256':protocol['prediction_sha256'],'native_prediction_sha256':hashlib.sha256((tracker/'results.json').read_bytes()).hexdigest(),'policy':'only video IDs remapped using exact source filename and frame/dimension correspondence, no predictions changed'},indent=2))
cfg=aviseval.datasets.AVIS.get_default_dataset_config();cfg.update({'GT_FOLDER':str(scoring),'GT_File':'gt.json','TRACKERS_FOLDER':str(scoring/'trackers'),'TRACKERS_TO_EVAL':['candidate']})
ds=aviseval.datasets.AVIS(cfg);mc=aviseval.metrics.TrackMAP.get_default_metric_config();mc['USE_TIME_RANGES']=False;mc['AREA_RANGES']=[[0,128**2],[128**2,256**2],[256**2,1e10]]
eval_cfg=aviseval.Evaluator.get_default_eval_config();eval_cfg['LOG_ON_ERROR']=str(scoring/'native_errors.log')
evaluator=aviseval.Evaluator(eval_cfg);metrics=[aviseval.metrics.TrackMAP(mc),aviseval.metrics.HOTA()]
result,message=evaluator.evaluate([ds],metrics)
per_video={};full_sequence_list=ds.seq_list
for sequence in full_sequence_list:
 ds.seq_list=[sequence];single,_=evaluator.evaluate([ds],metrics);per_video[sequence]=single['AVIS']['candidate']
ds.seq_list=full_sequence_list
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean(v) for v in value]
 if isinstance(value,np.ndarray):return clean(value.tolist())
 if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
 if isinstance(value,np.generic):return value.item()
 return value
native=clean(result)
(scoring/'metrics.json').write_text(json.dumps({'run_state':'scored','prediction_sha256':protocol['prediction_sha256'],'scope':f"fixed {len(selected)}-video shortened development panel, not full benchmark",'undefined_metric_policy':'nonfinite conditional-stratum scores become null; native stratum denominator counts are retained','metrics':native,'per_video':clean(per_video),'message':message},indent=2,allow_nan=False));print(json.dumps(native,indent=2,allow_nan=False))
