"""Enumerate every raw official-VAL caption; no target class/bbox field is indexed."""
from pathlib import Path
import json,datetime,hashlib
from collections import Counter
from common import R,ROUND,P,sha,clean_caption,noun_caption
DATA=P/'dataset_full/3eed_dataset';SALT='3eed-round5-frozen-official-val-v1-20261030'
def order(s):return hashlib.sha256((SALT+'|'+s).encode()).hexdigest()
def main():
 dest=R/'inputs';dest.mkdir(exist_ok=False);scoring=[];allrows=[];summary={}
 for platform in ['waymo','drone','quad']:
  split=DATA/'splits'/(platform+'_val.txt');scenes=[s.strip() for s in split.read_text().splitlines() if s.strip()];missing=[s for s in scenes if not (DATA/platform/s).is_dir()];rows=[];issues=[];rawobj=0;frames=0
  for scene in scenes:
   if not (DATA/platform/scene).is_dir():continue
   for frame in sorted(p for p in (DATA/platform/scene).iterdir() if p.is_dir()):
    meta=frame/'meta_info.json';lidar=frame/('lidar.npy' if platform=='waymo' else 'lidar.bin');frames+=1
    if not meta.is_file():issues.append({'frame':str(frame),'reason':'missing_metadata'});continue
    container=json.loads(meta.read_text());objects=container['ground_info'];rawobj+=len(objects)
    for j,obj in enumerate(objects):
     key=f'{platform}/{scene}/{frame.name}/{j}';sid=order(key)[:16];caption=obj.get('caption');status='ready'
     if not isinstance(caption,str) or not caption.strip():status='invalid_or_empty_caption'
     if not lidar.is_file():status='missing_point_file'
     if status!='ready':issues.append({'id':sid,'key':key,'reason':status});continue
     clean=clean_caption(caption);noun,fallback=noun_caption(clean);seed=int(order('sampling|'+key)[:16],16)
     row={'id':sid,'key':key,'platform':platform,'split':'val','scene':scene,'frame':frame.name,'object_index':j,'caption':clean,'noun_caption':noun,'noun_fallback_identity':fallback,'lidar_path':str(lidar),'sampling_seed':seed,'waymo_view_suffix':int(frame.name.split('_')[-1]) if platform=='waymo' else None};rows.append(row);scoring.append({'id':sid,'key':key,'platform':platform,'scene':scene,'frame':frame.name,'object_index':j,'meta_path':str(meta),'metadata_container_sha256':sha(meta),'waymo_view_suffix':row['waymo_view_suffix']})
  rows=sorted(rows,key=lambda x:x['id']);assert len(rows)>=16;allrows.extend(rows);(dest/(platform+'_all.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');summary[platform]={'official_split_listed_scenes':len(scenes),'missing_listed_scene_dirs':missing,'available_frames':frames,'raw_ground_info_objects':rawobj,'ready_model_inputs':len(rows),'invalid_or_missing_inputs':issues,'noun_fallback_identity_count':sum(x['noun_fallback_identity'] for x in rows),'smoke16_ids':[s['id'] for s in rows[:16]],'split_sha256':sha(split)}
 assert len({x['id'] for x in allrows})==len(allrows)
 allrows=sorted(allrows,key=lambda x:(['waymo','drone','quad'].index(x['platform']),x['id']));(dest/'all.json').write_text(json.dumps(allrows,ensure_ascii=False,indent=2)+'\n');smoke=[s for p in ['waymo','drone','quad'] for s in allrows if s['platform']==p and s['id'] in summary[p]['smoke16_ids']];(dest/'smoke48.json').write_text(json.dumps(smoke,ensure_ascii=False,indent=2)+'\n');(R/'SCORING_INDEX_PRIVATE.json').write_text(json.dumps(scoring,indent=2)+'\n')
 receipt={'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_salt':SALT,'smoke_rule':'First16 SHA256(salt|platform/scene/frame/object_index) perplatform before content/model/GT scores; no silent replacement.','all_val_rule':'All raw objects from every on-disk officialVAL scene; missing listedscene dirs and unusableinputs explicit.','model_input_fields':'XYZ and fullcaption/firstlexiconnoun only; source manifest does not contain trueclass, numericbbox or metadata path.','annotation_container_opened':True,'only_annotation_field_extracted':'caption; object count/index retained','numeric_GT_or_class_values_indexed':False,'platforms':summary,'n_all':len(allrows),'n_smoke':len(smoke),'input_manifest_sha256':{p.name:sha(p) for p in dest.glob('*.json')},'scoring_index_sha256':sha(R/'SCORING_INDEX_PRIVATE.json'),'code_sha256':{f:sha(R/f) for f in ['common.py','prepare.py']},'status':'prepared_no_GPU_or_scoring'};(R/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'n_all':len(allrows),'n_smoke':len(smoke),'platforms':summary},indent=2))
if __name__=='__main__':main()
