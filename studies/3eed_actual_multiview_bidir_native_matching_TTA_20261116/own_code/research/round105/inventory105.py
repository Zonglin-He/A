"""Only sensor identities/counters and registered non-protected frames; no task GT."""
from pathlib import Path
import json,hashlib,collections,math,ijson
R=Path(__file__).resolve().parents[2];H=R/'research/round105';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def number(x):
 try:y=float(x);return y if math.isfinite(y) else None
 except (ValueError,TypeError):return None
def main():
 protected=set()
 with (R/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json').open('rb') as f:
  for pre,event,value in ijson.parse(f):
   if pre=='item.metapath' and event=='string':protected.add(str(Path(value).resolve()))
 old=rd(R/'research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json')['rows_PRIVATE'];index={str(Path(x['meta_path_PRIVATE']).resolve()):x for x in old};missing=protected-set(index);assert not missing,'protected sensor-hash registry incomplete; do not load new media';denied_images={index[x]['image_sha256'] for x in protected};frames=[];lookup={};summary=[]
 for pl in ['waymo','drone','quad']:
  rows=rd(R/f'research/round97/input/TARGET_{pl.upper()}_PRIVATE.json');local={}
  for r in rows:
   mp=str(Path(r['metapath']).resolve());assert mp not in protected;assert Path(mp).is_relative_to(R/'crossplatform/dataset_full/3eed_dataset');key=(mp,str(Path(r['imagepath']).resolve()),str(Path(r['pointpath']).resolve()));suffix=int(r['suffix'] or 0)
   if key in local:continue
   prior=index.get(mp);stamp=None
   if prior:stamp=number(prior['input_timestamp'])
   else:
    with Path(mp).open('rb') as f:
     for prefix,event,value in ijson.parse(f):
      if prefix=='timestamp' and event in ['number','string']:stamp=number(value)
   folder=Path(mp).parent.name
   try:ordinal=int(folder.split('_')[0])
   except ValueError:ordinal=None
   alias=hashlib.sha256('|'.join(key).encode()).hexdigest()[:24];blocked=bool(prior and prior['image_sha256'] in denied_images);x={'frame_PRIVATE':alias,'platform':pl,'scene_PRIVATE':r['scene'],'metapath':mp,'imagepath':key[1],'pointpath':key[2],'suffix':suffix,'timestamp_counter':stamp,'folder_ordinal':ordinal,'image_sha256':prior['image_sha256'] if prior else None,'point_sha256':prior['point_sha256'] if prior else None,'protected_image_content_alias':blocked,'eligible_for_extra_media':not blocked,'hashes_from_sensor_only_inventory':prior is not None};local[key]=x;lookup[mp]=x
  zz=list(local.values());frames.extend(zz);summary.append({'platform':pl,'expressions':len(rows),'physical_path_frames':len(zz),'eligible_extra_media_frames':sum(x['eligible_for_extra_media'] for x in zz),'protected_image_alias_frames_not_decoded':sum(x['protected_image_content_alias'] for x in zz),'with_numeric_timestamp':sum(x['timestamp_counter'] is not None for x in zz),'scenes':len({x['scene_PRIVATE'] for x in zz})})
 # Engineering anchors: deterministic scene round-robin, never output/label selected.
 anchors=[];pairs=[]
 for pl in ['waymo','drone','quad']:
  dev=rd(R/f'research/round99/input/TARGET_{pl.upper()}128_PRIVATE.json');pool={}
  for r in dev:
   f=lookup[str(Path(r['metapath']).resolve())]
   if f['eligible_for_extra_media']:pool[f['frame_PRIVATE']]=f
  groups={}
  for f in pool.values():groups.setdefault(f['scene_PRIVATE'],[]).append(f)
  for z in groups.values():z.sort(key=lambda x:x['frame_PRIVATE'])
  scenes=sorted(groups,key=lambda s:hashlib.sha256(s.encode()).hexdigest());selected=[];k=0
  while len(selected)<8 and any(k<len(groups[s]) for s in scenes):
   for scene in scenes:
    if k<len(groups[scene]) and len(selected)<8:selected.append(groups[scene][k])
   k+=1
  for A in selected:
   others=[f for f in frames if f['platform']==pl and f['scene_PRIVATE']==A['scene_PRIVATE'] and f['suffix']==A['suffix'] and f['eligible_for_extra_media'] and f['frame_PRIVATE']!=A['frame_PRIVATE'] and (A['image_sha256'] is None or f['image_sha256']!=A['image_sha256'])];field='timestamp_counter' if A['timestamp_counter'] is not None and any(f['timestamp_counter'] is not None for f in others) else 'folder_ordinal';others=[f for f in others if f[field] is not None and A[field] is not None];others.sort(key=lambda f:(abs(f[field]-A[field]),f['frame_PRIVATE']));neighbors=[];seen=set()
   for f in others:
    marker=f['image_sha256'] or f['frame_PRIVATE']
    if marker in seen:continue
    seen.add(marker);neighbors.append(f)
    if len(neighbors)==2:break
   wrong=[f for f in frames if f['platform']==pl and f['scene_PRIVATE']!=A['scene_PRIVATE'] and f['suffix']==A['suffix'] and f['eligible_for_extra_media']];wrong.sort(key=lambda f:hashlib.sha256((A['frame_PRIVATE']+'|wrong|'+f['frame_PRIVATE']).encode()).hexdigest());anchors.append({'anchor_PRIVATE':A['frame_PRIVATE'],'platform':pl,'retrieval_coordinate':field,'available_neighbors':len(neighbors)})
   for j,B in enumerate(neighbors):pairs.append({'pair_PRIVATE':A['frame_PRIVATE']+'_'+str(j),'kind':'same_scene_distinct_sensor_candidate','A':A,'B':B,'counter_difference':B[field]-A[field],'counter_unit_seconds_verified':False})
   if wrong:pairs.append({'pair_PRIVATE':A['frame_PRIVATE']+'_wrong','kind':'wrong_scene_control','A':A,'B':wrong[0],'counter_difference':None,'counter_unit_seconds_verified':False})
 dump(H/'FRAMES_PRIVATE.json',frames);dump(H/'ANCHORS_PRIVATE.json',anchors);dump(H/'PAIR_CANDIDATES_PRIVATE.json',pairs);dump(H/'INVENTORY_SUMMARY.json',{'state':'PASS_CURRENT_TARGET_WHITELIST_SENSOR_ONLY_NEIGHBOR_INVENTORY','platforms':summary,'protected_frame_identities_excluded':len(protected),'protected_image_hashes_exclusion_only':len(denied_images),'protected_media_read':False,'no_external_model_or_mask_cache':True,'engineering_anchors':len(anchors),'pair_attempts':len(pairs),'real_pair_candidates':sum(x['kind']!='wrong_scene_control' for x in pairs),'wrong_scene_controls':sum(x['kind']=='wrong_scene_control' for x in pairs),'no_GT_or_model_prediction_used':True,'sensor_registry_sha256':sha(R/'research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json'),'registration_not_yet_run':True});print(summary);print({'anchors':len(anchors),'pairs':len(pairs)})
if __name__=='__main__':main()
