"""Lock all exposed development anchors, before model predictions or task scoring."""
from inventory105 import *
import datetime
def main():
 frames=rd(H/'FRAMES_PRIVATE.json'); bymeta={x['metapath']:x for x in frames}
 # Waymo has no legacy content registry. Hash only explicitly allowed inputs.
 for f in frames:
  if f['eligible_for_extra_media'] and f['image_sha256'] is None:f['image_sha256']=sha(f['imagepath'])
 pairs=[];anchors=[]
 for pl in ['waymo','drone','quad']:
  rows=rd(R/f'research/round99/input/TARGET_{pl.upper()}128_PRIVATE.json');seen=set()
  for r in rows:
   A=bymeta[str(Path(r['metapath']).resolve())];key=A['frame_PRIVATE']
   if key in seen:continue
   seen.add(key);rec={'A':A,'B_pair':None,'C_pair':None,'status':'protected_content_alias_frozen_fallback' if not A['eligible_for_extra_media'] else 'planned'}
   if A['eligible_for_extra_media']:
    others=[f for f in frames if f['platform']==pl and f['scene_PRIVATE']==A['scene_PRIVATE'] and f['suffix']==A['suffix'] and f['eligible_for_extra_media'] and f['image_sha256']!=A['image_sha256']]
    field='timestamp_counter' if A['timestamp_counter'] is not None and any(f['timestamp_counter'] is not None for f in others) else 'folder_ordinal'
    others=[f for f in others if f[field] is not None and A[field] is not None];others.sort(key=lambda f:(abs(f[field]-A[field]),f['frame_PRIVATE']));unique=[];hashes=set()
    for B in others:
     if B['image_sha256'] in hashes:continue
     hashes.add(B['image_sha256']);unique.append(B)
     if len(unique)==2:break
    for j,B in enumerate(unique):
     pid=key+'_'+str(j);rec[['B_pair','C_pair'][j]]=pid;pairs.append({'pair_PRIVATE':pid,'kind':'same_scene_distinct_sensor_candidate','A':A,'B':B,'counter_difference':B[field]-A[field],'counter_unit_seconds_verified':False})
    rec['available_neighbors']=len(unique);rec['retrieval_coordinate']=field
   anchors.append(rec)
 dump(H/'FULL_ANCHORS_PRIVATE.json',anchors);dump(H/'FULL_PAIRS_PRIVATE.json',pairs)
 lock={'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pairs':len(pairs),'anchors':len(anchors),'pair_sha256':sha(H/'FULL_PAIRS_PRIVATE.json'),'anchor_sha256':sha(H/'FULL_ANCHORS_PRIVATE.json'),'selection':'same-scene/suffix distinct RGB hashes, closest counter then frame hash; first B, second C, never swap after eligibility','thresholds':'identical stage0 numeric forward+reverse PnP and heldout cycle','reused_probe_records':'only exact pair, sensors, registration method and seed','no_model_or_GT':True,'all_fallbacks_retained':True}
 dump(H/'experiment/FULL_REGISTRATION_LOCK.json',lock);print(json.dumps({k:v for k,v in lock.items() if k not in ['anchor_sha256','pair_sha256']},indent=2))
if __name__=='__main__':main()
