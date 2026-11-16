"""Exact existing nearest-counter sensor candidate policy, before geometry/model/GT."""
from pathlib import Path
import json,hashlib,datetime,functools,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round109';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 lock=rd(H/'experiment/INPUT_LOCK.json');assert sha(H/'ANCHORS_STAGE0_PRIVATE.json')==lock['anchor_sha256'];assert not (H/'experiment/REGISTRATION_LOCK.json').exists();frames=rd(R/'research/round105/FRAMES_PRIVATE.json');groups=collections.defaultdict(list)
 for f in frames:groups[(f['platform'],f['scene_PRIVATE'],f['suffix'])].append(f)
 @functools.lru_cache(maxsize=None)
 def digest(path):return sha(path)
 def imagehash(f):
  assert f['eligible_for_extra_media'];h=digest(f['imagepath']);assert f['image_sha256'] is None or h==f['image_sha256'];return h
 anchors=rd(H/'ANCHORS_STAGE0_PRIVATE.json');pairs=[]
 for rec in anchors:
  A=rec['A']
  if not A['eligible_for_extra_media']:continue
  A['image_sha256']=imagehash(A);others=[x for x in groups[(A['platform'],A['scene_PRIVATE'],A['suffix'])] if x['eligible_for_extra_media'] and x['frame_PRIVATE']!=A['frame_PRIVATE']];field='timestamp_counter' if A['timestamp_counter'] is not None and any(x['timestamp_counter'] is not None for x in others) else 'folder_ordinal';others=[x for x in others if x[field] is not None and A[field] is not None];others.sort(key=lambda f:(abs(f[field]-A[field]),f['frame_PRIVATE']));seen={A['image_sha256']};count=0
  for x in others:
   h=imagehash(x)
   if h in seen:continue
   seen.add(h);B=dict(x,image_sha256=h);pid=A['frame_PRIVATE']+'_'+str(count);rec[['B_pair','C_pair'][count]]=pid;pairs.append({'pair_PRIVATE':pid,'kind':'same_scene_distinct_sensor_candidate','A':dict(A),'B':B,'counter_difference':B[field]-A[field],'counter_unit_seconds_verified':False});count+=1
   if count==2:break
  rec['available_neighbors']=count;rec['retrieval_coordinate']=field
 dump(H/'ANCHORS_PRIVATE.json',anchors);dump(H/'PAIRS_PRIVATE.json',pairs);dump(H/'PAIR_SUMMARY.json',{'anchors':len(anchors),'pairs':len(pairs),'platforms':{pl:{'anchors':sum(x['A']['platform']==pl for x in anchors),'pairs':sum(x['A']['platform']==pl for x in pairs),'protected_alias_fallback':sum(x['A']['platform']==pl and not x['A']['eligible_for_extra_media'] for x in anchors)} for pl in ['waymo','drone','quad']},'no_model_GT_or_media_decode':True,'first_B_second_C_locked_before_geometric_acceptance':True});dump(H/'experiment/REGISTRATION_LOCK.json',{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pair_manifest_sha256':sha(H/'PAIRS_PRIVATE.json'),'anchor_sha256':sha(H/'ANCHORS_PRIVATE.json'),'R105_protocol':'same exact SIFT/depthPnP numeric heldout plus reverse/cycle thresholds; no foregroundmask or providedpose','original_registration_code_sha256':sha(R/'research/round105/register_views105.py'),'sensor_backend_sha256':sha(R/'crossplatform/round12/sensor_pnp.py'),'camera_helpers_sha256':sha(R/'crossplatform/round12/temporal_association.py'),'no_candidate_replacement_after_gate':True,'FINAL_access':False});print(json.dumps(rd(H/'PAIR_SUMMARY.json'),indent=2))
if __name__=='__main__':main()
