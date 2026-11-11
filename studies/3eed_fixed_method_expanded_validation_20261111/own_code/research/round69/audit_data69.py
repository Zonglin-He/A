"""Independent selected-input/physical-byte identity audit; no model or GT labels."""
from common69 import *
import collections,re,ijson
H=ROOT/'research/round69';rows=read(H/'input/ALL2304_PRIVATE.json');cohort=read(H/'input/COHORT_BY_ID_PRIVATE.json');hashes=read(H/'input/SELECTED_MEDIA_HASHES_PRIVATE.json');index={x['id']:x for x in read(H/'input/SCORING_INDEX_PRIVATE.json')};full=read(ROOT/'research/round8/experiment/full5912_gpu_inputs.json');old=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');final=read(ROOT/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json');forbidden={r['id'] for r in old+final};cache={}
def hashfile(path):
 if path not in cache:cache[path]=sha(path)
 return cache[path]
ep={hashfile(r['pointpath']) for r in old+final};ei={hashfile(r['imagepath']) for r in old+final};vp={hashfile(r['pointpath']) for r in full};vi={hashfile(r['imagepath']) for r in full};fullmap={r['id']:r for r in full};canonical=lambda s:re.sub(r'_\d+$','',s);stats=[]
for split in ['VAL1280','TRAIN1024']:
 for pl in ['drone','quad']:
  rr=[r for r in rows if cohort[r['id']]==split and r['platform']==pl];assert len(rr)==(640 if split=='VAL1280' else 512);seen=set()
  valnames={r['scene'] for r in full if r['platform']==pl};trainnames=set((ROOT/'crossplatform/upstream/data/splits'/(pl+'_train.txt')).read_text().splitlines());media=collections.Counter()
  for r in rr:
   assert r['id'] not in forbidden;ph=hashfile(r['pointpath']);ih=hashfile(r['imagepath']);assert hashes[r['id']]=={'point_sha256':ph,'image_sha256':ih};key=(ph,ih,r['caption']);assert key not in seen;seen.add(key);media[(ph,ih)]+=1
   if split=='VAL1280':assert r==fullmap[r['id']] and ph not in ep and ih not in ei
   else:assert r['scene'] in trainnames and canonical(r['scene']) not in set(map(canonical,valnames)) and ph not in vp and ih not in vi
   ix=index[r['id']];assert ix['meta_path']==r['metapath'] and hashfile(ix['meta_path'])==ix['metadata_container_sha256'];current=-1;found=False
   with Path(ix['meta_path']).open('rb') as f:
    for prefix,event,val in ijson.parse(f):
     if prefix=='ground_info.item' and event=='start_map':current+=1
     elif prefix=='ground_info.item.caption' and event=='string' and current==ix['object_index']:assert ' '.join(val.replace(',',' ,').split())==r['caption'];found=True
   assert found
  stats.append({'cohort':split,'platform':pl,'n':len(rr),'recording_sequences':len({r['scene'] for r in rr}),'unique_media_pairs':len(media),'max_descriptions_per_media':max(media.values())})
assert len({r['id'] for r in rows})==2304;out=H/'independent_data_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R69_NEW_MEDIA_EXCLUSION_FIXED_COHORTS_AND_CAPTION_INDEX','n':2304,'cohort_counts':stats,'DEV_FINAL_ID_and_selected_VAL_media_overlap':0,'TRAIN_selected_media_overlap_all_VAL_all_platforms':0,'source_or_predictions_or_GT_box_class_used':False,'protected_metadata_and_media_fingerprints_exclusion_only':True,'protected_media_decoding_inference_scoring':False,'independent_global_geographic_or_absolute_time_claim':False,'manifest_sha256':sha(H/'input/ALL2304_PRIVATE.json')};dump(out/'AUDIT_RESULT.json',result);print(result)
