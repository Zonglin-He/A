"""Input-only bytehash duplicate check; no image pixels or point arrays decoded."""
from pathlib import Path
import json,hashlib,collections
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):h.update(z)
 return h.hexdigest()
rows=json.loads((D/'TARGET5912_INPUT_FIELDS_PRIVATE.json').read_text())['rows_PRIVATE'];groups=collections.defaultdict(list)
for r in rows:groups[(r['platform'],r['scene_PRIVATE'])].append(r)
checked=[]
for key,rr in sorted(groups.items()):
 bytime=collections.defaultdict(list)
 for r in rr:bytime[r['input_fields_PRIVATE']['timestamp']].append(r)
 timestamp,dup=next((t,xs) for t,xs in sorted(bytime.items()) if len(xs)>1);pair=sorted(dup,key=lambda x:x['numeric_index'])[:2];payload=[]
 for r in pair:
  p=Path(r['meta_path_PRIVATE']).parent;payload.append({'folder_PRIVATE':p.name,'timestamp':timestamp,'pose_sha256':hashlib.sha256(json.dumps(r['input_fields_PRIVATE']['pose'],sort_keys=True).encode()).hexdigest(),'image_sha256':sha(p/'image.jpg'),'point_sha256':sha(p/'lidar.bin')})
 checked.append({'platform':key[0],'scene_PRIVATE':key[1],'examples_PRIVATE':payload,'timestamp_equal':True,'pose_equal':payload[0]['pose_sha256']==payload[1]['pose_sha256'],'image_bytes_identical':payload[0]['image_sha256']==payload[1]['image_sha256'],'point_bytes_identical':payload[0]['point_sha256']==payload[1]['point_sha256']})
out={'state':'R12_5scene_repeatedtimestamp_examples_exact_inputcontent_audit','program_sha256':sha(__file__),'input_metadata_table_sha256':sha(D/'TARGET5912_INPUT_FIELDS_PRIVATE.json'),'per_scene_PRIVATE':checked,'no_image_pixel_pointarray_GT_taskscore_model_GPU_decode':True};p=D/'DUPLICATE_TIMESTAMP_CONTENT_AUDIT.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'n_pairs':5,'all_identical_image_point':all(x['image_bytes_identical'] and x['point_bytes_identical'] for x in checked),'per_scene':[{'platform':x['platform'],'scene':x['scene_PRIVATE'],'image_same':x['image_bytes_identical'],'P_same':x['point_bytes_identical'],'dirs':[p['folder_PRIVATE'] for p in x['examples_PRIVATE']]} for x in checked],'receipt_sha256':sha(p)}))
