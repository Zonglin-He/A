"""Read only exposed R24 manifests/NPZ headers; no arrays, GT, model or FINAL."""
import json,hashlib,zipfile,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=R/'research/round24/input';splits={k:read(root/(k+'_PRIVATE.json')) for k in ['D208','V64','M64']};allrows=read(root/'ALL336_PRIVATE.json');assert len(allrows)==336 and len({r['id'] for r in allrows})==336;assert set(r['id'] for r in allrows)==set(r['id'] for rows in splits.values() for r in rows);assert sum(len(x) for x in splits.values())==336
 expected={'h','x','text_h','tokens','query','token_mask','boxes','scores'};counter=collections.Counter();lengths=[];schemas=[]
 for r in allrows:
  p=R/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz');fields={}
  with zipfile.ZipFile(p) as z:
   assert set(x[:-4] for x in z.namelist())==expected
   for name in z.namelist():
    with z.open(name) as f:
     version=np.lib.format.read_magic(f);shape,order,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f));fields[name[:-4]]={'shape':shape,'fortran':order,'dtype':str(dtype)}
  T=fields['tokens']['shape'][0];assert fields['query']['shape']==(256,64) and fields['boxes']['shape']==(256,6) and fields['scores']['shape']==(256,) and fields['token_mask']['shape']==(T,) and fields['text_h']['shape']==(T,288);lengths.append(T);counter[r['platform']]+=1;schemas.append({k:{'dtype':v['dtype'],'shape':list(v['shape']) if k not in ['tokens','text_h','token_mask'] else ['T',*list(v['shape'])[1:]]} for k,v in fields.items()})
 assert all(s==schemas[0] for s in schemas);out={'state':'P48_EXPOSED_CACHE_SCHEMA_HEADERS_ONLY','n':336,'splits':{k:{'n':len(v),'platform_counts':dict(collections.Counter(r['platform'] for r in v)),'manifest_sha256':sha(root/(k+'_PRIVATE.json')),'row_keysets':[list(x) for x in sorted({tuple(sorted(r)) for r in v})]} for k,v in splits.items()},'platform_counts':dict(counter),'cached_schema':schemas[0],'T_min':min(lengths),'T_max':max(lengths),'input_ids_not_cached':True,'actual_input_text_from_capture_code':'caption + literal sentinel . not mentioned','tokenization_proof_boundary':'Only projected token features/mask/length cached; actual input IDs must be reconstructed under locked tokenizer, not claimed original stored-ID exactness.','official_rawGT_class_or_boxes_or_any_FINAL_access':False,'models_arrays_replay_or_taskscore_run':False,'only_NPZ_headers_read':True};(O/'SCHEMA_HEADERS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
