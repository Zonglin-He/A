"""Schema-only expansion of observed ground_info[].others, no value output."""
from pathlib import Path
import argparse,collections,json
from inspect_schema import sha,read,dump,shape

def main(a):
 if sha(a.release)!=a.release_sha256:raise ValueError('Release hash')
 r=read(a.release)
 if r['scope']!='R13_schema_only_nested_others_no_values_no_IoU' or not r['schema_inspection_permitted'] or r['IoU_or_query_selection_permitted']:raise ValueError('Scope')
 for name in ('plan','program','cases','scoring_index'):
  if sha(r[name+'_path'])!=r[name+'_sha256']:raise ValueError('Binding '+name)
 for item in r['bound_seals']:
  if sha(item['path'])!=item['sha256']:raise ValueError('Seal')
 index={x['id']:x for x in read(r['scoring_index_path'])};cases=read(r['cases_path'])
 keytypes=collections.defaultdict(collections.Counter);shapes=collections.Counter();lengths=[];private=[]
 def walk(value,path):
  keytypes[path][type(value).__name__]+=1
  if isinstance(value,dict):
   for key,item in value.items():walk(item,path+'.'+key)
  elif isinstance(value,list):
   if not value or all(not isinstance(x,(dict,list)) for x in value):
    shapes[path+' shape='+json.dumps(shape(value),separators=(',',':'))]+=1
   else:
    for item in value:walk(item,path+'[]')
 for case in cases:
  ix=index[case['id']];path=Path(ix['meta_path'])
  if sha(path)!=ix['metadata_container_sha256']:raise ValueError('Metadata hash')
  objects=read(path)['ground_info'];counts=[]
  for obj in objects:
   other=obj.get('others');walk(other,'ground_info[].others')
   counts.append(len(other) if isinstance(other,list) else None)
  lengths.extend(c for c in counts if c is not None)
  private.append({'case_alias':case['case_alias'],'others_list_counts_per_annotated_entry':counts})
 result={'state':'schema_only_nested_others_no_values_no_IoU','n_cases':len(cases),
  'path_key_type_counts':{k:dict(v) for k,v in keytypes.items()},'scalar_list_shape_counts':dict(shapes),
  'others_list_count_histogram':dict(collections.Counter(map(str,lengths))),
  'no_ID_value_caption_or_numeric_coordinates_exported':True,
  'scope':'Structure only; no expansion to an all-object geometry diagnostic is authorized by this schema-only check. Root must freeze an explicit adapter before any IoU.'}
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);dump(a.output/'NESTED_SCHEMA_SUMMARY.json',result);dump(a.output/'NESTED_SCHEMA_PRIVATE.json',private)
 dump(a.output/'SEALED.json',{'state':'sealed_schema_only_nested_others','files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'release_sha256':sha(a.release),'program_sha256':sha(Path(__file__))})
 print(json.dumps({'state':result['state'],'n_cases':len(cases),'paths':result['path_key_type_counts'],'shapes':result['scalar_list_shape_counts'],'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
