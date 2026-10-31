"""Explicit schema-only label access: no values/captions/coordinates/IoU output."""
from pathlib import Path
import argparse,collections,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def shape(x):
 if not isinstance(x,list):return []
 if not x:return [0]
 sub=[shape(v) for v in x]
 return [len(x)]+sub[0] if all(v==sub[0] for v in sub) else [len(x),'ragged']
def main(a):
 if sha(a.release)!=a.release_sha256:raise ValueError('Release hash')
 r=read(a.release)
 if r['scope']!='R13_schema_only_no_values_no_IoU' or not r['schema_inspection_permitted'] or r['IoU_or_query_selection_permitted']:raise ValueError('Scope')
 for name in ('plan','program','cases','scoring_index'):
  if sha(r[name+'_path'])!=r[name+'_sha256']:raise ValueError(name+' binding')
 for item in r['bound_seals']:
  if sha(item['path'])!=item['sha256']:raise ValueError('Seal binding')
 cases=read(r['cases_path']);index={v['id']:v for v in read(r['scoring_index_path'])}
 topkeys=collections.Counter();objectkeys=collections.Counter();types=collections.defaultdict(collections.Counter)
 shapes=collections.Counter();gcounts=[];private=[];seen=set();targets=collections.Counter();ids=collections.Counter()
 for c in cases:
  ix=index[c['id']];mp=Path(ix['meta_path'])
  if sha(mp)!=ix['metadata_container_sha256']:raise ValueError('Original metadata hash')
  d=read(mp);objects=d.get('ground_info')
  if not isinstance(objects,list):raise ValueError('ground_info is not annotated list')
  if not 0<=ix['object_index']<len(objects):raise ValueError('Target join outside list')
  target=objects[ix['object_index']]
  targets['target_object_dict' if isinstance(target,dict) else 'target_not_dict']+=1
  if isinstance(target,dict):targets['target_bbox_field_present' if 'bbox_3d' in target else 'target_bbox_field_missing']+=1
  if str(mp) not in seen:
   seen.add(str(mp));topkeys.update(d.keys());gcounts.append(len(objects))
   for key,value in d.items():types['container.'+key][type(value).__name__]+=1
   for obj in objects:
    if not isinstance(obj,dict):types['ground_info_element'][type(obj).__name__]+=1;continue
    objectkeys.update(obj.keys())
    for key,value in obj.items():
     types['ground_info.'+key][type(value).__name__]+=1
     if key.lower() in ('id','object_id','instance_id','track_id'):ids[key]+=1
    shapes[json.dumps(shape(obj.get('bbox_3d')),separators=(',',':'))]+=1
  private.append({'case_alias':c['case_alias'],'annotated_list_n':len(objects),
    'other_list_entries_n':max(0,len(objects)-1),'target_keys':sorted(target) if isinstance(target,dict) else [],
    'target_bbox_shape':shape(target.get('bbox_3d')) if isinstance(target,dict) else None})
 summary={'state':'completed_authorized_schema_only_no_values_no_IoU',
  'n_cases':len(cases),'unique_metadata_containers':len(seen),
  'container_key_presence_counts':dict(topkeys),'ground_info_key_presence_counts':dict(objectkeys),
  'field_type_counts':{k:dict(v) for k,v in types.items()},
  'bbox_shape_counts':dict(shapes),'explicit_ID_key_presence_counts':dict(ids),
  'annotated_list_entry_n_min':min(gcounts),'annotated_list_entry_n_max':max(gcounts),
  'total_annotated_entries_unique_containers':sum(gcounts),'target_schema_counts':dict(targets),
  'limitations':['ground_info is available annotated geometry, not certified complete scene instances',
   'No ID inferred; key presence does not establish unique physical object identity',
   'Only structural keys/types/counts/shapes retained; captions/numeric bbox values not output',
   'No transform, IoU, dedup, query selection, model, G or task metric recomputation performed']}
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 dump(a.output/'SCHEMA_KEYS_SUMMARY.json',summary);dump(a.output/'SCHEMA_QUALIFICATION_PRIVATE.json',private)
 dump(a.output/'SEALED.json',{'state':'sealed_R13_schema_only_diagnostic',
  'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'release_sha256':sha(a.release),
  'plan_sha256':r['plan_sha256'],'program_sha256':sha(Path(__file__))})
 print(json.dumps({'state':summary['state'],'n_cases':len(cases),'unique_containers':len(seen),
  'bbox_shape_counts':summary['bbox_shape_counts'],'ID_key_presence':dict(ids),'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
