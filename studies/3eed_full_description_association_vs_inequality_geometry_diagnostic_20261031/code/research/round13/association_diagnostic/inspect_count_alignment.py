"""Authorized others_num/others structural alignment only; no geometry values."""
from pathlib import Path
import argparse,collections,json
from inspect_schema import read,sha,dump,shape

def main(a):
 if sha(a.release)!=a.release_sha256:raise ValueError('Release hash')
 r=read(a.release)
 if r['scope']!='R13_schema_only_count_alignment_no_values_no_IoU' or r['IoU_or_query_selection_permitted']:raise ValueError('Scope')
 for name in ('plan','program','cases','scoring_index'):
  if sha(r[name+'_path'])!=r[name+'_sha256']:raise ValueError('Binding')
 for item in r['bound_seals']:
  if sha(item['path'])!=item['sha256']:raise ValueError('Seal')
 cases=read(r['cases_path']);index={x['id']:x for x in read(r['scoring_index_path'])};counts=collections.Counter()
 for c in cases:
  ix=index[c['id']];p=Path(ix['meta_path'])
  if sha(p)!=ix['metadata_container_sha256']:raise ValueError('Metadata identity')
  for obj in read(p)['ground_info']:
   others=obj.get('others');declared=obj.get('others_num')
   counts['ground_info_entries']+=1
   if isinstance(others,list) and isinstance(declared,int) and not isinstance(declared,bool):
    counts['count_field_int_and_list']+=1
    counts['declared_equals_list_length' if declared==len(others) else 'count_mismatch']+=1
    counts['other_entries_total']+=len(others)
    for entry in others:
     counts['dict_with_bbox_3d_other_shape7' if isinstance(entry,dict) and shape(entry.get('bbox_3d_other'))==[7] else 'other_schema_missing_or_different']+=1
   else:counts['undefined_count_comparison']+=1
 result={'state':'completed_schema_only_declared_count_alignment','counts':dict(counts),
   'n_cases':len(cases),'GT_coordinate_values_class_values_captions_or_ID_values_output':False,
   'IoU_transform_dedup_or_query_selection_computed':False}
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);dump(a.output/'SUMMARY.json',result)
 dump(a.output/'SEALED.json',{'state':'sealed_R13_schema_count_alignment','files':{'SUMMARY.json':sha(a.output/'SUMMARY.json')},'program_sha256':sha(Path(__file__)),'release_sha256':sha(a.release)})
 print(json.dumps({**result,'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
