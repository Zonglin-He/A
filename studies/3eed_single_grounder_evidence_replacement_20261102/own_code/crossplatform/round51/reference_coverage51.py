"""Unexecuted pure-text reference vocabulary vs saved source training support.

No new model, image, XYZ, target class/box label, method score or GT access.
Unmatched head spelling is NOT proof of unseen semantics or zero capability.
"""
from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import re
import time
ROOT=Path(__file__).resolve().parents[2]
MANIFEST='research/round24/input/ALL336_PRIVATE.json'
SOURCE_SUMMARY='research/round6/source_data_audit/PUBLIC_SCALAR_SUMMARY.json'
DATASET_CODE='crossplatform/round3/p0/upstream/src/joint_det_dataset.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p

def pure_text_fields(p):
 import ijson
 from ijson.common import ObjectBuilder
 allowed={'text_graph_unchanged','entity_attribute_requests'};builders={}
 with Path(p).open('rb') as f:
  for prefix,event,value in ijson.parse(f):
   first=prefix.split('.',1)[0]
   if first in allowed:builders.setdefault(first,ObjectBuilder()).event(event,value)
 assert set(builders)==allowed
 return {k:b.value for k,b in builders.items()}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_PURE_TEXT_REFERENCE_SOURCE_SUPPORT_DIAGNOSTIC_NO_GT' and auth['plan_sha256']==sha(args.plan)
 assert plan['state']=='ROOT_LOCKED_R51_PURE_TEXT_REFERENCE_SOURCE_SUPPORT_DIAGNOSTIC'
 assert plan['n']==336 and plan['manifest']==MANIFEST and plan['source_training_summary']==SOURCE_SUMMARY
 assert plan['no_media_new_models_method_scores_or_target_GT'] is True
 expected={MANIFEST,SOURCE_SUMMARY,plan['prepared_root']+'/SEALED.json'}
 assert set(plan['inputs_sha256'])==expected
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(path(f))==h,('pin',f)
 assert plan['programs_sha256'][DATASET_CODE]==sha(ROOT/DATASET_CODE)
 assert plan['programs_sha256'][Path(__file__).relative_to(ROOT).as_posix()]==sha(__file__)
 parsed=ast.parse((ROOT/DATASET_CODE).read_text());constants={}
 for node in parsed.body:
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name) and target.id in ('CLASS_MAPPINGS','WAYMO_SYNONYMS'):
     constants[target.id]=ast.literal_eval(node.value)
 assert set(constants)=={'CLASS_MAPPINGS','WAYMO_SYNONYMS'}
 source=read(ROOT/SOURCE_SUMMARY)['source_train'];classes=source['class_counts']
 assert sum(classes.values())==source['eligible_actual_loader_annos']==3682
 assert set(classes)<=set(constants['CLASS_MAPPINGS'])
 aliases=collections.defaultdict(set)
 for c,words in constants['WAYMO_SYNONYMS'].items():
  if c in classes:
   for word in [c]+words:aliases[word.strip().casefold()].add(c)
 rows=read(ROOT/MANIFEST);assert len(rows)==336
 prepared=path(plan['prepared_root']);seal=read(prepared/'SEALED.json')
 assert prepared.parent==ROOT/'crossplatform/round51' and seal['n']==336 and seal['engineering_limit'] is None
 assert seal['state']=='R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT'
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir(parents=True)
 count=collections.Counter();heads=collections.Counter();byplatform={p:collections.Counter() for p in ('drone','quad')};private=[]
 public_roles={p:{role:collections.Counter() for role in ('subject','reference')} for p in ('drone','quad')}
 color_counts=collections.Counter()
 start=time.perf_counter()
 for row in rows:
  rel='attributes_PRIVATE/'+row['id']+'.json';p=prepared/rel;assert sha(p)==seal['files'][rel]
  attributes=pure_text_fields(p);graph=attributes['text_graph_unchanged'];base=graph.get('base_text',graph)
  requests={r['entity']:r for r in attributes['entity_attribute_requests']}
  entities=[{'entity':'T','role':'subject','head':base.get('category',graph.get('category','')),
             'declared_own_attributes':bool((base.get('explicit_appearance_field') or {}).get('text','').strip()),
             'attribute_scope':'explicit_appearance_field_only_for_subject_attribute_claim'}]
  entities += [{'entity':r['entity'],'role':'reference','head':r.get('head',''),
                'declared_own_attributes':bool(r.get('has_role_attributes',False)),
                'attribute_scope':'existing_pure_text_has_role_attributes_flag'} for r in graph['references']]
  for entity in entities:
   head=entity['head'];assert isinstance(head,str)
   normalized=head.strip().casefold();matched=sorted(aliases.get(normalized,()))
   exact_class=normalized in classes
   class_token_matches=sorted(c for c in classes if re.search(r'(?<!\w)'+re.escape(c)+r'(?!\w)',normalized))
   status='SOURCE_TARGET_CLASS_OR_OFFICIAL_SYNONYM_MATCH' if matched else 'NO_EXACT_OFFICIAL_TARGET_CLASS_LEXICAL_MATCH_NOT_CAPABILITY_PROOF'
   key=entity['role']+':'+normalized;heads[key]+=1;byplatform[row['platform']][key]+=1;count[status]+=1
   role_counts=public_roles[row['platform']][entity['role']];role_counts['entities']+=1;role_counts[status]+=1
   role_counts['exact_source_class_name']+=exact_class;role_counts['has_exact_source_class_token']+=bool(class_token_matches)
   request=requests.get(entity['entity'],{});colors=request.get('recognized_color_words',[])
   assert all(isinstance(w,str) for w in colors)
   role_counts['measurable_color_requested']+=bool(colors)
   literal=request.get('literal_attributes','');assert isinstance(literal,str)
   role_counts['nonempty_literal_without_recognized_color_UNKNOWN']+=bool(literal.strip()) and not colors
   declared_no_color=entity['declared_own_attributes'] and not colors
   role_counts['graph_declared_attributes_without_color_UNKNOWN']+=declared_no_color
   role_counts['complex_appearance_unimplemented_status_UNKNOWN']+=request.get('unimplemented_appearance_status','').startswith('UNKNOWN')
   color_counts.update(colors)
   private.append({'id_PRIVATE':row['id'],'entity_PRIVATE':entity['entity'],'role':entity['role'],'head_PRIVATE':head,
       'matched_source_positive_target_classes':matched,'exact_class_name':exact_class,
       'exact_source_class_token_matches':class_token_matches,'lexical_status':status,
       'recognized_color_words':colors,'graph_declared_attributes_without_color_UNKNOWN':bool(declared_no_color),
       'declared_attribute_scope':entity['attribute_scope'],
       'unimplemented_appearance_status':request.get('unimplemented_appearance_status','UNKNOWN_missing_request')})
 count['rows']=len(rows);count['subject_entities']=sum(x['role']=='subject' for x in private);count['reference_entities']=sum(x['role']=='reference' for x in private)
 dump(out/'SUMMARY.json',{'source_train_target_class_counts':classes,
      'source_train_total_positive_expressions':3682,'dataset_declared_target_class_names':sorted(constants['CLASS_MAPPINGS']),
      'counts':dict(count),'by_platform_role':{p:{role:dict(v) for role,v in rr.items()} for p,rr in public_roles.items()},
      'recognized_color_request_counts':dict(color_counts),
      'wall_building_not_source_direct_positive_target_classes':not bool({'wall','building'}&set(classes)),
      'not_a_grounding_accuracy_measurement':True,'unmatched_lexical_head_not_equivalent_to_semantically_unseen':True,
      'context_exposure_and_zero_shot_transfer_not_measured':True,'reference_native_logodds_not_calibrated_UNKNOWN_posterior':True,
      'pure_complex_appearance_not_reparsed_or_rule_inferred':True,
      'unknown_complex_proxy_definition':'subject explicit appearance field/reference existing has_role_attributes, no recognized color; not a complete semantic census'})
 dump(out/'HEAD_COUNTS_PRIVATE.json',{'role_head_counts':dict(heads),'by_platform':{k:dict(v) for k,v in byplatform.items()}})
 dump(out/'ROWS_PRIVATE.json',private)
 dump(out/'SEALED.json',{'state':'R51_PURE_TEXT_SOURCE_REFERENCE_SUPPORT_DIAGNOSTIC_NO_GT','n':336,'entities':len(private),
      'plan_sha256':sha(args.plan),'seconds':time.perf_counter()-start,'model_media_method_scores_or_target_GT_reads':0,
      'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})

if __name__=='__main__':main()
