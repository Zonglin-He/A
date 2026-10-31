"""R15 full-caption role hypotheses + independent observable relation matching.
Pure CPU JSON/arrays. No model/data IO, GT, source boxes/q0/text scores, or
candidate truncation. VLM agreement is a proposal, not semantic identity truth.
"""
from dataclasses import dataclass,asdict
import itertools,json,math,re
import numpy as np

@dataclass(frozen=True)
class SemanticConfig:
 max_references:int=4
 max_attributes:int=6
 max_relations:int=6
 max_masks:int=100
 max_assignments:int=25000
 text_max_new_tokens:int=1024
 visual_max_new_tokens:int=512
 image_direction_margin:float=.025
 mask_centroid_uncertainty_extent_fraction:float=.10
 observer_depth_margin_m:float=.25
 minimum_exclusive_depth_points:int=8
 global_left:tuple=(0.,.40)
 global_center:tuple=(.30,.70)
 global_right:tuple=(.60,1.)
DEFAULT_CONFIG=SemanticConfig()
class ContractError(ValueError):pass
SUPPORT='SUPPORT';CONTRADICT='CONTRADICT';UNKNOWN='UNKNOWN'
PREDICATES={'LEFT_OF','RIGHT_OF','ABOVE','BELOW','GLOBAL_LEFT','GLOBAL_CENTER','GLOBAL_RIGHT','NEARER_OBSERVER','FARTHER_OBSERVER','NEAR','FRONT_OF','BEHIND','UNKNOWN'}
HEAD_FAMILIES={'person':(1,),'pedestrian':(1,),'man':(1,),'woman':(1,),'people':(1,),'child':(1,),'boy':(1,),'girl':(1,),'worker':(1,),'walker':(1,),'cyclist':(1,),'biker':(1,),'car':(3,6,8),'vehicle':(3,6,8),'sedan':(3,6,8),'van':(3,6,8),'truck':(3,6,8),'bus':(3,6,8),'taxi':(3,6,8),'suv':(3,6,8),'jeep':(3,6,8),'bike':(2,),'bicycle':(2,),'skateboard':(41,)}

def strict_object(raw):
 if not isinstance(raw,str):raise ContractError('JSON string required')
 text=raw.strip();removed=False
 if text.startswith('```'):
  m=re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```',text)
  if m is None or text.count('```')!=2:raise ContractError('Only one complete JSON fence, no prose')
  text=m.group(1);removed=True
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise ContractError('Duplicate JSON key')
   out[k]=v
  return out
 try:obj=json.loads(text,object_pairs_hook=pairs)
 except (ValueError,TypeError) as e:raise ContractError('Invalid strict JSON') from e
 if not isinstance(obj,dict):raise ContractError('JSON object required')
 return obj,removed

def quote(text,value):
 if not isinstance(value,str) or not value.strip() or value=='UNKNOWN':return None
 # Whole contiguous literal, distinctive quote; nested spans are allowed.
 pat=(r'(?<!\w)' if value[0].isalnum() else '')+re.escape(value)+(r'(?!\w)' if value[-1].isalnum() else '')
 hits=list(re.finditer(pat,text,re.I))
 if len(hits)!=1:return None
 hit=hits[0];return {'text':text[hit.start():hit.end()],'span':[hit.start(),hit.end()]}
def exactkeys(obj,keys):
 if not isinstance(obj,dict) or set(obj)!=set(keys):raise ContractError('Exact schema fields required: '+','.join(keys))
def entity_parse(item,text,eid):
 exactkeys(item,('quote','head'));q=quote(text,item['quote']);head=item['head']
 if q is None or not isinstance(head,str) or not head or len(head)>64 or re.search(r'(?<!\w)'+re.escape(head)+r'(?!\w)',q['text'],re.I) is None:return {'entity':eid,'status':UNKNOWN,'quote':None,'head':None}
 return {'entity':eid,'status':'PARSED','quote':q,'head':head.casefold()}

def parse_caption_graph(raw,caption,config=DEFAULT_CONFIG):
 try:
  d,fenced=strict_object(raw);exactkeys(d,('status','target','references','attributes','relations'))
  if d['status']=='UNKNOWN':return {'status':'UNKNOWN_text_graph','entities':[],'attributes':[],'relations':[],'format_wrapper_removed':fenced}
  if d['status']!='RESOLVED':raise ContractError('Strict status required')
  target=entity_parse(d['target'],caption,'T')
  if target['status']!= 'PARSED':raise ContractError('Distinctive literal main target/head required')
  if not all(isinstance(d[k],list) for k in ('references','attributes','relations')):raise ContractError('Graph collections must be lists')
  if len(d['references'])>config.max_references or len(d['attributes'])>config.max_attributes or len(d['relations'])>config.max_relations:raise ContractError('Graph schema cap exceeded; no truncation')
  entities=[target]
  for i,item in enumerate(d['references']):
   exactkeys(item,('entity','quote','head','distinct_from_target'))
   eid='R'+str(i+1)
   if item['entity']!=eid or item['distinct_from_target'] is not True:raise ContractError('Ordered distinct reference role required')
   ref=entity_parse({'quote':item['quote'],'head':item['head']},caption,eid)
   # Full main noun phrase may contain a reference modifier. Reject identical
   # grounded span only, not nested/overlapping modifier spans in general.
   if ref['status']=='PARSED' and ref['quote']['span']==target['quote']['span']:ref.update(status=UNKNOWN,quote=None,head=None)
   entities.append(ref)
  ids={x['entity'] for x in entities};attrs=[];rels=[]
  for i,item in enumerate(d['attributes']):
   exactkeys(item,('id','entity','quote','kind'))
   if item['id']!='A'+str(i+1) or item['entity'] not in ids or item['kind'] not in ('appearance','color','clothing','pose','action','part','other'):raise ContractError('Attribute IDs/role/kind contract')
   q=quote(caption,item['quote']);attrs.append({'id':item['id'],'entity':item['entity'],'kind':item['kind'],'quote':q,'status':'PARSED' if q else UNKNOWN})
  for i,item in enumerate(d['relations']):
   exactkeys(item,('id','subject','object','quote','predicate','frame','viewpoint_quote'))
   if item['id']!='L'+str(i+1) or item['subject'] not in ids or item['object'] not in ids|{'SCENE'} or item['subject']==item['object'] or item['predicate'] not in PREDICATES or item['frame'] not in ('camera_image','observer_depth','object_intrinsic','world','UNKNOWN'):raise ContractError('Relation schema/role contract')
   q=quote(caption,item['quote']);view=quote(caption,item['viewpoint_quote']);rels.append({**{k:item[k] for k in ('id','subject','object','predicate','frame')},'quote':q,'viewpoint_quote':view,'status':'PARSED' if q else UNKNOWN})
  return {'status':'PARSED_text_graph','entities':entities,'attributes':attrs,'relations':rels,'format_wrapper_removed':fenced,'literal_roles_are_unverified_hypotheses':True,'nested_target_reference_spans_allowed':True}
 except (ContractError,TypeError) as e:return {'status':'NA_text_graph_contract','reason':str(e),'entities':[],'attributes':[],'relations':[]}

def parse_visual_graph(raw,candidates,graph):
 """Sparse full-scene claims; omitted role/mask/attribute pairs stay UNKNOWN."""
 try:
  d,fenced=strict_object(raw);exactkeys(d,('status','roles','attribute_checks'))
  if d['status']=='UNKNOWN':return {'status':'UNKNOWN_visual_graph','role_proposals':{},'attribute_claims':{},'format_wrapper_removed':fenced}
  if d['status']!='RESOLVED' or graph['status']!='PARSED_text_graph':raise ContractError('Resolved text/visual contract')
  if not isinstance(d['roles'],list) or not isinstance(d['attribute_checks'],list):raise ContractError('List roles/checks required')
  ids={x['entity'] for x in graph['entities']};attrs={x['id']:x for x in graph['attributes']};look={c['candidate_id']:c['mask_id'] for c in candidates}
  if len(look)!=len(candidates) or len(set(look.values()))!=len(candidates):raise ContractError('Unique physical-mask candidate mapping required')
  roles={};checks={};ignored=0
  for rr in d['roles']:
   exactkeys(rr,('entity','candidates'))
   if rr['entity'] not in ids or rr['entity'] in roles or not isinstance(rr['candidates'],list):raise ContractError('Unique role proposal list required')
   values=[]
   for answer in rr['candidates']:
    if isinstance(answer,str) and answer in look:values.append(look[answer])
    elif answer!='UNKNOWN':ignored+=1
   roles[rr['entity']]=sorted(set(values))
  for item in d['attribute_checks']:
   exactkeys(item,('attribute','candidate','evidence'))
   if item['attribute'] not in attrs or not isinstance(item['candidate'],str) or item['candidate'] not in look or item['evidence'] not in (SUPPORT,CONTRADICT,UNKNOWN):ignored+=1;continue
   key=(item['attribute'],look[item['candidate']])
   if key in checks:raise ContractError('Duplicate attribute/mask claim')
   checks[key]=item['evidence']
  return {'status':'PARSED_visual_graph','role_proposals':roles,'attribute_claims':checks,'ignored_invalid_claim_n':ignored,'format_wrapper_removed':fenced,'omitted_masks_and_claims_are_UNKNOWN_not_negative':True,'visual_evidence_not_independent_truth':True}
 except (ContractError,TypeError) as e:return {'status':'NA_visual_graph_contract','reason':str(e),'role_proposals':{},'attribute_claims':{}}

def combine_visual_graphs(parsed):
 if len(parsed)!=2:raise ValueError('Exactly2 numbering calls')
 valid=all(x['status']=='PARSED_visual_graph' for x in parsed);roles={};checks={};conflicts=0
 if valid:
  for key in set(parsed[0]['role_proposals'])|set(parsed[1]['role_proposals']):roles[key]=sorted(set(parsed[0]['role_proposals'].get(key,[]))&set(parsed[1]['role_proposals'].get(key,[])))
  for key in set(parsed[0]['attribute_claims'])|set(parsed[1]['attribute_claims']):
   aa=parsed[0]['attribute_claims'].get(key,UNKNOWN);bb=parsed[1]['attribute_claims'].get(key,UNKNOWN);checks[key]=aa if aa==bb else UNKNOWN;conflicts+=aa!=bb
 return {'defined':valid,'role_proposals':roles,'attribute_claims':checks,'attribute_order_conflict_n':conflicts,'two_numbering_agreement_is_not_correctness':True}

def _out(status,reason,**extra):return {'status':status,'reason':reason,**extra}
def _literal_frame(rel):
 if rel['status']!='PARSED' or not rel['viewpoint_quote']:return False
 view=rel['viewpoint_quote']['text'].casefold()
 if rel['frame']=='camera_image':return bool(re.search(r'\b(image|picture|view|frame|screen|camera|photo|photograph)\b',view))
 if rel['frame']=='observer_depth':return bool(re.search(r'\b(observer|viewer|camera)\b',view))
 return False

def relation_evidence(rel,left,right,config=DEFAULT_CONFIG):
 """Tri-state observation. Never derive depth from model XYZ or body facing."""
 if not _literal_frame(rel):return _out(UNKNOWN,'viewpoint_absent_ambiguous_or_unimplemented')
 pred=rel['predicate'];text=rel['quote']['text'].casefold()
 if re.search(r"\b(not|never|neither|without)\b|n['’]t\b",text):return _out(UNKNOWN,'negated_relation_not_implemented_no_automatic_positive_interpretation')
 expected={'LEFT_OF':r'\bleft\b','RIGHT_OF':r'\bright\b','ABOVE':r'\b(above|higher)\b','BELOW':r'\b(below|lower)\b','GLOBAL_LEFT':r'\bleft\b','GLOBAL_RIGHT':r'\bright\b','GLOBAL_CENTER':r'\b(center|centre|middle)\b','NEARER_OBSERVER':r'\b(closer|nearer)\b','FARTHER_OBSERVER':r'\b(farther|further)\b'}
 quote_pred=rel.get('diagnostic_original_predicate',pred)
 if quote_pred not in expected or re.search(expected[quote_pred],text) is None:return _out(UNKNOWN,'predicate_quote_not_executable_or_mismatch')
 if left.get('mask_centroid_defined',True) is False or (right is not None and right.get('mask_centroid_defined',True) is False):return _out(UNKNOWN,'mask_centroid_explicitly_undefined')
 if left.get('touches_image_border',False) or (right is not None and right.get('touches_image_border',False)):return _out(UNKNOWN,'truncated_mask_centroid_or_extent_not_full_object')
 if pred.startswith('GLOBAL_') and rel['frame']=='camera_image' and right is None:
  c=np.asarray(left.get('mask_centroid_norm2'),float)
  if c.shape!=(2,) or not np.isfinite(c).all():return _out(UNKNOWN,'undefined_mask_centroid')
  lo,hi={'GLOBAL_LEFT':config.global_left,'GLOBAL_CENTER':config.global_center,'GLOBAL_RIGHT':config.global_right}[pred];x=float(c[0]);margin=config.image_direction_margin
  value=SUPPORT if lo<=x<=hi else CONTRADICT if x<lo-margin or x>hi+margin else UNKNOWN
  return _out(value,'mask_centroid_in_semantic_interval_not_fixed_point',observed_normalized_x=x,semantic_interval=[lo,hi])
 if right is None:return _out(UNKNOWN,'pair_reference_unobserved')
 if pred in ('LEFT_OF','RIGHT_OF','ABOVE','BELOW') and rel['frame']=='camera_image':
  axis=0 if pred in ('LEFT_OF','RIGHT_OF') else 1
  def interval(obj):
   if obj.get('mask_bbox_defined',True) is False:return None
   c=np.asarray(obj.get('mask_centroid_norm2'),float);e=np.asarray(obj.get('mask_bbox_norm4'),float)
   if c.shape!=(2,) or e.shape!=(4,) or not np.isfinite(c).all() or not np.isfinite(e).all():return None
   u=config.mask_centroid_uncertainty_extent_fraction*max(float(e[axis+2]-e[axis]),0.)
   return (float(c[axis]-u),float(c[axis]+u))
  a=interval(left);b=interval(right)
  if a is None or b is None:return _out(UNKNOWN,'undefined_mask_position')
  if pred in ('RIGHT_OF','BELOW'):a,b=b,a
  margin=config.image_direction_margin;status=SUPPORT if a[1]+margin<b[0] else CONTRADICT if a[0]>b[1]+margin else UNKNOWN
  return _out(status,'mask_centroid_uncertainty_intervals',directional_gap=float(b[0]-a[1]))
 if pred in ('NEARER_OBSERVER','FARTHER_OBSERVER') and rel['frame']=='observer_depth':
  a=left.get('depth_camera',{});b=right.get('depth_camera',{})
  def valid(d):return d.get('defined') is True and d.get('n',0)>=config.minimum_exclusive_depth_points and all(isinstance(d.get(k),(int,float)) and math.isfinite(d[k]) for k in ('q10','q90')) and d['q10']<=d['q90']
  if not valid(a) or not valid(b):return _out(UNKNOWN,'insufficient_exclusive_global_zbuffer_depth')
  if pred=='FARTHER_OBSERVER':a,b=b,a
  m=config.observer_depth_margin_m;status=SUPPORT if a['q90']+m<b['q10'] else CONTRADICT if a['q10']>b['q90']+m else UNKNOWN
  return _out(status,'exclusive_camera_depth_quantile_intervals',nearer_q90_m=float(a['q90']),farther_q10_m=float(b['q10']))
 return _out(UNKNOWN,'predicate_frame_combination_unimplemented')

def _class_evidence(entity,obj):
 if entity['status']!='PARSED':return UNKNOWN
 family=HEAD_FAMILIES.get(entity['head'])
 if family is None:return UNKNOWN
 return SUPPORT if int(obj['predicted_class_id']) in family else CONTRADICT

def match_scene_graph(graph,scene,visual,*,relation_check=True,config=DEFAULT_CONFIG):
 """All-mask exhaustive finite search. Unknown contenders cannot create unique."""
 base={'selected_target_mask_id':None,'possible_target_mask_ids':[],'supported_target_mask_ids':[],'search_complete':False,'assignment_evidence_not_identity_truth':True,'relation_check':relation_check,'relation_mode':'literal' if relation_check else 'disabled','literal_relations_satisfied':False,'config':asdict(config)}
 if graph['status']!='PARSED_text_graph' or not visual['defined']:return dict(base,status='NA_text_or_visual_unknown',assignments_evaluated=0)
 objects=scene['objects'];ids=[x['mask_id'] for x in objects]
 if len(ids)!=len(set(ids)) or len(ids)>config.max_masks:return dict(base,status='NA_mask_inventory_invalid_or_cap',assignments_evaluated=0)
 if any(re.fullmatch(r'M[0-9]{5}',x) is None for x in ids):raise ValueError('Stable physical mask Mnnnnn required')
 byid={x['mask_id']:x for x in objects};ents=graph['entities'];attrs=graph['attributes'];rels=graph['relations'];unary={};domains={};evidence={}
 for ent in ents:
  eid=ent['entity'];domains[eid]=[]
  for obj in objects:
   mid=obj['mask_id'];checks=[{'kind':'predicted_family_only','status':_class_evidence(ent,obj),'identity_certified':False}]
   for a in attrs:
    if a['entity']==eid:checks.append({'kind':'visual_attribute','id':a['id'],'status':visual['attribute_claims'].get((a['id'],mid),UNKNOWN) if a['status']=='PARSED' else UNKNOWN,'two_view_model_claim_not_independent_truth':True})
   if relation_check:
    for rel in rels:
     if rel['subject']==eid and rel['object']=='SCENE':checks.append({'kind':'independent_relation','id':rel['id'],**relation_evidence(rel,obj,None,config)})
   status=CONTRADICT if any(x['status']==CONTRADICT for x in checks) else SUPPORT if all(x['status']==SUPPORT for x in checks) else UNKNOWN
   unary[(eid,mid)]=status;evidence[(eid,mid)]=checks
   if status!=CONTRADICT:domains[eid].append(mid)
 nproduct=math.prod(len(domains[e['entity']]) for e in ents)
 if nproduct>config.max_assignments:return dict(base,status='NA_assignment_search_cap',assignments_evaluated=0,unpruned_assignment_product=nproduct,all_mask_unary_evidence=[{'entity':e,'mask_id':m,'checks':c} for (e,m),c in evidence.items()])
 possible=set();supported=set();evaluated=0;complete_assignments=[]
 for members in itertools.product(*(domains[e['entity']] for e in ents)):
  evaluated+=1
  if len(set(members))!=len(members):continue
  assignment={e['entity']:v for e,v in zip(ents,members)};states=[unary[(eid,mid)] for eid,mid in assignment.items()];independent_support=0;checks=[]
  for rel in (rels if relation_check else []):
   if rel['object']=='SCENE':ev=relation_evidence(rel,byid[assignment[rel['subject']]],None,config)
   else:ev=relation_evidence(rel,byid[assignment[rel['subject']]],byid[assignment[rel['object']]],config)
   checks.append({'id':rel['id'],**ev});states.append(ev['status']);independent_support+=ev['status']==SUPPORT
  if CONTRADICT in states:continue
  target=assignment['T'];possible.add(target)
  # No unmeasured language constraint passes by default. Full independent
  # support and a two-numbering target proposal are necessary, not truth.
  good=bool(states) and all(x==SUPPORT for x in states) and (independent_support>0 if relation_check else True) and target in visual['role_proposals'].get('T',[])
  if good:supported.add(target)
  complete_assignments.append({'assignment':assignment,'fully_supported':good,'relation_evidence':checks})
 unique=len(possible)==1 and possible==supported
 status='unique_supported' if unique else 'ambiguous_possible_targets' if len(possible)>1 else 'NA_unknown_constraints_or_no_supported_assignment'
 return dict(base,status=status,selected_target_mask_id=next(iter(possible)) if unique else None,possible_target_mask_ids=sorted(possible),supported_target_mask_ids=sorted(supported),search_complete=True,assignments_evaluated=evaluated,unpruned_assignment_product=nproduct,all_mask_unary_evidence=[{'entity':e,'mask_id':m,'checks':c} for (e,m),c in evidence.items()],assignments=complete_assignments,unknown_is_not_pass=True,omitted_masks_not_removed_by_VLM_beam=True,literal_relations_satisfied=bool(unique and relation_check))


def proposal_consensus(visual):
 """Pure full-query candidate proposal control, not a graph uniqueness proof."""
 candidates=sorted(set(visual.get('role_proposals',{}).get('T',[]))) if visual.get('defined') else []
 return {'status':'proposal_control_unique' if len(candidates)==1 else 'proposal_control_ambiguous_or_unknown','selected_target_mask_id':candidates[0] if len(candidates)==1 else None,'search_complete':True,'semantic_allmask_uniqueness_claimed':False,'two_numbering_proposals':candidates,'relation_mode':'VLM_fullquery_proposal_only','language_attributes_and_relations_remain_in_model_input':True,'proposal_agreement_not_correctness':True}

def match_relation_flip_diagnostic(graph,scene,visual,config=DEFAULT_CONFIG):
 """Counterfactual whitelist interpreter only, never executable generated code."""
 import copy
 flipped=copy.deepcopy(graph);changed=[]
 pairs={'LEFT_OF':'RIGHT_OF','RIGHT_OF':'LEFT_OF','GLOBAL_LEFT':'GLOBAL_RIGHT','GLOBAL_RIGHT':'GLOBAL_LEFT','NEARER_OBSERVER':'FARTHER_OBSERVER','FARTHER_OBSERVER':'NEARER_OBSERVER'}
 for rel in flipped.get('relations',[]):
  p=rel['predicate'];frame=rel['frame'];quote_text=rel['quote']['text'].casefold() if rel.get('quote') else ''
  allowed_frame=(p in ('LEFT_OF','RIGHT_OF','GLOBAL_LEFT','GLOBAL_RIGHT') and frame=='camera_image') or (p in ('NEARER_OBSERVER','FARTHER_OBSERVER') and frame=='observer_depth')
  if p in pairs and allowed_frame and _literal_frame(rel) and not re.search(r"\b(not|never|neither|without)\b|n['’]t\b",quote_text):
   rel['diagnostic_original_predicate']=p;rel['predicate']=pairs[p];changed.append(rel['id'])
 if not changed:
  result=match_scene_graph(graph,scene,visual,relation_check=True,config=config);result.update(relation_mode='flip_diagnostic',flip_applied=False,flipped_relation_ids=[],counterfactual_override=[],diagnostic_not_method_candidate=True,raw_caption_quotes_unmodified=True,nonapplicable_policy='exact_original_fullgraph_output_not_source_cancellation')
  return result
 overrides=[{'relation_id':r['id'],'original_predicate':r['diagnostic_original_predicate'],'counterfactual_predicate':r['predicate']} for r in flipped['relations'] if 'diagnostic_original_predicate' in r]
 result=match_scene_graph(flipped,scene,visual,relation_check=True,config=config);result.update(relation_mode='flip_diagnostic',literal_relations_satisfied=False,flip_applied=True,flipped_relation_ids=changed,counterfactual_override=overrides,diagnostic_not_method_candidate=True,raw_caption_quotes_unmodified=True)
 return result
