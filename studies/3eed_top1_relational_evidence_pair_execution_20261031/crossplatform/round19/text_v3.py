"""Componentwise literal validation; invalid clauses stay unknown, not global veto.
Explicit subject-location grammar supplements generated relations only when a
literal located/in/at clause precedes every quoted reference. No RGB/GT inputs.
"""
import re
from semantic_graph_v4 import strict_object
from text_v2 import text_messages
def parse_caption_graph(raw,caption):
 def quote(s):
  if not isinstance(s,str) or not s.strip():return None
  hits=list(re.finditer(re.escape(s),caption,re.I))
  if not hits:return None
  return {'text':caption[hits[0].start():hits[0].end()],'span':[hits[0].start(),hits[0].end()],'spans':[[h.start(),h.end()] for h in hits],'unique_occurrence_claimed':len(hits)==1}
 try:d,_=strict_object(raw)
 except ValueError:return {'status':'NA_text_graph_contract_v3','entities':[],'attributes':[],'relations':[],'reason':'no_JSON'}
 if not {'subject','attributes','references','relations'}<=set(d) or not isinstance(d['references'],list) or not isinstance(d['relations'],list):return {'status':'NA_text_graph_contract_v3','entities':[],'attributes':[],'relations':[],'reason':'missing_components'}
 subject=quote(d['subject'])
 if subject is None:return {'status':'NA_text_graph_contract_v3','entities':[],'attributes':[],'relations':[],'reason':'subject_not_literal'}
 errors=[];entities=[{'entity':'T','quote':subject,'status':'PARSED','head':'not_used_for_spatial_execution'}];attrs=[];relations=[]
 for i,r in enumerate(d['references'][:8]):
  q=quote(r.get('phrase')) if isinstance(r,dict) else None;entities.append({'entity':'R'+str(i),'quote':q,'status':'PARSED' if q else 'UNKNOWN','head':'not_used_for_spatial_execution'})
 for i,e in enumerate(entities):
  aa=d['attributes'] if i==0 else d['references'][i-1].get('attributes',[]) if isinstance(d['references'][i-1],dict) else []
  if not isinstance(aa,list):errors.append('attributes_not_list');continue
  for a in aa:
   q=quote(a)
   if q and e['quote'] and a.casefold() in e['quote']['text'].casefold():attrs.append({'id':'A'+str(len(attrs)),'entity':e['entity'],'kind':'appearance','quote':q,'status':'PARSED'})
   else:errors.append('attribute_not_bound_to_literal_entity')
 allowed={'LEFT_OF','RIGHT_OF','ABOVE','BELOW','GLOBAL_LEFT','GLOBAL_RIGHT','GLOBAL_CENTER','NEARER_OBSERVER','FARTHER_OBSERVER','UNKNOWN'}
 for rr in d['relations'][:12]:
  if not isinstance(rr,dict):errors.append('relation_not_object');continue
  q=quote(rr.get('quote'));view=quote(rr.get('viewpoint'));ref=rr.get('reference');valid_ref=ref is None or (type(ref)==int and 0<=ref<len(entities)-1 and entities[ref+1]['status']=='PARSED');valid=bool(q and valid_ref and type(rr.get('target_is_subject'))==bool and rr.get('predicate') in allowed and rr.get('frame') in ('camera_image','observer_depth','UNKNOWN'))
  if not valid:errors.append('relation_component_invalid_unknown');continue
  rid='SCENE' if ref is None else 'R'+str(ref);relations.append({'id':'L'+str(len(relations)),'subject':'T' if rr['target_is_subject'] else rid,'object':rid if rr['target_is_subject'] else 'T','predicate':rr['predicate'],'frame':rr['frame'] if view else 'UNKNOWN','quote':q,'viewpoint_quote':view,'status':'PARSED','origin':'generated_literal_component'})
 # A literal spatial clause is measured as a partial constraint, never as a
 # complete description. Orientation clauses and referenced-object positions
 # are excluded. No change to old semantic spatial margins or geometry.
 end=min([e['quote']['span'][0] for e in entities[1:] if e['quote'] and e['quote']['span'][0]>subject['span'][1]]+[len(caption)])
 prefix=caption[subject['span'][0]:end]
 pat=r'\b(?:located|positioned|stationary|parked|standing|situated|in|at)\b[^.;]{0,65}?\b(left|right|center|centre|middle)\b[^.;]{0,35}?\b(image|view|picture|frame|screen)\b'
 for m in re.finditer(pat,prefix,re.I):
  text=m.group();direction=m.group(1).lower()
  if re.search(r'\b(facing|pointing|front|rear|behind|head|wheel)\b',text,re.I):continue
  if direction in ('left','right') and re.search(r'\b(center|centre|middle)\b',text,re.I):continue
  pred={'left':'GLOBAL_LEFT','right':'GLOBAL_RIGHT','center':'GLOBAL_CENTER','centre':'GLOBAL_CENTER','middle':'GLOBAL_CENTER'}[direction]
  if any(r['subject']=='T' and r['object']=='SCENE' and r['predicate']==pred and r['frame']=='camera_image' for r in relations):continue
  relations.append({'id':'L'+str(len(relations)),'subject':'T','object':'SCENE','predicate':pred,'frame':'camera_image','quote':quote(text),'viewpoint_quote':quote(m.group(2)),'status':'PARSED','origin':'literal_subject_location_clause'})
 return {'status':'PARSED_text_graph','entities':entities,'attributes':attrs,'relations':relations,'component_unknown_reasons':errors,'unverified_roles':True,'unknown_omitted_clauses_not_assumed_satisfied':True,'local_constraints_only':True}
