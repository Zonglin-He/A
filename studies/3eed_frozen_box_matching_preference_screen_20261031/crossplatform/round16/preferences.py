"""Partial candidate preferences, not amodal pseudo boxes. No task GT API.

Mask IDs are hypotheses. Different IDs alone are never negative evidence.
Source boxes are read-only supports; all queries associated to one component
form a positive set, avoiding contradictory labels for duplicate queries.
"""
import json,re
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RELATIONS=('none','left_of','right_of','above','below','nearer','farther','global_left','global_right','global_center','unknown')
def parse_text(raw,caption):
 try:
  s=raw.strip();s=re.sub(r'^```(?:json)?\s*|\s*```$','',s);o=json.loads(s)
  assert set(o)=={'target','reference','relation','frame'}
  assert isinstance(o['target'],str) and o['target'].strip() and o['target'].casefold() in caption.casefold()
  assert o['reference'] is None or (isinstance(o['reference'],str) and o['reference'].strip() and o['reference'].casefold() in caption.casefold())
  assert o['relation'] in RELATIONS and o['frame'] in ('image','camera_depth','unknown','none')
  assert o['reference'] is None or o['target'].casefold()!=o['reference'].casefold()
  if re.search(r'\b(not|never|neither|without)\b',caption,re.I):assert o['relation'] in ('none','unknown')
  return {'status':'parsed',**o}
 except (ValueError,AssertionError,TypeError,KeyError):return {'status':'unparsed','target':None,'reference':None,'relation':'unknown','frame':'unknown'}

def parse_visual(raw,inventory):
 try:
  s=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip());o=json.loads(s);assert set(o)=={'target','reference'}
  lookup={c['candidate_id']:c['mask_index'] for c in inventory}
  assert o['target'] in lookup
  assert o['reference'] is None or o['reference'] in lookup
  assert o['reference']!=o['target']
  return {'status':'parsed','target':lookup[o['target']],'reference':None if o['reference'] is None else lookup[o['reference']]}
 except (ValueError,AssertionError,TypeError,KeyError):return {'status':'unparsed','target':None,'reference':None}

def relation_status(text,target,reference):
 p=text['relation'];frame=text['frame']
 if p=='none':return 'not_required'
 if text['status']!='parsed' or p=='unknown':return 'unknown'
 c=np.asarray(target['mask_centroid_norm2']);r=None if reference is None else np.asarray(reference['mask_centroid_norm2'])
 if c.shape!=(2,) or target['touches_image_border']:return 'unknown'
 if p.startswith('global_') and frame=='image':
  lo,hi={'global_left':(0.,.4),'global_right':(.6,1.),'global_center':(.35,.65)}[p]
  return 'support' if lo<=c[0]<=hi else 'contradict' if c[0]<lo-.03 or c[0]>hi+.03 else 'unknown'
 if reference is None or reference['touches_image_border']:return 'unknown'
 if p in ('left_of','right_of','above','below') and frame=='image':
  axis=0 if p in ('left_of','right_of') else 1
  a,b=float(c[axis]),float(r[axis]);a,b=(b,a) if p in ('right_of','below') else (a,b)
  # Visible-mask extent adds uncertainty; overlapping silhouettes are unknown.
  ea=float(target['mask_extent_norm2'][axis])*.15;eb=float(reference['mask_extent_norm2'][axis])*.15
  return 'support' if a+ea+.03<b-eb else 'contradict' if a-ea>b+eb+.03 else 'unknown'
 if p in ('nearer','farther') and frame=='camera_depth':
  a=target['depth_camera'];b=reference['depth_camera']
  if not a['defined'] or not b['defined']:return 'unknown'
  if p=='farther':a,b=b,a
  return 'support' if a['q90']+.5<b['q10'] else 'contradict' if a['q10']>b['q90']+.5 else 'unknown'
 return 'unknown'

def component_mapping(scene,boxes):
 """Depth-connected exclusive support; no visible-surface box refitting."""
 points=np.asarray(scene['visible_points_model']);objects=scene['objects'];components=[];diagnostics=[]
 for o in objects:
  ids=np.asarray(o['exclusive_visible_indices'],int);p=points[ids]
  if len(p)<4:
   components.append(np.empty(0,int));diagnostics.append({'status':'too_few_points','n':len(p)});continue
  radius=float(np.clip(o['depth_camera']['median']*.015 if o['depth_camera']['defined'] else .25,.12,1.))
  tree=cKDTree(p);dist,nn=tree.query(p,k=min(8,len(p)),distance_upper_bound=radius)
  source=np.repeat(np.arange(len(p)),nn.shape[1]);target=nn.ravel();ok=target<len(p)
  graph=coo_matrix((np.ones(ok.sum()),(source[ok],target[ok])),shape=(len(p),len(p))).tocsr()
  nc,labels=connected_components(graph,directed=False);sizes=np.bincount(labels);winner=int(sizes.argmax());use=ids[labels==winner]
  accepted=len(use)>=4 and len(use)/len(p)>=.6
  components.append(use if accepted else np.empty(0,int));diagnostics.append({'status':'dominant_component' if accepted else 'depth_components_ambiguous','n':len(p),'component_n':len(use),'radius_m':radius,'components':nc})
 Q=len(boxes);M=len(objects);support=np.zeros((Q,M),int);inside_n=np.zeros(Q,int)
 for q,b in enumerate(boxes):
  if (b[3:]<=0).any():continue
  inside=(np.abs(points-b[:3])<=b[3:]/2+.1).all(1);inside_n[q]=inside.sum()
  for m,ids in enumerate(components):support[q,m]=inside[ids].sum()
 assignment=np.full(Q,-1,int)
 for q in range(Q):
  if M==0:continue
  order=np.argsort(-support[q],kind='stable');m=int(order[0]);n=int(support[q,m]);second=int(support[q,order[1]]) if M>1 else 0
  if n>=3 and n/max(inside_n[q],1)>=.6 and n/max(len(components[m]),1)>=.15 and n>=1.5*max(second,1):assignment[q]=m
 return {'assignment':assignment,'support':support,'inside_n':inside_n,'components':components,'component_diagnostics':diagnostics}

def make_preference(text,visual,scene,boxes,scores):
 mapping=component_mapping(scene,boxes);out={'status':'unusable','reason':None,'positive_queries':[],'negative_queries':[],'direct_query':int(np.argmax(scores)),'target_mask':None,'reference_mask':None,'relation_status':'unknown','component_diagnostics':mapping['component_diagnostics'],'query_mask_assignment':mapping['assignment'].tolist()}
 if text['status']!='parsed':out['reason']='text_contract';return out
 if any(v['status']!='parsed' for v in visual) or visual[0]!=visual[1]:out['reason']='two_orders_disagree_or_unparsed';return out
 t=visual[0]['target'];r=visual[0]['reference'];out['target_mask']=t;out['reference_mask']=r
 if text['reference'] is not None and r is None:out['reason']='reference_unobserved';return out
 objects=scene['objects'];rel=relation_status(text,objects[t],None if r is None else objects[r]);out['relation_status']=rel
 # Unimplemented/unknown required relations never become negative labels.
 if rel not in ('support','not_required'):out['reason']='relation_'+rel;return out
 comp=mapping['components'];pos=np.flatnonzero(mapping['assignment']==t);negative_masks=[]
 if not len(comp[t]) or not len(pos):out['reason']='target_component_has_no_3D_query';return out
 tp=scene['visible_points_model'][comp[t]];tc=np.median(tp,axis=0);te=np.ptp(tp,axis=0)
 for m,o in enumerate(objects):
  if m==t or not len(comp[m]) or o['predicted_class_id']!=objects[t]['predicted_class_id']:continue
  p=scene['visible_points_model'][comp[m]];sep=np.linalg.norm(np.median(p,axis=0)-tc)
  # Different predicted mask IDs or disjoint pixels alone are insufficient.
  inter=np.count_nonzero(o['mask_bool']&objects[t]['mask_bool']);union=np.count_nonzero(o['mask_bool']|objects[t]['mask_bool'])
  if sep>max(.5,.5*(np.linalg.norm(te)+np.linalg.norm(np.ptp(p,axis=0)))) and inter/max(union,1)<.05:negative_masks.append(m)
 neg=np.flatnonzero(np.isin(mapping['assignment'],negative_masks))
 if not len(neg):out['reason']='no_separated_same_class_negative';return out
 # All same-component queries are positives. No within-instance query rivalry.
 out.update(status='usable',reason='fulltext_role_consensus_measured_relation_component_sets',positive_queries=pos.tolist(),negative_queries=neg.tolist(),negative_masks=negative_masks,direct_query=int(pos[np.argmax(scores[pos])]))
 return out
