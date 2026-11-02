"""R51 full5912 same unique CPU G-carrier repair, fixed c=.5, no external feedback/model.

Groups are candidate hypotheses, not known independent objects. Each carrier
is one actual member's whole point-support medoid, not a cross-query union.
Its globally visible point pixels may also belong to alternative hypotheses;
sharing is recorded, not removed from the RGB measurements.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import json
import os
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from proposal_groups51 import build_proposal_groups
from attributes51 import color_support
from noassociation51 import pixel_color_record
from evidence51_fidelity import attr_evidence,object_geometry,raw_relations,propagate_measured,reference_hypotheses

MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
ARMS=['source','grouped_replace_all_c05','grouped_no_relation_c05']
FACTOR=.5
CANONICAL='crossplatform/round8/full5912_attempt001'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def load(p,fields):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in fields}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name
def pack(parts,width=None):
 off=np.r_[0,np.cumsum([len(p) for p in parts])].astype(np.int64)
 flat=np.concatenate(parts).astype(np.int64,copy=False) if parts else np.empty((0,) if width is None else (0,width),np.int64)
 return off,flat

def pure_graph_fields(p):
 import ijson
 from ijson.common import ObjectBuilder
 selected={'text_graph_unchanged','entity_attribute_requests'};builders={}
 with Path(p).open('rb') as f:
  for prefix,event,value in ijson.parse(f):
   first=prefix.split('.',1)[0]
   if first in selected:builders.setdefault(first,ObjectBuilder()).event(event,value)
 assert set(builders)==selected
 return {k:b.value for k,b in builders.items()}

def proposal_medoids(group_info,offsets,flat):
 """Actual member point Jaccard mean; exact mean tie -> lowest original q."""
 bags=[np.sort(np.asarray(flat[offsets[q]:offsets[q+1]],np.int64)) for q in range(256)]
 assert all(len(x)==len(np.unique(x)) for x in bags)
 groups=group_info['groups'];assert [g['group_id'] for g in groups]==list(range(len(groups)))
 medoids=[];details=[];mean_values=np.full(256,np.nan)
 for group in groups:
  members=sorted(map(int,group['members']));assert members and len(members)==len(set(members))
  sets={q:frozenset(map(int,bags[q])) for q in members};assert all(len(sets[q])>=6 for q in members)
  means=[]
  for q in members:
   values=[]
   for other in members:
    intersection=len(sets[q]&sets[other]);values.append(intersection/(len(sets[q])+len(sets[other])-intersection))
   value=float(np.mean(np.asarray(values,np.float64)));means.append(value);mean_values[q]=value
  best=max(means);winner=min(q for q,v in zip(members,means) if v==best)
  medoids.append(winner)
  details.append({'group_id':group['group_id'],'member_queries_PRIVATE':members,
      'mean_point_jaccard_by_member':means,'medoid_query_PRIVATE':winner,
      'medoid_point_count':len(bags[winner]),'selection_uses_GT_or_source_score':False,
      'cross_query_point_union_used':False})
 return np.asarray(medoids,np.int64),bags,mean_values,details

def carrier_from_medoids(group_info,group_arrays,visible_raw_ids,visible_pixels,visible_uv):
 offsets=group_arrays['support_offsets'];flat=group_arrays['support_indices']
 medoids,bags,means,details=proposal_medoids(group_info,offsets,flat)
 points=[bags[q] for q in medoids];positions=[np.flatnonzero(np.isin(visible_raw_ids,ids,assume_unique=True)) for ids in points]
 pixels=[visible_pixels[ix] for ix in positions];M=len(medoids)
 rect=np.zeros((M,4),np.int64);centroid=np.full((M,2),np.nan)
 for m,px in enumerate(pixels):
  if len(px):rect[m]=np.r_[px.min(0),px.max(0)+1];centroid[m]=visible_uv[positions[m]].mean(0)
 po,pf=pack(points);vo,vf=pack([visible_raw_ids[ix] for ix in positions]);mo,mf=pack(pixels,2)
 qid=np.asarray(group_info['query_group_ids'],np.int64);assert qid.shape==(256,) and ((qid>=-1)&(qid<M)).all()
 W=np.zeros((256,M),np.float64);known=qid>=0;W[np.flatnonzero(known),qid[known]]=1.
 arrays={'group_medoid_queries':medoids,'member_mean_point_jaccard':means,
   'instance_point_offsets':po,'instance_point_indices':pf,
   'instance_visible_point_offsets':vo,'instance_visible_point_indices':vf,
   'instance_crop_mask_pixel_offsets':mo,'instance_crop_mask_pixel_xy':mf,
   'instance_rect_xyxy':rect,'instance_visible_centroid_uv':centroid,
   'query_group_ids':qid,'query_instance_conditional_weights':W}
 return arrays,positions,details

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_FULL_GROUP_MEDOID_G_NO_GT' and auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_FULL_GROUP_MEDOID_G' and plan['n']==5912 and plan['manifest']==MANIFEST
 assert plan['appearance_factor']==FACTOR and plan['single_registered_G_carrier_mechanism_repair'] is True
 assert plan['method_selected_before_full_run'] is True and plan['selected_method']=='grouped_replace_all_c05'
 assert plan['no_new_Qwen_models_GT_or_TTA'] is True and 0<plan['max_wall_seconds']<=7200
 assert plan['canonical_source_root']==CANONICAL
 roots={k:path(plan[k]) for k in ('prepared_root','reference_root','canonical_source_root')}
 expected={MANIFEST}|{plan[k]+'/SEALED.json' for k in roots}
 assert set(plan['inputs_sha256'])==expected
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(path(f))==h,('pin',f)
 for f in ('crossplatform/round51/grouped_full51.py','crossplatform/round51/proposal_groups51.py','crossplatform/round51/attributes51.py','crossplatform/round51/noassociation51.py','crossplatform/round51/evidence51_fidelity.py'):
  assert plan['programs_sha256'][f]==sha(ROOT/f)
 seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 assert seals['prepared_root']['state']=='R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT'
 assert seals['prepared_root']['n']==5912 and seals['prepared_root'].get('engineering_limit') is None
 assert seals['reference_root']['state']=='R51_FULL5912_SELECTED_METHOD_REFERENCE_ONLY_HEADSPAN_CAPTURE_NO_GT' and seals['reference_root']['n']==5912
 assert seals['reference_root']['external_visual_calls']==0
 assert seals['prepared_root']['external_vision_outputs_or_mask_derived_cache_read'] is False
 rows=read(ROOT/MANIFEST);assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir(parents=True)
 for d in ('samples_PRIVATE','carriers_PRIVATE','attributes_PRIVATE','references_PRIVATE','cases_PRIVATE','groups_PRIVATE'):(out/d).mkdir()
 ledger=[];journal=out/'INPUT_LEDGER_PRIVATE.jsonl'
 def bind(p,role,expected,sid=None,fields=None):
  p=Path(p);h=sha(p);r={'path_PRIVATE':str(p),'role':role,'sha256':h,'id_PRIVATE':sid,'selected_fields':fields}
  with journal.open('a') as stream:stream.write(json.dumps(r)+'\n');stream.flush();os.fsync(stream.fileno())
  ledger.append(r);assert h==expected,role
  return p
 def member(key,rel,role,sid,fields=None):return bind(roots[key]/rel,role,seals[key]['files'][rel],sid,fields)
 start=time.perf_counter();counts=collections.Counter();done=[]
 try:
  for row in rows:
   if time.perf_counter()-start>plan['max_wall_seconds']:raise TimeoutError('Fixed CPU repair wall cap')
   tick=time.perf_counter();sid=row['id']
   pc=read(member('prepared_root','cases_PRIVATE/'+sid+'.json','RAW_SAFE_INPUT_BINDINGS',sid))
   fields=['source_boxes','source_scores','visible_raw_point_indices','visible_pixels_xy','visible_uv',
           'query_support_point_offsets','query_support_point_indices','query_component_counts']
   p=load(member('prepared_root','arrays_PRIVATE/'+sid+'.npz','RAW_SAFE_GLOBAL_VISIBILITY_SOURCE_ONLY',sid,fields),fields)
   gi,ga=build_proposal_groups(source_boxes=p['source_boxes'],point_offsets=p['query_support_point_offsets'],
       point_indices=p['query_support_point_indices'],component_counts=p['query_component_counts'])
   assert gi['semantic_identity_certified'] is False
   dump(out/'groups_PRIVATE'/(sid+'.json'),gi)
   source=load(member('canonical_source_root',sid+'.npz','CANONICAL_SOURCE_ONLY_BRIDGE',sid,
       ['source_boxes_raw6','source_text_scores256']),['source_boxes_raw6','source_text_scores256'])
   exact(source['source_boxes_raw6'],p['source_boxes'],'canonical R8 source boxes bytes')
   exact(source['source_text_scores256'],p['source_scores'],'canonical R8 source native scores bytes')
   pure=pure_graph_fields(member('prepared_root','attributes_PRIVATE/'+sid+'.json','PURE_TEXT_FIELDS_ONLY',sid,['text_graph_unchanged','entity_attribute_requests']))
   graph=pure['text_graph_unchanged'];assert hashlib.sha256(row['caption'].encode()).hexdigest()==pc['caption_sha256']
   bind(row['imagepath'],'ORIGINAL_RGB',pc['raw_bindings']['image_sha256'],sid)
   bind(row['pointpath'],'ORIGINAL_RAW_XYZ',pc['raw_bindings']['point_sha256'],sid)
   with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
   pt=Path(row['pointpath']);raw=np.load(pt,allow_pickle=False) if pt.suffix=='.npy' else np.fromfile(pt,np.float32).reshape(-1,4)
   raw=np.ascontiguousarray(raw[:,:3],np.float32);assert len(raw)==pc['n_raw_points']
   W,H=pc['image_size_wh'];assert rgb.shape[:2]==(H,W)
   carrier,positions,medoids=carrier_from_medoids(gi,ga,p['visible_raw_point_indices'],p['visible_pixels_xy'],p['visible_uv'])
   px=p['visible_pixels_xy'];scene_color=color_support(rgb[px[:,1],px[:,0]]);M=len(medoids)
   attrs={'instances':[],'entity_attribute_requests':pure['entity_attribute_requests'],'text_graph_unchanged':graph,
     'schema':'R51_PROPOSAL_MEDOID_VISIBLE_PIXEL_ATTRIBUTES','scene_visible_color_soft_support':scene_color,
     'n_scene_visible_pixels':len(px),'candidate_alternatives_shared_points_are_NOT_removed':True,
     'not_foreground_or_semantic_identity_certified':True,'cross_group_union_used':False}
   for m,ids in enumerate(positions):
    selected=px[ids];attrs['instances'].append(pixel_color_record(rgb[selected[:,1],selected[:,0]],scene_color,len(px),m))
   e,ek=attr_evidence(attrs,'T')
   refs_case=read(member('reference_root','cases_PRIVATE/'+sid+'.json','SAME_GROUNDER_REFERENCE_BINDINGS',sid))
   assert refs_case['graph_sha256']==pc['pure_text_graph_sha256']
   subjectroute=refs_case['subject']
   subjectp=member('reference_root',subjectroute['file'],'REFERENCE_STAGE_ORIGINAL_R8_SUBJECT_BRIDGE',sid,['boxes','scores'])
   assert sha(subjectp)==subjectroute['sha256']
   subject=load(subjectp,['boxes','scores'])
   exact(subject['boxes'],p['source_boxes'],'reference stage original subject boxes')
   exact(subject['scores'],p['source_scores'],'reference stage original subject scores')
   refs={};ref_details={};ref_arrays={}
   for ref in graph['references']:
    entity=ref['entity'];route=refs_case['references'][entity]
    rp=member('reference_root',route['file'],'SAME_GROUNDER_REFERENCE_BOXES_LOGODDS',sid,['boxes','logodds']);assert sha(rp)==route['sha256']
    call=read(member('reference_root','calls_PRIVATE/'+sid+'__'+entity+'.json','REFERENCE_SAMPLING_AND_MASK_RECEIPT',sid))
    assert call['state']=='COMPLETED' and call['binding']['point_sample_sha256']==refs_case['point_sample_sha256']
    assert call['binding']['point_indices_sha256']==refs_case['point_indices_sha256'] and route['mask']==call['text_head_mask']
    z=load(rp,['boxes','logodds'])
    if route['mask']['reference_usable']:
     hs,info=reference_hypotheses(raw,carrier,z['boxes'],z['logodds'],row['platform'],attrs,entity,FACTOR)
    else:
     hs=[];info={'unknown_probability':1.,'source_unknown_probability':1.,'source_truncated_probability':0.,
       'hypotheses':0,'retained_probability':0.,'source_top3_instance_ids':[],
       'source_hypothesis_logits':[None]*M,'source_probabilities':[0.]*M,'source_only_top3_fixed_before_color':True}
    decorated=[]
    for h in hs:
     h=dict(h,instance_domain='raw3d',uncertainty_mass_explicit=True)
     h['xy_norm2']=[h['xy_pixel'][0]/W,h['xy_pixel'][1]/H] if h['xy_pixel'] is not None else None
     h['width_norm']=(h['rect'][2]-h['rect'][0])/W;decorated.append(h)
    refs[entity]=decorated;ref_details[entity]={'hypotheses':decorated,'distribution':info,
       'independent_reference_query_indices_not_subject_indices':True}
    ref_arrays[entity+'_source_logits']=np.array([x if x is not None else -np.inf for x in info['source_hypothesis_logits']])
    ref_arrays[entity+'_source_probability']=np.asarray(info['source_probabilities'])
   objects=object_geometry(carrier,(W,H));pr,pk=raw_relations(graph,objects,refs,(W,H))
   F,cues,known=propagate_measured(p['source_scores'],carrier['query_instance_conditional_weights'],FACTOR*e,pr,ek,pk)
   nrpr=pr.copy();nrpr[:,1]=0.;nrpk=pk.copy();nrpk[:,1]=False
   noR,noRcues,noRknown=propagate_measured(p['source_scores'],carrier['query_instance_conditional_weights'],FACTOR*e,nrpr,ek,nrpk)
   scores=np.stack([p['source_scores'].astype(float),F,noR]);queries=scores.argmax(1).astype(np.int64)
   selected=p['source_boxes'][queries]
   sample=out/'samples_PRIVATE'/(sid+'.npz')
   np.savez_compressed(sample,arm_names=np.array(ARMS),scores=scores,queries=queries,selected_original_boxes=selected,
      source_boxes=p['source_boxes'],source_scores=p['source_scores'],appearance_factor_values=np.array([FACTOR]),
      no_relation_scores=noR[None],no_relation_queries=queries[2:3],no_relation_selected_original_boxes=selected[2:3],
      components=cues[None],raw_PR=pr[None],raw_PR_known=pk[None],
      cue_query_known=known['query_known'][None],cue_anchors=known['anchors'][None],cue_query_known_weight_mass=known['query_known_weight_mass'][None],
      no_relation_components=noRcues[None],instance_attribute_e=e,instance_attribute_known=ek,
      query_instance_conditional_weights=carrier['query_instance_conditional_weights'],group_medoid_queries=carrier['group_medoid_queries'])
   np.savez_compressed(out/'carriers_PRIVATE'/(sid+'.npz'),**carrier)
   dump(out/'attributes_PRIVATE'/(sid+'.json'),attrs)
   np.savez_compressed(out/'references_PRIVATE'/(sid+'.npz'),**ref_arrays)
   dump(out/'references_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'appearance_factor_values':[FACTOR],'references_by_factor':[ref_details]})
   unavailable=not bool(known['query_known'].any());nr_unavailable=not bool(noRknown['query_known'].any())
   status={ARMS[0]:{'unknown':False,'fallback':False,'no_parameter_update':True},
     ARMS[1]:{'unknown':unavailable,'fallback':unavailable,'no_parameter_update':True},
     ARMS[2]:{'unknown':nr_unavailable,'fallback':nr_unavailable,'no_parameter_update':True}}
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'platform':row['platform'],'arm_statuses':status,
      'medoids':medoids,'group_count':M,'unbound_queries':int((carrier['query_group_ids']<0).sum()),
      'source_boxes_scores_canonical_byteexact':True,'semantic_or_foreground_identity_certified':False,
      'visible_points_shared_with_alternative_groups_retained':True,'whole_rectangle_RGB_sampled':False,
      'no_new_Qwen_control_claim':True,'external_vision_outputs_or_mask_derived_caches_read':False,
      'pure_text_graph_sha256':pc['pure_text_graph_sha256'],'seconds':time.perf_counter()-tick})
   counts['n']+=1;counts['groups']+=M;counts['group_color_unknown']+=int((~ek).sum())
   counts['unbound_queries']+=int((carrier['query_group_ids']<0).sum());counts['references']+=len(refs)
   done.append(sid);dump(out/'PROGRESS.json',{'n':len(done),'total':5912,'seconds':time.perf_counter()-start})
   if len(done)%16==0:print('R51_GROUP_MEDOID_G_REPAIR',len(done),5912,flush=True)
  dump(out/'INPUT_LEDGER_PRIVATE.json',{'files':ledger,'new_RGB_XYZ_measurement_only_no_external_cached_feedback':True})
  dump(out/'COUNTS.json',dict(counts));dump(out/'COSTS.json',{'seconds':time.perf_counter()-start,'model_forwards':0,'Qwen_forwards':0,'DINO_forwards':0,'optimizer_steps':0})
  dump(out/'SEALED.json',{'state':'R51_FULL_GROUP_MEDOID_G_PREDICTIONS_NO_GT','n':len(done),'arm_names':ARMS,
    'appearance_factor_values':[FACTOR],'single_registered_mechanism_repair':True,'method_selected_before_full_run':True,
    'proposal_groups_built_online_same_frozen_algorithm':True,
    'all_selected_boxes_original_source_256':True,'external_vision_outputs_or_mask_derived_caches_read':False,
    'TTA_or_new_Qwen_controls_run_or_claimed':False,'plan_sha256':sha(args.plan),
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'INPUT_LEDGER_PRIVATE.json',{'files':ledger})
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_n':len(done),'automatic_retry':False,'GT_read':False});raise

if __name__=='__main__':main()
