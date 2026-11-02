"""R51 DIAGNOSTIC hybrid readout assembly; never feeds final-safe prediction.

The final three-factor method and its no-relation ablation are COPIED from
sealed safe_predict51 outputs. This process alone may read old external caches.
No inference, optimizer, task GT or final-method recalculation occurs here.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evidence51_fidelity import attr_evidence,object_geometry,raw_relations,propagate_measured
from attributes51 import color_support,COLOR_RGB_PROTOTYPES,COLOR_BETA_PSEUDOCOUNT,COLOR_LOGODDS_CLIP,MINIMUM_COLOR_PIXELS
ROOT=Path(__file__).resolve().parents[2]
MAIN=['source','external_G','replace_instances','replace_appearance_c05','replace_appearance_c1','replace_appearance_c2','replace_references','replace_all_c05','replace_all_c1','replace_all_c2']
SAFE=['source','replace_all_c05','replace_all_c1','replace_all_c2']
ABLATIONS=['replace_all_c05_no_relation','replace_all_c1_no_relation','replace_all_c2_no_relation']
NOASSOCIATION=['no_instance_association_c05','no_instance_association_c1','no_instance_association_c2']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def local(s):
 p=Path(s);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
 assert 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def sigmoid(z):return float(1/(1+np.exp(-np.clip(z,-30,30))))
def identity(r):return tuple(r[k] for k in ('imagepath','pointpath','metapath','caption','platform','scene'))

def masked_objects(masks,detector_boxes):
 H,W=masks.shape[1:];result=[]
 for j,m in enumerate(masks):
  yx=np.argwhere(m);ok=len(yx)>0
  result.append({'visible':ok,'domain':'oldmask','instance_index':j,'xy':(yx[:,::-1].mean(0)+.5)/[W,H] if ok else None,
   'rect':np.asarray(detector_boxes[j],float),
   'border':bool(ok and (m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any()))})
 return result

def global_known(graph,objects):
 valid={'GLOBAL_LEFT','LEFT_OF_CENTER','GLOBAL_RIGHT','RIGHT_OF_CENTER','GLOBAL_CENTER'}
 has=any(g.get('predicate') in valid for g in graph.get('global_relations',[]))
 return np.asarray([has and o['visible'] and not o['border'] for o in objects],bool)

def old_reference_hypotheses(records,W,H):
 result={}
 for entity,rr in records.items():
  hs=[]
  for r in rr:
   box=np.asarray(r['box_PRIVATE'],float);weight=sigmoid(r['full']['log_odds_yes_no'])*sigmoid(r['visual_evidence'])*float(r['detector_score'])
   normalized=box/np.array([W,H,W,H])
   hs.append({'xy_norm2':[float((normalized[0]+normalized[2])/2),float((normalized[1]+normalized[3])/2)],'width_norm':float(normalized[2]-normalized[0]),'rect':box.tolist(),'probability':weight,'uncertainty_mass_explicit':False})
  result[entity]=hs
 return result

def cpu_proposal_color(rgb,visible_xy,exclusive_oldmask,box,role_words):
 """No Qwen numerical input: original DINO box/conf plus sparse raw-safe RGB."""
 xy=visible_xy;box=np.asarray(box,float)
 keep=exclusive_oldmask&(xy[:,0]>=box[0])&(xy[:,0]<box[2])&(xy[:,1]>=box[1])&(xy[:,1]<box[3])
 px=xy[keep];nc=len(px);ns=len(xy)
 if nc<MINIMUM_COLOR_PIXELS or not role_words:return 0.,False,nc
 background=color_support(rgb[xy[:,1],xy[:,0]]);observed=color_support(rgb[px[:,1],px[:,0]]);values=[]
 for word in role_words:
  if word not in observed or observed[word] is None or background[word] is None:continue
  p=(nc*observed[word]+COLOR_BETA_PSEUDOCOUNT)/(nc+2*COLOR_BETA_PSEUDOCOUNT);b=(ns*background[word]+COLOR_BETA_PSEUDOCOUNT)/(ns+2*COLOR_BETA_PSEUDOCOUNT)
  values.append(float(np.clip(np.log(p/(1-p))-np.log(b/(1-b)),-COLOR_LOGODDS_CLIP,COLOR_LOGODDS_CLIP)))
 return (float(np.mean(values))*nc/(nc+MINIMUM_COLOR_PIXELS),True,nc) if values else (0.,False,nc)

def appearance_reference_hypotheses(dino,graph,attrs,rgb,visible_xy,masks,factor):
 # Only DINO proposal box/conf/stable-index fields enter here. The caller does
 # not pass old Qwen reference records; unsupported color keeps original conf.
 W,H=rgb.shape[1],rgb.shape[0];hits=masks[:,visible_xy[:,1],visible_xy[:,0]];exclusive=hits.sum(0)==1
 requests={r['entity']:r for r in attrs['entity_attribute_requests']};out={};diagnostics={}
 for ref in graph['references']:
  ent=ref['entity'];obj=dino.get('objects',{}).get(ent);hs=[];diag=[]
  if obj is not None:
   for j in obj['stable_indices'][:3]:
    box=np.asarray(obj['original']['boxes_PRIVATE'][j],float);conf=float(obj['original']['scores'][j]);assert 0<=conf<=1
    e,known,n=cpu_proposal_color(rgb,visible_xy,exclusive,box,requests.get(ent,{}).get('recognized_color_words',[]))
    probability=conf*sigmoid(factor*e) if known else conf
    normalized=box/np.array([W,H,W,H])
    hs.append({'xy_norm2':[float((normalized[0]+normalized[2])/2),float((normalized[1]+normalized[3])/2)],'width_norm':float(normalized[2]-normalized[0]),'rect':box.tolist(),'probability':probability,'uncertainty_mass_explicit':False})
    diag.append({'proposal_index_PRIVATE':int(j),'color_known':known,'color_e':e,'exclusive_visible_pixels':n,'factor':factor,'weight':probability,'detector_confidence':conf,'Qwen_value_used':False})
  out[ent]=hs;diagnostics[ent]=diag
 return out,diagnostics

def visual_map(folder,worklist):
 seal=read(folder/'SEALED.json');assert seal['state']=='R51_DIAGNOSTIC_NUMERIC_VISUAL_COMPLETE_NO_GT'
 items=read(worklist);out={}
 for i,w in enumerate(items):
  rel=f'items_PRIVATE/{i:07d}.json';p=folder/rel;assert sha(p)==seal['files'][rel];r=read(p);assert r['work_index']==i
  key=(w['id_PRIVATE'],w['kind'],w.get('reference_entity'),int(w['source_instance_id']))
  assert key not in out
  if r['status']=='COMPLETED':
   assert r['id_PRIVATE']==w['id_PRIVATE'] and r['source_instance_id']==w['source_instance_id'] and r['kind']==w['kind'] and r.get('reference_entity')==w.get('reference_entity')
   views=r['views'];assert len(views)==(2 if w['recipe']=='TARGET_DUAL_CROP_FULL_GRAY' else 1)
   for v in views:assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
   out[key]=(w,r,float(np.mean([v['visual_evidence'] for v in views])),float(np.mean([v['full']['log_odds_yes_no'] for v in views])))
  else:assert r['status'].startswith('UNKNOWN');out[key]=(w,r,None,None)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);a=ap.parse_args()
 plan=read(a.plan);auth=read(a.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_DIAGNOSTIC_HYBRID_ASSEMBLY_NO_GT' and auth['plan_sha256']==sha(a.plan)
 assert plan['n']==336 and plan['main_arm_names']==MAIN and plan['diagnostic_only'] is True
 assert plan['new_reference_weight_rule']=='source_probability_times_sigmoid_full_times_sigmoid_evidence_no_renormalization'
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(local(f))==h,('pin',f)
 for key in ('manifest','target_worklist','reference_worklist','dino_case_index','dino_registration'):assert plan[key] in plan['inputs_sha256']
 assert plan['programs_sha256'][Path(__file__).relative_to(ROOT).as_posix()]==sha(__file__)
 rows=read(local(plan['manifest']));assert len(rows)==336 and plan['manifest']=='research/round24/input/ALL336_PRIVATE.json'
 roots={k:local(plan[k]) for k in ('rawsafe_root','hybrid_root','safe_prediction_root','r50_prediction_root','target_visual_root','reference_visual_root','noassociation_root')}
 seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 for k,p in roots.items():assert plan['inputs_sha256'][(p/'SEALED.json').relative_to(ROOT).as_posix()]==sha(p/'SEALED.json')
 assert seals['safe_prediction_root']['state']=='R51_FINAL_SAFE_THREE_FACTOR_PREDICTIONS_NO_GT' and seals['safe_prediction_root']['external_vision_outputs_or_mask_derived_caches_read'] is False
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 target=visual_map(roots['target_visual_root'],local(plan['target_worklist']));refvis=visual_map(roots['reference_visual_root'],local(plan['reference_worklist']))
 dindex=read(local(plan['dino_case_index']));dreg=read(local(plan['dino_registration']));did={identity(r):r['id'] for r in dreg};assert len(did)==len(dreg)
 out=local(plan['output_root']);assert not out.exists() and out.is_relative_to(ROOT/'crossplatform/round51');out.mkdir(parents=True)
 for d in ('samples_PRIVATE','cases_PRIVATE'):(out/d).mkdir()
 assert plan['include_no_relation_ablations'] is True and plan['include_noassociation_ablations'] is True
 assert seals['noassociation_root']['state']=='R51_RAW_SAFE_NO_INSTANCE_ASSOCIATION_ABLATION_COMPLETE_NO_GT' and seals['noassociation_root']['n']==336
 arms=MAIN+ABLATIONS+NOASSOCIATION
 statuses=[];done=[];start=time.perf_counter()
 try:
  for row in rows:
   sid=row['id'];raw=load(bound('rawsafe_root','arrays_PRIVATE/'+sid+'.npz'));rc=read(bound('rawsafe_root','cases_PRIVATE/'+sid+'.json'))
   attrs=read(bound('rawsafe_root','attributes_PRIVATE/'+sid+'.json'));graph=attrs['text_graph_unchanged'];safe=load(bound('safe_prediction_root','samples_PRIVATE/'+sid+'.npz'))
   ctrl=load(bound('hybrid_root','controls_PRIVATE/'+sid+'.npz'));oldref=read(bound('hybrid_root','controls_PRIVATE/'+sid+'.json'))['old_references']
   maskattrs=read(bound('hybrid_root','oldmask_attributes_PRIVATE/'+sid+'.json'));old=load(bound('r50_prediction_root','samples_PRIVATE/'+sid+'.npz'))
   assert safe['arm_names'].tolist()==SAFE and exact(raw['source_boxes'],safe['source_boxes']) and exact(raw['source_scores'],safe['source_scores'])
   assert exact(raw['source_boxes'],old['source_boxes']) and exact(raw['source_scores'],old['source_scores'])
   assert exact(safe['scores'][0],old['scores'][0])
   s=raw['source_scores'];W,H=rc['image_size_wh'];newobjects=object_geometry(raw,(W,H));oldobjects=masked_objects(ctrl['masks_bool'],ctrl['boxes_xyxy'])
   refsold=old_reference_hypotheses(oldref,W,H);newpr,newprknown=raw_relations(graph,newobjects,refsold,(W,H))
   newe=np.zeros(len(newobjects));newknown=np.zeros(len(newobjects),bool)
   for j in range(len(newe)):
    key=(sid,'target',None,j);assert key in target
    _,record,e,_=target[key]
    if e is not None:newe[j]=e;newknown[j]=True
   Fm,Cm,Im=propagate_measured(s,raw['query_instance_conditional_weights'],newe,newpr,newknown,newprknown)
   pool=ctrl['wide_pool'];oldW=ctrl['wide_old_weights'];old_e=ctrl['image_e'][pool];oldpr=ctrl['raw_mask_PR'][pool]
   entry=dindex[did[identity(row)]];dp=local(entry['file']);assert sha(dp)==entry['sha256'];det=read(dp)
   assert sha(row['imagepath'])==rc['raw_bindings']['image_sha256']
   with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
   mask_e,maskknown=attr_evidence(maskattrs,'T');appearance=[];appearance_diag=[];appearance_info=[];appearance_components=[];appearance_pr=[];appearance_pk=[];appearance_hs=[]
   for factor in (.5,1.,2.):
    colored_refs,diag=appearance_reference_hypotheses(det,graph,maskattrs,rgb,raw['visible_pixels_xy'],ctrl['masks_bool'],factor)
    apr,ak=raw_relations(graph,oldobjects,colored_refs,(W,H));apr[:,0]=ctrl['raw_mask_PR'][:,0];ak[:,0]=global_known(graph,oldobjects)
    fq,cq,iq=propagate_measured(s,oldW,factor*mask_e[pool],apr[pool],maskknown[pool],ak[pool]);appearance.append(fq);appearance_diag.append(diag);appearance_info.append(iq);appearance_components.append(cq);appearance_pr.append(apr);appearance_pk.append(ak);appearance_hs.append(colored_refs)
   refsnew={};ref_status={};old_reference_request_gate=bool(len(ctrl['old_pool']))
   for ref in graph['references'] if old_reference_request_gate else []:
    ent=ref['entity'];hs=[];mass=0.;total=0.
    for key,(w,r,e,full) in refvis.items():
     if key[:3]!=(sid,'reference',ent):continue
     probability=float(w['source_probability']);assert 0<=probability<=1;total+=probability
     if e is None:continue
     j=key[3];rect=np.asarray(raw['instance_rect_xyxy'][j],float);xy=raw['instance_visible_centroid_uv'][j]/[W,H]
     weight=probability*sigmoid(full)*sigmoid(e);mass+=weight
     hs.append({'xy_norm2':xy.tolist(),'width_norm':float((rect[2]-rect[0])/W),'rect':rect.tolist(),'probability':weight,'uncertainty_mass_explicit':True,'instance_domain':'raw3d','instance_index':j})
    assert total<=1+1e-12 and len(hs)<=3;refsnew[ent]=hs;ref_status[ent]={'source_top3_mass':total,'retained_fresh_visual_mass':mass,'unknown_mass_at_least':1-mass,'renormalized':False}
   rpr,rknown=raw_relations(graph,oldobjects,refsnew,(W,H));rpr[:,0]=ctrl['raw_mask_PR'][:,0];rknown[:,0]=global_known(graph,oldobjects)
   Fr,Cr,Ir=propagate_measured(s,oldW,old_e,rpr[pool],np.ones(len(pool),bool),rknown[pool])
   externalindex=old['arm_names'].tolist().index('wideentry_oldW')
   fs=[safe['scores'][0],old['scores'][externalindex],Fm,*appearance,Fr,*[safe['scores'][i] for i in (1,2,3)]]
   fs.extend(safe['no_relation_scores'])
   na=load(bound('noassociation_root','samples_PRIVATE/'+sid+'.npz'));assert na['arm_names'].tolist()==NOASSOCIATION
   assert exact(na['source_boxes'],raw['source_boxes']) and exact(na['source_scores'],s)
   fs.extend(na['scores'])
   scores=np.stack(fs);q=scores.argmax(1).astype(np.int64);boxes=raw['source_boxes'][q]
   assert np.array_equal(q[0:1],safe['queries'][:1]) and q[1]==old['queries'][externalindex]
   assert np.array_equal(q[7:10],safe['queries'][1:]) and exact(boxes[7:10],safe['selected_original_boxes'][1:])
   assert np.array_equal(q[10:13],safe['no_relation_queries']) and exact(boxes[10:13],safe['no_relation_selected_original_boxes'])
   assert np.array_equal(q[13:16],na['queries']) and exact(boxes[13:16],na['selected_original_boxes'])
   np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.asarray(arms),scores=scores,queries=q,selected_original_boxes=boxes,source_boxes=raw['source_boxes'],source_scores=s,
    hybrid_arm_names=np.asarray(['replace_instances',*MAIN[3:6],'replace_references']),hybrid_components=np.stack([Cm,*appearance_components,Cr]),
    hybrid_cue_query_known=np.stack([x['query_known'] for x in [Im,*appearance_info,Ir]]),
    hybrid_cue_anchors=np.stack([x['anchors'] for x in [Im,*appearance_info,Ir]]),
    hybrid_cue_query_known_weight_mass=np.stack([x['query_known_weight_mass'] for x in [Im,*appearance_info,Ir]]),
    replace_instances_e=newe,replace_instances_e_known=newknown,replace_instances_raw_PR=newpr,replace_instances_raw_PR_known=newprknown,
    replace_appearance_raw_PR=np.stack(appearance_pr),replace_appearance_raw_PR_known=np.stack(appearance_pk),
    replace_references_raw_PR=rpr,replace_references_raw_PR_known=rknown)
   status={arm:{'unknown':False,'fallback':False,'no_parameter_update':True} for arm in arms}
   unavailable=bool(not Im['query_known'].any());status['replace_instances'].update(unknown=unavailable,fallback=unavailable)
   for arm,info in zip(MAIN[3:6],appearance_info):
    unavailable=bool(not info['query_known'].any());status[arm].update(unknown=unavailable,fallback=unavailable)
   unavailable=bool(not Ir['query_known'].any());status['replace_references'].update(unknown=unavailable,fallback=unavailable)
   safe_case=read(bound('safe_prediction_root','cases_PRIVATE/'+sid+'.json'))
   assert safe_case['external_vision_outputs_or_mask_derived_caches_read'] is False
   for arm in SAFE[1:]:status[arm]=safe_case['arm_statuses'][arm]
   for i,arm in enumerate(ABLATIONS):
    if arm in status:
     unavailable=bool(not safe['cue_query_known'][i,:2].any());status[arm]={'unknown':unavailable,'fallback':unavailable,'no_parameter_update':True}
   nacase=read(bound('noassociation_root','cases_PRIVATE/'+sid+'.json'))
   assert nacase['external_vision_outputs_or_mask_derived_caches_read'] is False
   for arm in NOASSOCIATION:status[arm]=nacase['arm_statuses'][arm]
   statuses.append({'id':sid,'arm_statuses':status});done.append(sid)
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'diagnostic_only':True,'allreplace_scores_copied_not_recalculated':True,'external_G_bitexact':True,'Qonly_Qwen_numeric_feedback_used':False,'appearance_ref_diagnostics_PRIVATE':appearance_diag,'appearance_reference_hypotheses_PRIVATE':appearance_hs,'new_reference_hypotheses_PRIVATE':refsnew,'legacy_reference_hypotheses_PRIVATE':refsold,'reference_mass':ref_status,'Donly_original_reference_request_gate':old_reference_request_gate,'Donly_reference_items_collected_but_not_consumed':sum(k[0]==sid for k in refvis) if not old_reference_request_gate else 0,'all_reference_collection_costs_retained':True,'new_target_unknown_count':int((~newknown).sum()),'new_reference_quality_renormalized':False,'source_original_boxes_fixed':True})
  dump(out/'READOUT_STATUS_PRIVATE.json',statuses)
  dump(out/'COSTS.json',{'new_model_forwards':0,'optimizer_steps':0,'seconds':time.perf_counter()-start,'fresh_target_numeric_seal_sha256':sha(roots['target_visual_root']/'SEALED.json'),'fresh_reference_numeric_seal_sha256':sha(roots['reference_visual_root']/'SEALED.json'),'joint_collection_not_independent_arm_deployment_wall':True})
  dump(out/'SEALED.json',{'state':'R51_DIAGNOSTIC_HYBRID_AND_SAFE_COPIED_READOUTS_NO_GT','n':len(done),'arm_names':arms,'main_arm_names':MAIN,'plan_sha256':sha(a.plan),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'no_GT_or_model_execution':True})
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry_or_GT_scoring':True});raise
if __name__=='__main__':main()
