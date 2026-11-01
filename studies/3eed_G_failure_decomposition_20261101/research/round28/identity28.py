"""Unblind sealed visual labels; annotation-geometry identity PROXIES are not IoU labels."""
import sys,numpy as np,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round28'),str(ROOT/'research/round5/audit')]
from common28 import *
from observations27 import load
from independent_core_v2 import transform_gt,candidate_ious
from camera_projection import project_model_points
from association import box_corners,rect_iou
def transformed_other(b,p):
 a=np.asarray(b,float).copy();assert a.shape==(7,)
 if p=='drone':a[2]+=1.8
 return a.astype(np.float32)
def inside(points,boxes):
 p=np.asarray(points,float);bb=np.asarray(boxes,float);d=p[:,None,:]-bb[None,:,:3];c=np.cos(bb[:,6]);s=np.sin(bb[:,6]);local=np.stack([d[:,:,0]*c+d[:,:,1]*s,-d[:,:,0]*s+d[:,:,1]*c,d[:,:,2]],-1);return (abs(local)<=bb[None,:,3:6]/2+1e-7).all(-1)
def center_labels(points,boxes):
 a=inside(points,boxes);t=a[:,0];o=a[:,1:].any(1);return np.where(t&~o,'target',np.where(o&~t,'other',np.where(t&o,'ambiguous','unassigned')))
def setstatus(ids,labels):
 ss={str(labels[q]) for q in ids}
 return next(iter(ss)) if len(ss)==1 else 'mixed_'+','.join(sorted(ss))
def main():
 verify(BASE/'blind_review_attempt001');ann={x['case']:x for x in read(BASE/'blind_review_attempt001/ANNOTATIONS.json')};keys=read(EXP.parent/'input/BLIND_KEY_PRIVATE.json');lookup={x['id']:x for x in rows()};priorrows={x['alias']:x for x in read(ROOT/'research/round27/experiment/ALL528_ROWS_PRIVATE.json')};index={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};records=[]
 for key in keys:
  r=lookup[key['id_PRIVATE']];a=ann[key['blind_case']];pp=paths(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));w=load_npz(pp['prep']/'arrays_PRIVATE'/(r['id']+'.npz'))['weights'];pool=prep['mask_pool_PRIVATE'];ev=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'))['targets_PRIVATE'];es={label:ev[str(m)]['crop_scene_joint']['log_odds_yes_no'] for label,m in key['labels_to_mask_PRIVATE'].items()};em=np.array([ev[str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in pool]);best=max(es.values());top=[l for l,v in es.items() if abs(v-best)<=1e-7];targets=a['target_labels'];imgstatus='unresolved_visual'
  if a['status']=='resolved':imgstatus='correct' if set(top)<=set(targets) else 'incorrect' if not set(top)&set(targets) else 'tie_target_other'
  # Only now read GT scene geometry, after identity review is immutable.
  ix=index[r['id']];assert sha(ix['meta_path'])==ix['metadata_container_sha256'];meta=read(ix['meta_path']);obj=meta['ground_info'][ix['object_index']];target=transform_gt(obj['bbox_3d'],r['platform']);boxes=[target];seen={tuple(target.tolist())};duplicate=0
  rawothers=[(v['bbox_3d'],False) for j,v in enumerate(meta['ground_info']) if j!=ix['object_index']]+[(v['bbox_3d_other'],True) for v in obj['others']]
  for raw,isother in rawothers:
   b=transformed_other(raw,r['platform']) if isother else transform_gt(raw,r['platform']);t=tuple(b.tolist())
   if t in seen:duplicate+=1;continue
   seen.add(t);boxes.append(b)
  bb=np.stack(boxes);z=feature(r);qlabels=center_labels(z['boxes'][:,:3],bb);frame,scene,binding=load(r);assert binding==prep['bindings'];cue=w@em;mapped=w.sum(1)>0;maxcue=cue[mapped].max();qm=np.flatnonzero(mapped&(abs(cue-maxcue)<=1e-9));margin=None
  if targets:
   pos=[es[t] for t in targets];neg=[v for l,v in es.items() if l not in targets];margin=max(pos)-max(neg) if neg else None
  maskstats={}
  for label,m in key['labels_to_mask_PRIVATE'].items():
   pts=scene['objects'][m]['points_model'];membership=inside(pts,bb);n=len(pts);tfrac=float(membership[:,0].mean()) if n else None;ofrac=float(membership[:,1:].any(1).mean()) if n else None
   physical='target_supported' if n>=5 and tfrac>=.8 and ofrac<=.1 else 'other_supported' if n>=5 and ofrac>=.8 and tfrac<=.1 else 'mixed_or_unsupported'
   column=pool.index(m);strong=np.flatnonzero(w[:,column]>=.8);maskstats[label]={'n_exclusive_points':n,'target_point_fraction':tfrac,'other_point_fraction':ofrac,'physical_support':physical,'query_weight_ge08_n':len(strong),'query_center_labels':dict(collections.Counter(qlabels[strong].tolist()))}
  q=priorrows[r['alias']]['q']['crop_scene_joint'];q0=priorrows[r['alias']]['q']['source'];qold=priorrows[r['alias']]['q']['legacy_complex'];qids=np.array([q0,qold,q]);bs=z['boxes'][qids].copy();bs[:,3:]=np.maximum(bs[:,3:],1e-6);taskvals=candidate_ious(bs,target)[0];maxstatus=setstatus(qm,qlabels);finalstatus=str(qlabels[q]);sourcebesttarget=np.flatnonzero(qlabels=='target')
  reason='unresolved'
  if imgstatus=='incorrect':reason='visual_ranking_error'
  elif imgstatus=='tie_target_other':reason='visual_preference_tied'
  elif imgstatus=='correct':
   if not len(sourcebesttarget):reason='no_center_resolved_target_proposal'
   elif maxstatus=='other':reason='mapping_or_aggregation_to_other'
   elif maxstatus=='target' and finalstatus=='other':reason='source_fusion_overrides_target_cue'
   elif maxstatus=='target' and finalstatus=='target':reason='all_three_center_proxy_success'
   elif maxstatus=='target' and finalstatus in ['unassigned','ambiguous']:reason='fusion_selected_geometry_unresolved'
   else:reason='mapping_top_set_geometry_unresolved'
  # Exact margin witness against highest F among center-target candidates; no new inference selection.
  witness=None
  if len(sourcebesttarget):
   f=z['scores']+2*cue;qt=int(sourcebesttarget[f[sourcebesttarget].argmax()]);witness={'selected_source':float(z['scores'][q]),'target_source':float(z['scores'][qt]),'selected_cue':float(cue[q]),'target_cue':float(cue[qt]),'F_selected_minus_target':float(f[q]-f[qt]),'source_margin':float(z['scores'][q]-z['scores'][qt]),'weighted_cue_margin':float(2*(cue[q]-cue[qt])),'target_candidate_chosen_offline_for_diagnostic_only':True}
  # Projection consistency is a dependent annotation check, NOT independent visual truth.
  corners=box_corners(target[:6]);dy=corners-target[:3];c,s=np.cos(target[6]),np.sin(target[6]);R=np.array([[c,-s,0],[s,c,0],[0,0,1]]);corners=dy@R.T+target[:3];pr=project_model_points(corners,scene['camera_spec'],tuple(scene['image_size_wh']));projected=np.r_[pr['uv'].min(0),pr['uv'].max(0)];pi=rect_iou(projected,obj['bbox_2d_proj']) if np.isfinite(projected).all() and (pr['depth']>.1).all() else None
  records.append({'case':key['blind_case'],'platform':r['platform'],'alias':r['alias'],'stage':r['stage'],'visual_annotation':a,'image_scores':es,'image_top_labels':top,'image_ranking_status':imgstatus,'visual_target_margin':margin,'max_mapped_cue_query_n':len(qm),'max_mapped_cue_center_status':maxstatus,'final_center_status':finalstatus,'source_center_status':str(qlabels[q0]),'legacy_center_status':str(qlabels[qold]),'center_resolved_target_candidate_n':len(sourcebesttarget),'annotated_distinct_boxes':len(bb),'exact_duplicate_annotation_boxes_removed':duplicate,'mask_point_support':maskstats,'chain_diagnosis':reason,'margin_witness':witness,'task_iou':dict(zip(['source','legacy','joint'],map(float,taskvals))),'stored2D_vs_projected3D_IoU_dependent_diagnostic':pi,'no_literal_instanceIDs_or_complete_scene_inventory':True})
 dump(EXP/'IDENTITY24_ROWS.json',records);counts=collections.Counter(r['chain_diagnosis'] for r in records);cross={}
 for p in ['drone','quad']:
  rs=[r for r in records if r['platform']==p];cross[p]={'n':len(rs),'visual_resolved':sum(r['visual_annotation']['status']=='resolved' for r in rs),'image_ranking':dict(collections.Counter(r['image_ranking_status'] for r in rs)),'chain':dict(collections.Counter(r['chain_diagnosis'] for r in rs))}
 dump(EXP/'IDENTITY24_SUMMARY.json',{'n':24,'one_assistant_visual_annotator_not_expert_gold':True,'blind_review_seal_sha256':sha(BASE/'blind_review_attempt001/SEALED.json'),'geometry_identity_is_center_containment_proxy_not_IoU_or_literalIDs':True,'all_uncertain_and_counterexamples_kept':True,'chain_counts':dict(counts),'by_platform':cross,'no_additive_population_attribution':True});print('IDENTITY28',dict(counts),cross)
if __name__=='__main__':main()
