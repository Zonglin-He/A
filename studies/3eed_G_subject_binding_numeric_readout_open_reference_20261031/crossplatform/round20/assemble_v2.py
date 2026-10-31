"""Frozen-box local evidence composition. No GT, optimizer or string parser."""
import re
import numpy as np
from common_v2 import *
from observations import load
from detect import rect_iou
def box_support(detections,obj):
 a=detections['original'];values=[float(a['scores'][j])*rect_iou(a['boxes_PRIVATE'][j],obj['detector_box_xyxy']) for j in detections['stable_indices']];return max(values,default=0.)
def interval(obj):
 c=np.asarray(obj['mask_centroid_norm2'],float);box=np.asarray(obj['mask_bbox_norm4'],float);u=.1*max(box[2]-box[0],0.);return float(c[0]-u),float(c[0]+u)
def relation_state(rel,target,anchors):
 if rel['frame']!='camera_image':return {'A':None,'status':'UNKNOWN','reason':'frame_unverified'}
 if target['touches_image_border']:return {'status':'UNKNOWN','reason':'target_truncated_visible_position_not_full_position'}
 xlo,xhi=interval(target);pred=rel['predicate'];margin=.025
 if rel['reference'] is None:
  if pred in ('GLOBAL_LEFT','GLOBAL_RIGHT','GLOBAL_CENTER'):
   lo,hi={'GLOBAL_LEFT':(0.,.4),'GLOBAL_CENTER':(.3,.7),'GLOBAL_RIGHT':(.6,1.)}[pred];s='SUPPORT' if xlo>=lo and xhi<=hi else 'CONTRADICT' if xhi<lo-margin or xlo>hi+margin else 'UNKNOWN'
  elif pred in ('LEFT_OF_CENTER','RIGHT_OF_CENTER'):
   s=('SUPPORT' if xhi+.025<.5 else 'CONTRADICT' if xlo>.525 else 'UNKNOWN') if pred=='LEFT_OF_CENTER' else ('SUPPORT' if xlo>.525 else 'CONTRADICT' if xhi+.025<.5 else 'UNKNOWN')
  else:s='UNKNOWN'
  return {'status':s,'reason':'literal_image_position_interval'}
 if not anchors:return {'status':'UNKNOWN','reason':'reference_not_stably_localized'}
 states=[]
 for bb in anchors:
  cx=(bb[0]+bb[2])/2;u=.1*(bb[2]-bb[0]);rlo,rhi=cx-u,cx+u;left,right=((rlo,rhi),(xlo,xhi)) if rel['subject']!='T' else ((xlo,xhi),(rlo,rhi))
  if pred=='RIGHT_OF':left,right=right,left
  if pred not in ('LEFT_OF','RIGHT_OF'):states.append('UNKNOWN');continue
  states.append('SUPPORT' if left[1]+margin<right[0] else 'CONTRADICT' if left[0]>right[1]+margin else 'UNKNOWN')
 return {'status':states[0] if len(set(states))==1 else 'UNKNOWN','reason':'all_plausible_reference_boxes_same_direction' if len(set(states))==1 else 'reference_identity_ambiguous','anchor_count':len(anchors),'anchor_states':states}
def main():
 spec=locked('ASSEMBLE_v2_LOCK.json');cfg=read(EXP/'G_SPEC_v2_LOCK.json');verify(ROOT/'crossplatform/round20/dino144_attempt001');verify(ROOT/'crossplatform/round20/visual144_attempt001');verify(ROOT/'crossplatform/round20/dinoZ64_attempt001');verify(ROOT/'crossplatform/round20/visualZ64_attempt001');out=ROOT/'crossplatform/round20/direct208_attempt002';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();counts={}
 for r in read(EXP.parent/'input/ALL208_PRIVATE.json'):
  z=feature(r);q0=int(z['scores'].argmax());prep=read(ROOT/'crossplatform/round20'/('pairsZ64_attempt001' if r['stage']=='Z' else 'pairs144_attempt001')/'cases_PRIVATE'/(r['id']+'.json'));g=read(ROOT/'crossplatform/round20'/('textZ64_attempt001' if r['stage']=='Z' else 'text144_attempt001')/'cases_PRIVATE'/(r['id']+'.json'));d=read(ROOT/'crossplatform/round20'/('dinoZ64_attempt001' if r['stage']=='Z' else 'dino144_attempt001')/'cases_PRIVATE'/(r['id']+'.json'));v=read(ROOT/'crossplatform/round20'/('visualZ64_attempt001' if r['stage']=='Z' else 'visual144_attempt001')/'cases_PRIVATE'/(r['id']+'.json'));results=[];selected={a:q0 for a in ('source','direct_fullG','direct_no_subject_attributes','direct_no_reference','unconditional_visual_pair','direct_legacy_attribute_binding','direct_detector_veto')}
  if prep['pairs']:
   _,scene,binding=load(r);W,H=scene['image_size_wh'];anchors={}
   for ref in g['references']:
    det=d['objects'].get(ref['entity']);anchors[ref['entity']]=[np.asarray(det['original']['boxes_PRIVATE'][j])/np.array([W,H,W,H]) for j in det['stable_indices']] if det else []
   for pair in prep['pairs']:
    a,b=pair['mask_indices_PRIVATE'];A=scene['objects'][a];B=scene['objects'][b];va=v['masks_PRIVATE'].get(str(a));vb=v['masks_PRIVATE'].get(str(b))
    if va is None or vb is None:continue
    full_delta=[y['visual_full_evidence']-x['visual_full_evidence'] for x,y in zip(va,vb)];attr_delta=[y['attribute_increment']-x['attribute_increment'] for x,y in zip(va,vb)];category_delta=[y['category']['log_odds_yes_no']-y['gray_category']['log_odds_yes_no']-x['category']['log_odds_yes_no']+x['gray_category']['log_odds_yes_no'] for x,y in zip(va,vb)];fullD=[box_support(d['objects']['subject'],obj) for obj in (A,B)];catD=[box_support(d['objects']['category'],obj) for obj in (A,B)];rr=[]
    for rel in g['relations']:
     eid=rel['subject'] if rel['subject']!='T' else rel['reference'];refboxes=anchors.get(eid,[]);ea=relation_state(rel,A,refboxes);eb=relation_state(rel,B,refboxes);rr.append({'origin':rel['origin'],'predicate':rel['predicate'],'frame':rel['frame'],'is_reference_relation':rel['reference'] is not None,'A':ea,'B':eb})
    veto=any(x['B']['status']=='CONTRADICT' for x in rr);discriminating=any(x['A']['status']=='CONTRADICT' and x['B']['status']=='SUPPORT' for x in rr);discriminating_global=any(not x['is_reference_relation'] and x['A']['status']=='CONTRADICT' and x['B']['status']=='SUPPORT' for x in rr);globalveto=any(not x['is_reference_relation'] and x['B']['status']=='CONTRADICT' for x in rr)
    descriptor=g['attribute_binding']=='explicit_subject_fields' and bool(g['own_attributes_literal']) or len(re.sub(r'\b(?:a|an|the|this|is|it|'+re.escape(g['category'])+r')\b','',g['subject_query'],flags=re.I).strip())>0
    visual=bool(min(full_delta)>=cfg['min_Qwen_pair_margin'] and min(attr_delta)>=cfg['min_attribute_increment_margin']);corroborated=bool(fullD[1]>=cfg['DINO_subject_support_min'] and fullD[1]-fullD[0]>=cfg['DINO_pair_score_advantage']);category=bool(min(category_delta)>=cfg['min_Qwen_pair_margin']);no_attrs=(category or discriminating and min(category_delta)>=0) and not veto
    strict_full=(visual and corroborated or discriminating and fullD[1]>=cfg['DINO_subject_support_min'] and min(full_delta)>=0) and not veto;full=(visual or discriminating and min(full_delta)>=0) and not veto;no_ref=(visual or discriminating_global and min(full_delta)>=0) and not globalveto
    results.append({'pair_alias':pair['pair_alias'],'qA_PRIVATE':pair['qA_PRIVATE'],'qB_PRIVATE':pair['qB_PRIVATE'],'full_delta':full_delta,'attribute_delta':attr_delta,'category_delta':category_delta,'DINO_full_support':fullD,'DINO_category_support':catD,'relations':rr,'reference_box_counts':{k:len(x) for k,x in anchors.items()},'visual_preference':visual,'corroborated':corroborated,'has_own_descriptor':bool(descriptor),'relation_veto':veto,'relation_discriminating':discriminating,'accepted':bool(full),'strict_detector_accepted':bool(strict_full),'no_attributes_accepted':bool(no_attrs),'no_reference_accepted':bool(no_ref),'unconditional_accepted':min(full_delta)>0,'evidence_strength':float(min(full_delta))})
   for arm,flag in [('direct_fullG','accepted'),('direct_detector_veto','strict_detector_accepted'),('direct_no_subject_attributes','no_attributes_accepted'),('direct_no_reference','no_reference_accepted'),('unconditional_visual_pair','unconditional_accepted')]:
    okay=[p for p in results if p[flag]]
    if okay:selected[arm]=max(okay,key=lambda p:(p['evidence_strength'],-p['qB_PRIVATE']))['qB_PRIVATE']
  selected['direct_legacy_attribute_binding']=selected['direct_no_subject_attributes'] if g.get('attribute_binding')=='explicit_subject_fields' else selected['direct_fullG']
  names=list(selected);indices=np.array(list(selected.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),query_indices=indices,arm_boxes_raw6=z['boxes'][indices]);payload={'alias':r['alias'],'id_PRIVATE':r['id'],'stage':r['stage'],'platform':r['platform'],'preparation_reason':prep['reason'],'pair_diagnostics':results,'fullG_intervenes':selected['direct_fullG']!=q0,'actual_subject_attribute_question_involved':bool(v['masks_PRIVATE']) and v['own_attributes_in_actual_scored_question'],'no_format_rejection':True,'parameters_updated':False};dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload);key=(r['stage'] if r['stage'] in ('W','Z') else 'D')+'_'+r['platform'];counts.setdefault(key,{'n':0,'accepted':0});counts[key]['n']+=1;counts[key]['accepted']+=selected['direct_fullG']!=q0
 dump(out/'COUNTS.json',counts);seal(out,'R20_direct_G_frozen_source_boxes_sealed_before_current_GT',n=208,counts=counts,parameter_updates=0);print('DIRECT144_SEALED',counts)
if __name__=='__main__':main()
