"""Mapping support from blind 2D identity and exclusive visible LiDAR points.
No 3D IoU or GT geometry enters this identity-support classifier.
"""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
from observations27 import load
def main():
 verify(BASE/'blind_review_attempt001');keys={x['blind_case']:x for x in read(EXP.parent/'input/BLIND_KEY_PRIVATE.json')};lookup={x['id']:x for x in rows()};ident=read(EXP/'IDENTITY24_ROWS.json');out=[]
 for x in ident:
  key=keys[x['case']];r=lookup[key['id_PRIVATE']];a=x['visual_annotation'];pp=paths(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));w=load_npz(pp['prep']/'arrays_PRIVATE'/(r['id']+'.npz'))['weights'];pool=prep['mask_pool_PRIVATE'];e=read(ROOT/'crossplatform/round27/joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'))['targets_PRIVATE'];ev=np.array([e[str(m)]['crop_scene_joint']['log_odds_yes_no'] for m in pool]);z=feature(r);cue=w@ev;M=np.flatnonzero(w.sum(1)>0);maxset=M[abs(cue[M]-cue[M].max())<=1e-9];qcue=int(maxset[z['scores'][maxset].argmax()]);qm=int(M[(z['scores'][M]+2*cue[M]).argmax()]);qfinal=int((z['scores']+2*cue).argmax());q0=int(z['scores'].argmax());frame,scene,binding=load(r)
  if a['status']!='resolved':out.append({'case':x['case'],'platform':r['platform'],'image_ranking_status':x['image_ranking_status'],'status':'visual_unresolved'});continue
  targets={key['labels_to_mask_PRIVATE'][s] for s in a['target_labels']};P=scene['visible_points_model'];owner=scene['point_owner'];tp=np.isin(owner,list(targets));op=(owner>=0)&~tp
  nt=[];no=[];nu=[]
  for box in z['boxes']:
   inside=(abs(P-box[:3])<=box[3:]/2).all(1);nt.append(int((inside&tp).sum()));no.append(int((inside&op).sum()));nu.append(int(inside.sum()))
  nt=np.array(nt);no=np.array(no);nu=np.array(nu);known=nt+no;frac=nt/np.maximum(known,1);labels=np.where((known>=5)&(frac>=.8),'target_supported',np.where((known>=5)&(frac<=.2),'other_supported','insufficient_or_mixed'))
  detail={}
  for n,q in [('source',q0),('cue_max_source_tiebreak',qcue),('mapped_final',qm),('original_final',qfinal)]:
   detail[n]={'support_label':str(labels[q]),'target_points':int(nt[q]),'other_foreground_points':int(no[q]),'all_box_visible_points':int(nu[q]),'target_share_of_foreground_points':float(frac[q]),'source_score':float(z['scores'][q]),'mapped_cue':float(cue[q]),'mapped':bool(w[q].sum()>0),'target_mask_weight':float(w[q,[pool.index(m) for m in targets]].sum())}
  image=x['image_ranking_status'];reason='image_wrong' if image=='incorrect' else 'image_tied' if image=='tie_target_other' else 'unresolved'
  if image=='correct':
   if labels[qcue]=='other_supported':reason='visual_correct_cue_maps_to_other_support'
   elif labels[qcue]=='target_supported' and labels[qfinal]=='other_supported':reason='visual_and_mapped_cue_support_target_fusion_selects_other'
   elif labels[qcue]=='target_supported' and not w[qfinal].sum() and labels[qm]=='target_supported':reason='unmapped_zero_score_escape_after_correct_image'
   elif labels[qfinal]=='target_supported':reason='final_supported_by_visual_target_points'
   else:reason='mapping_or_final_support_uncertain'
  out.append({'case':x['case'],'platform':r['platform'],'alias':r['alias'],'image_ranking_status':image,'visual_confidence':a['confidence'],'visual_margin':x['visual_target_margin'],'max_cue_query_count':len(maxset),'details':detail,'chain_support_diagnosis':reason,'center_proxy_final':x['final_center_status'],'original_final_IoU_evaluation_only':x['task_iou']['joint'],'actual_correct_image_does_not_certify_calibration':True})
 counts=collections.Counter(x.get('chain_support_diagnosis',x['status'] if 'status' in x else 'NA') for x in out);dump(EXP/'POINT_SUPPORT24_ROWS.json',out);dump(EXP/'POINT_SUPPORT24_SUMMARY.json',{'n':24,'classifier':'>=5exclusive-mask foreground LiDAR points;target fraction>=.8,other<=.2;elseunknown','no_GT_geometry_or_IoU_in_support_labels':True,'independent_caption_visual_identity_but_shared_camera_mask_geometry':True,'not_complete_world_identity_gold':True,'counts':dict(counts)});print('SUPPORT28',dict(counts))
 for x in out:print(x['case'],x.get('image_ranking_status'),x.get('chain_support_diagnosis'),x.get('visual_margin'),{k:v['support_label'] for k,v in x.get('details',{}).items()})
if __name__=='__main__':main()
