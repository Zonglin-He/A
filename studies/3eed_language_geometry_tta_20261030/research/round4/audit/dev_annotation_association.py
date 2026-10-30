"""Post-decomposition DEV-only annotation association; no exact instance-ID claim.
Reads already-authorized labels after source seal. Never modifies sealed decomposition.
"""
import pathlib,json,numpy as np,hashlib,math
from candidate_decomposition import metric,rot,sha,read,S,R

def transform(box,s):
 b=np.array(box[:7],float)
 if s['platform']=='drone':b[2]+=1.8
 if s['platform']=='waymo':
  a=math.radians([0,-45,45,-90,90][s['waymo_view_suffix']]);b[:3]=rot(a)@b[:3];b[6]+=a
 return b

def main():
 primary=R/'dev_decomposition_v1';seal=read(primary/'SEALED.json')
 for name,h in seal['files'].items():assert sha(primary/name)==h
 d=read(primary/'RESULTS_PRIVATE.json');by={x['id']:x for x in d['rows']};source=S/'dev_source';ss=read(source/'sealed.json');assert sha(source/'sealed.json')=='b8a1553ad0c9c9811176928f90a859d3eeb3a2d61f84ae89e2de5342beff7842'
 index=S/'dev_scoring_index_PRIVATE.json';assert sha(index)==read(S/'INPUT_RECEIPT.json')['cohorts']['dev']['scoring_index_sha256'];rows=[]
 for s in read(index):
  sid=s['id'];name=sid+'_v0_full.npz';assert sha(source/name)==ss['files'][name]
  m=read(s['meta_path']);obj=m['ground_info'][s['object_index']];target=transform(obj['bbox_3d'],s);others=obj.get('others',[]);targetclass=str(obj.get('class','')).strip().lower()
  with np.load(source/name,allow_pickle=False) as z:
   q=int(np.lexsort((np.arange(256),-z['semantic_scores']))[0]);box=z['boxes_pca'][q].astype(float)
  target_iou=metric(box,target);assert abs(target_iou-by[sid]['geometries']['fixedPCA']['top1_iou'])<1e-12
  matches=[]
  for j,o in enumerate(others):
   b=transform(o['bbox_3d_other'],s);sameclass=str(o.get('class_other','')).strip().lower()==targetclass;duplicate=metric(b,target)>1-1e-8
   matches.append({'other_annotation_index':j,'q0_other_iou':metric(box,b),'same_class_label':sameclass,'spatial_duplicate_of_target_annotation':duplicate})
  eligible=[x for x in matches if not x['spatial_duplicate_of_target_annotation']];best=max(eligible,key=lambda x:x['q0_other_iou']) if eligible else None
  rec={'id':sid,'platform':s['platform'],'scene':s['scene'],'q0':q,'target_iou':target_iou,'others_n':len(others),'declared_others_num':obj.get('others_num'),'others_num_matches_list':obj.get('others_num')==len(others),'target_object_id_available':any(k in obj for k in ['id','object_id','track_id','instance_id']),'matches':matches,'q0_best_nontarget_annotation':best,'PCA_rankrecoverable_at05':by[sid]['geometries']['fixedPCA']['decomposition']['0.5']['rankrecoverable_top1fails_any256passes'],'q0_matches_other_above05_while_target_not_above05':bool(best and best['q0_other_iou']>.5 and target_iou<=.5),'clear_other_above05_target_not_above025':bool(best and best['q0_other_iou']>.5 and target_iou<=.25)};rows.append(rec)
 summary=[]
 for p in ['waymo','drone','quad']:
  rr=[r for r in rows if r['platform']==p];recover=[r for r in rr if r['PCA_rankrecoverable_at05']];summary.append({'platform':p,'n':len(rr),'q0_other_gt05_target_le05':sum(r['q0_matches_other_above05_while_target_not_above05'] for r in rr),'clear_other_gt05_target_le025':sum(r['clear_other_above05_target_not_above025'] for r in rr),'rankrecoverable05_n':len(recover),'rankrecoverable_with_other_gt05':sum(r['q0_matches_other_above05_while_target_not_above05'] for r in recover),'rankrecoverable_with_sameclass_other_gt05':sum(r['q0_matches_other_above05_while_target_not_above05'] and r['q0_best_nontarget_annotation']['same_class_label'] for r in recover),'rankrecoverable_cases':[{'id':r['id'],'target_iou':r['target_iou'],'best_other':r['q0_best_nontarget_annotation']} for r in recover]})
 out={'scope':'DEVonly post-decomposition descriptive annotation association, no new method or gate selected','source_seal_sha256':sha(source/'sealed.json'),'decomposition_result_sha256':sha(primary/'RESULTS_PRIVATE.json'),'schema':'ground_info entries include class/bbox and others[{bbox_3d_other,class_other}]; no stableinstanceID; not proven exhaustive catalog','summaries':summary,'rows':rows,'limitations':['Spatial match to annotated non-target is evidence of distractor affinity, not exact physical instance identity','No classifier, labelbased candidate selection, adaptation or confirmation access','No original scored files modified'],'no_model_or_GPU_calls':True}
 path=R/'DEV_ANNOTATION_ASSOCIATION_PRIVATE.json';assert not path.exists();path.write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
