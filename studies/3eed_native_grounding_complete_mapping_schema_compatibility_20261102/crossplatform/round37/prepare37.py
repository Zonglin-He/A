"""Exact R17 geometry matrices; remove only R29 AB dictionary truncation."""
import sys,time,numpy as np
from common37 import *
sys.path[:0]=[str(ROOT/'crossplatform/round27'),str(ROOT/'crossplatform/round17')]
from observations27 import load
from pair_geometry import prepare

def main():
 check_plan();out=BASE/'prepared208_attempt001';out.mkdir(exist_ok=False)
 for n in ['cases_PRIVATE','arrays_PRIVATE','panels_PRIVATE']:(out/n).mkdir()
 start=time.perf_counter();counts=[]
 for j,r in enumerate(rows()):
  frame,scene,binding=load(r);z=feature(r);raw=prepare(scene,z['boxes'],r['id']);S=np.asarray(raw['point_support_count']);C=np.asarray(raw['point_support_fraction']);I=np.asarray(raw['projection_iou']);M=len(scene['objects']);assert S.shape==C.shape==I.shape==(256,M);valid=(S>=3)&(C>=.15)&(I>=.1)&(z['boxes'][:,3:]>0).all(1)[:,None];unique=valid.sum(1)==1;full={str(m):np.flatnonzero(valid[:,m]&unique).tolist() for m in range(M)};old=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE']
  for m,g in old.items():assert sorted(g)==full[m],('old_full_geometry_mismatch',r['alias'],m,g,full[m])
  ins=frame['instances'];assert M==len(ins['masks_bool']) and all(o['array_index']==m and np.array_equal(o['mask_bool'],ins['masks_bool'][m]) for m,o in enumerate(scene['objects']));cc=candidates_for(r,frame);marker_panel(frame['rgb'],cc).save(out/'panels_PRIVATE'/(r['id']+'.png'));W,H=scene['image_size_wh'];np.savez_compressed(out/'arrays_PRIVATE'/(r['id']+'.npz'),masks_bool=ins['masks_bool'],mask_boxes=ins['boxes_xyxy'],source_boxes=z['boxes'],source_scores=z['scores'],support_count=S,support_coverage=C,projection_iou=I,valid_query_mask=valid,unique_query=unique)
  case={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'image_size_wh':[W,H],'image_sha256':sha(r['imagepath']),'original_RGB_array_sha256':array_sha(frame['rgb']),'panel_sha256':sha(out/'panels_PRIVATE'/(r['id']+'.png')),'bindings':binding,'candidates':cc,'full_groups':full,'old_subset_groups':old,'representatives_full':{m:max(g,key=lambda i:(float(z['scores'][i]),-i)) for m,g in full.items() if g},'ambiguous_queries':np.flatnonzero(valid.sum(1)>1).tolist(),'depth_trim_counts':raw['point_count_after_depth_trim'],'source_q0':int(z['scores'].argmax()),'no_GT_or_new_masks':True};dump(out/'cases_PRIVATE'/(r['id']+'.json'),case);counts.append({'alias':r['alias'],'platform':r['platform'],'independent_masks':M,'full_nonempty_groups':sum(bool(g) for g in full.values()),'old_subset_groups':len(old),'mapped_unique_queries':int(unique.sum()),'ambiguous_queries':int((valid.sum(1)>1).sum()),'old_dictionary_exactly_reproduced':True})
  if (j+1)%16==0:print('PREPARE37',j+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'COUNTS.json',counts);seal(out,'R37_A208_complete_mask_to_unique_query_groups_before_VLM_or_GT',n=208,old_R29_dictionary_exact=True,source_and_candidate_boxes_unchanged=True,full_geometry='R17 prepare exact matrices with depth-IQRtrim and +0.1m box support',seconds=time.perf_counter()-start,plan_sha256=sha(EXP/'PLAN_LOCK.json'));print('PREPARE37_SEALED')
if __name__=='__main__':main()
