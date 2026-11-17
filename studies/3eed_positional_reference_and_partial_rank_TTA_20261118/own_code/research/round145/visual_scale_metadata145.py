"""Frozen-support spatial resolution metadata only; no GT, images, model or taskscore."""
from common145 import *
import numpy as np,collections

def span(q,own,idx,weight):
 valid=(idx>=0)&(weight>0);ok=own[q].astype(bool)&valid.any(1);ids=idx.clip(0);w=weight*valid;w=w/w.sum(1,keepdims=True).clip(1e-12);uv=np.stack([((ids%48+.5)*w).sum(1),((ids//48+.5)*w).sum(1)],1);xy=uv[ok];n=len(xy)
 return {'support_points':n,'min_side_cells':float(np.ptp(xy,axis=0).min()) if n>=2 else None,'max_side_cells':float(np.ptp(xy,axis=0).max()) if n>=2 else None}
def main():
 check()
 for p,h in rd(H/'experiment/VISUAL_SCALE_METADATA_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p
 rr=[]
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  en=e['entry'];src,sid=en['source'],en['row']['id'];a=np.load(B/'source'/src/(sid+'.npz'));rt=np.load(R/en['files']['route']['path']);ga=np.load(R/en['files']['geometry']['path']);q=int(a['pA'].argmax());rr.append({'cohort':'source_own','source':src,'target':src,'id_PRIVATE':sid,'view':'A','has_B':bool(a['valid_B'].any()),**span(q,rt['own_support'],ga['p2i'],ga['p2i_weight'])})
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();ga=np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz'));rt=np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz'));a=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));q=int(ps['norm_Aimage'].argmax());rr.append({'cohort':'target_exposed','source':src,'target':tgt,'id_PRIVATE':sid,'view':'A','has_B':e['has_B'],**span(q,rt['own_support'],ga['p2i'],ga['p2i_weight'])})
 out=[]
 for cohort,src,tgt in sorted({(r['cohort'],r['source'],r['target']) for r in rr}):
  aa=[r for r in rr if (r['cohort'],r['source'],r['target'])==(cohort,src,tgt)];ss=[r for r in aa if r['min_side_cells'] is not None];v=np.array([r['min_side_cells'] for r in ss]);out.append({'cohort':cohort,'source':src,'target':tgt,'n':len(aa),'at_least_two_projected_support_points':len(ss),'median_min_side_cells':float(np.median(v)) if len(v) else None,'min_side_le_one_cell_fraction':float(np.mean(v<=1)) if len(v) else None,'min_side_le_two_cells_fraction':float(np.mean(v<=2)) if len(v) else None})
 dump(H/'VISUAL_SCALE_ROWS_PRIVATE.json',rr);x={'status':'FROZEN_VISIBLE_SUPPORT_METADATA_ONLY','summary':out,'image_input':[384,256],'CNN_grid':[48,32],'CNN_stride':8,'CNN_receptive_field_pixels':15,'support_span_not_true_object_extent':True,'point_sparse_support_can_underestimate_true_extent':True,'winner_can_be_wrong_no_identity_certification':True,'GT_or_new_model_images_or_taskscore_read':False,'R145_predictor_unchanged':True,'no_resolution_method_registered_or_run':True,'FINAL_access':False};dump(H/'VISUAL_SCALE_METADATA_ONLY.json',x);print(x,flush=True)
if __name__=='__main__':main()
