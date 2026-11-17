"""Source/target GT-free observability prerequisite, NOT a changed predictor."""
from common142 import *
import numpy as np,collections

def scope(z,assignment,validpoint,visible):
 unique,inverse=np.unique(z['xyz'][0],axis=0,return_inverse=True);vp=np.zeros(len(unique),bool);vi=np.zeros(len(unique),bool);np.logical_or.at(vp,inverse,validpoint);np.logical_or.at(vi,inverse,visible);boxes=z['boxes'];inside=(np.abs(unique[None]-boxes[:,None,:3])<=np.maximum(boxes[:,None,3:],1e-6)/2).all(2);groups=[np.flatnonzero(assignment==g) for g in range(int(assignment.max()+1))];supports=np.array([inside[g].any(0) for g in groups]);na=supports.sum(1);npnt=(supports&vp[None]).sum(1);nimg=(supports&vi[None]).sum(1);old=(na>=4)&(npnt>=4)&(npnt/np.maximum(na,1)>=.5);new=(na>=4)&(nimg>=4)&(nimg/np.maximum(na,1)>=.5);return old,new

def main():
 check();rows=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  if not e['has_B']:continue
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));f=np.load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));bb=np.load(B/'image'/name/'banks_PRIVATE'/(sid+'.npz'));old,new=scope(z,ev['assignment'],f['valid_B'],bb['B_graph_visible']);assert np.array_equal(old,ev['S']);top=int(ev['assignment'][ev['p0'].argmax()]);rows.append({'cohort':'target','source':src,'target':tgt,'id_PRIVATE':sid,'old_S':int(old.sum()),'image_S':int(new.sum()),'added':int((new&~old).sum()),'removed':int((old&~new).sum()),'image_S_ge2':bool(new.sum()>=2),'Top_global_old_supported':bool(old[top]),'Top_global_image_supported':bool(new[top]),'Top_global_newly_supported':bool(new[top] and not old[top]),'Top_global_no_longer_supported':bool(old[top] and not new[top])})
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  src,sid=e['entry']['source'],e['entry']['row']['id'];rec=rd(B/'source'/src/(sid+'.json'))
  if not rec['has_B']:continue
  z=np.load(R/e['entry']['files']['prefix']['path']);f=np.load(B/'source'/src/(sid+'.npz'));g=np.load(B/'source'/src/(sid+'_graph.npz'));old,new=scope(z,f['assignment'],f['valid_B'],g['visible']);assert np.array_equal(old,f['S']);rows.append({'cohort':'source','source':src,'target':src,'id_PRIVATE':sid,'old_S':int(old.sum()),'image_S':int(new.sum()),'added':int((new&~old).sum()),'removed':int((old&~new).sum()),'image_S_ge2':bool(new.sum()>=2)})
 summary=[]
 for cohort,src,tgt in sorted({(r['cohort'],r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['cohort'],r['source'],r['target'])==(cohort,src,tgt)];out={'cohort':cohort,'source':src,'target':tgt,'B_inputs':len(rr),'old_S_groups':sum(r['old_S'] for r in rr),'image_S_groups':sum(r['image_S'] for r in rr),'added_groups':sum(r['added'] for r in rr),'removed_groups':sum(r['removed'] for r in rr),'inputs_with_qualification_change':sum(r['added']+r['removed']>0 for r in rr),'inputs_image_S_ge2':sum(r['image_S_ge2'] for r in rr)}
  if cohort=='target':
   for k in ['Top_global_old_supported','Top_global_image_supported','Top_global_newly_supported','Top_global_no_longer_supported']:out[k]=sum(r[k] for r in rr)
  summary.append(out)
 out={'status':'GT_FREE_BPOINT_VS_BIMAGE_QUALIFICATION_METADATA_ONLY','thresholds_unchanged':{'minimum_unique_A_and_B_support':4,'minimum_B_coverage':.5},'group_assignment_independent_of_validity_mask_and_unchanged':True,'rows':summary,'target_or_source_GT_read':False,'TTA_predictions_read':False,'model_or_task_scoring':False,'current_TTA_evidence_C_teacher_scope_or_predictions_changed':False,'image_visibility_not_certified_instance_identity':True,'any_new_method_requires_new_design_calibration_and_matched_controls':True,'FINAL_access':False};dump(H/'VISIBILITY_QUALIFICATION_ROWS_PRIVATE.json',rows);dump(H/'VISIBILITY_QUALIFICATION_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
