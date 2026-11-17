"""All component endpoint scores after prediction seal and native audit."""
from common143 import *
import numpy as np
CORRECT=['partial_only','rank_only']
def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS' and rd(B/'formal/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));qs=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[q]) for k,q in qs.items()},'queries_PRIVATE':{k:int(q) for k,q in qs.items()},'components':{'partial_active':bool(ev['partial_active']),'rank_active':bool(ev['rank_active']),'wrong_C_effective':bool(ev['C_wrong_effective']),'true_C_mass':float(ev['true_C_conditional_group_mass']),'wrong_C_mass':float(ev['wrong_C_conditional_group_mass'])}})
 controls={m:['image_B_TTA','F_RGBB_only','direct_partial' if m=='partial_only' else 'direct_rank','partial_only_wrong_C' if m=='partial_only' else 'rank_only_wrong','F_Bimage_only_teacher','FA_FB_probability_mean','FAB_FB_probability_mean','Frozen_A','Frozen_F'] for m in CORRECT}
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in rr[0]['IoUs']:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for scene in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==scene for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':scene,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for method in CORRECT:
    for ctrl in controls[method]:
     dy=np.array([r['IoUs'][method]-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':method,'control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE'][method]!=r['queries_PRIVATE'][ctrl] for r in rr)})
 for ag in ['raw','design_weighted']:
  for arm in rows[0]['IoUs']:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 dec={'predeclared_reference':'R142 full partial+rank+keep','original_R142_adoption_unchanged':True,'new_architecture_or_loss':False,'endpoints':{},'FINAL_access':False,'no_target_GT_hyperparameter_or_per_direction_output_choice':True,'exposed_development_structure_ablation_not_independent_test':True}
 for method in CORRECT:
  comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==method)-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in controls[method]} for ag in ['raw','design_weighted']};passed=all(v>0 for x in comp.values() for v in x.values());dec['endpoints'][method]={'comparisons':comp,'retained_development_candidate':passed}
 dec['disposition']='retain_positive_component_endpoint_as_exposed_development_candidate' if any(x['retained_development_candidate'] for x in dec['endpoints'].values()) else 'keep_R142_full_reference_no_component_replacement';dec['partial_wrong_confidence_mass_only_factor2_matched_not_exact_identity_certainty']=True
 for k,v in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
