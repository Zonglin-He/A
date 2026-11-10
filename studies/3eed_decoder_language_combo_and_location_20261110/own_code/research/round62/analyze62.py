"""Saved-only overall/error-overlap/identity summaries; no runtime changes."""
import argparse
from common62 import *
H=ROOT/'research/round62'
def summarize(mode):
 score=H/('score_'+mode+'336_attempt001');tasks=read(score/'ALL_TASK_ROWS_PRIVATE.json');s=read(score/'TASK_SUMMARY.json')['ALL336'];arms=list(s['balanced_mIoU_percent']);scenes=[x for x in read(score/'ALL_SCENE_SUMMARIES_PRIVATE.json') if x['scope']=='ALL336'];metrics=[]
 for arm in arms:
  base='frozen_ABC';values=np.array([t['IoU'][arm] for t in tasks]);original=np.array([t['IoU'][base] for t in tasks]);d=values-original;metrics.append({'arm':arm,'Drone_mIoU_percent':s['drone'][arm]['mIoU_percent'],'Quad_mIoU_percent':s['quad'][arm]['mIoU_percent'],'balanced_mIoU_percent':s['balanced_mIoU_percent'][arm],'balanced_Acc25_percent':float(np.mean([s[p][arm]['Acc25_percent'] for p in ['drone','quad']])),'balanced_Acc50_percent':float(np.mean([s[p][arm]['Acc50_percent'] for p in ['drone','quad']])),'gain_vs_ABC_pp':s['balanced_mIoU_percent'][arm]-s['balanced_mIoU_percent'][base],'query_changes_vs_ABC':sum(t['queries'][arm]!=t['queries'][base] for t in tasks),'IoU_improved_vs_ABC':int((d>1e-10).sum()),'IoU_worsened_vs_ABC':int((d< -1e-10).sum()),'rescued25':int(((original<=.25)&(values>.25)).sum()),'broken25':int(((original>.25)&(values<=.25)).sum()),'scene_means_improved_vs_ABC':sum(x['mIoU_percent'][arm]>x['mIoU_percent'][base]+1e-10 for x in scenes),'scene_means_worsened_vs_ABC':sum(x['mIoU_percent'][arm]<x['mIoU_percent'][base]-1e-10 for x in scenes)})
 sc=[]
 for pl in ['drone','quad']:
  for k,r in enumerate(sorted([x for x in scenes if x['platform']==pl],key=lambda x:x['scene_PRIVATE']),1):
   for arm in arms:sc.append({'stage':mode,'platform':pl,'scene_cluster':k,'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
 return {'stage':mode,'n':336,'metrics':metrics},tasks,sc
def paired(tasks,arm,ref):
 d=np.array([t['IoU'][arm]-t['IoU'][ref] for t in tasks]);return {'arm':arm,'reference':ref,'delta_pp':float(d.mean()*100),'IoU_improved':int((d>1e-10).sum()),'IoU_worsened':int((d< -1e-10).sum()),'query_changes':sum(t['queries'][arm]!=t['queries'][ref] for t in tasks)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['main','semantics'],required=True);args=ap.parse_args();p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json')
 if args.mode=='main':
  assert read(H/'scalar_layers_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');combo,ct,cs=summarize('combo');layers,tasks,scenes=summarize('layers');layers['combo_pairs']=[paired(tasks,'R59_l5','r59_direct_all'),paired(tasks,'R59_l5','native_l5')];base=np.array([t['IoU']['frozen_ABC'] for t in tasks]);last=np.array([t['IoU']['native_l5'] for t in tasks]);overlap=[]
  for i in range(6):
   v=np.array([t['IoU']['native_l'+str(i)] for t in tasks]);rescue=(base<=.25)&(v>.25);old=(base<=.25)&(last>.25);positive=v>base+1e-10;lp=last>base+1e-10;overlap.append({'layer':i,'rescue25_n':int(rescue.sum()),'rescue25_shared_with_last':int((rescue&old).sum()),'rescue25_new_vs_last':int((rescue&~old).sum()),'IoU_improvement_n':int(positive.sum()),'IoU_improvement_new_vs_last':int((positive&~lp).sum())})
  layers['rescue_overlap']=overlap;identity=[]
  for idx,t in enumerate(tasks):
   sid=t['id_PRIVATE'];z=load(ROOT/p['capture_root']/'samples_PRIVATE'/(sid+'.npz'))
   for layer in range(6):
    root=ROOT/p['combo_root' if layer==5 else 'layers_root'];case=read(root/'cases_PRIVATE'/(sid+'.json'))['layer'+str(layer)]['identity'];st=load(root/'states_PRIVATE'/(sid+'.npz'),['layer'+str(layer)+'_A_internal_boxes']);current=st['layer'+str(layer)+'_A_internal_boxes'].astype(float);old=z['A_boxes'].astype(float);q=t['queries']['native_l'+str(layer)];aa=current[q];bb=old[q];inter=np.maximum(np.minimum(aa[:3]+aa[3:]/2,bb[:3]+bb[3:]/2)-np.maximum(aa[:3]-aa[3:]/2,bb[:3]-bb[3:]/2),0).prod();union=np.maximum(aa[3:],0).prod()+np.maximum(bb[3:],0).prod()-inter;selected_iou=float(inter/union) if union>0 else 0.
    for v,record in case.items():identity.append({'row':f'DEV336_{idx+1:05d}','platform':t['platform'],'layer':layer,'view':v,**record,'selected_A_same_slot_internal_box_IoU':selected_iou if v=='A' else None})
  agg=[]
  for layer in range(6):
   rr=[r for r in identity if r['layer']==layer];aa=[r for r in rr if r['view']=='A'];bc=[r for r in rr if r['view']!='A'];agg.append({'layer':layer,'median_same_slot_internal_box_IoU':float(np.median([r['median_same_slot_internal_box_IoU'] for r in rr])),'median_selected_A_same_slot_internal_box_IoU':float(np.median([r['selected_A_same_slot_internal_box_IoU'] for r in aa])),'mean_duplicate_aware_slot_geometry_risk_fraction':float(np.mean([r['outside_duplicate_group_n']/256 for r in rr])),'mean_old_transport_mass_losing_geometric_overlap':float(np.mean([r['old_transport_mass_with_zero_new_geometric_overlap_fraction'] for r in bc])),'physical_identity_not_independent_gold':True})
  layers['identity_summary']=agg;dump(H/'ANALYSIS_COMBO.json',combo);dump(H/'ANALYSIS_MAIN.json',layers);dump(H/'ALL_MAIN_SCENE_SCALARS_PUBLIC_SAFE.json',cs+scenes);dump(H/'ALL_IDENTITY_SCALARS_PUBLIC_SAFE.json',identity);print(json.dumps(layers),flush=True)
 else:
  assert read(H/'scalar_semantics_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');result,tasks,scenes=summarize('semantics');native=next(x for x in result['metrics'] if x['arm']=='selected_native15')['balanced_mIoU_percent'];result['native_gain_pp']=native-next(x for x in result['metrics'] if x['arm']=='frozen_ABC')['balanced_mIoU_percent'];m={x['arm']:x for x in result['metrics']};result['whole_Vshuffled_gain15_vs_gain1_pp']=m['selected_value15']['balanced_mIoU_percent']-m['selected_value1']['balanced_mIoU_percent'];result['pairs']=[paired(tasks,a,'selected_native15') for a in SEMANTIC_ARMS if a!='selected_native15'];strength=read(H/'semantics_audit_attempt001/FINAL_BRANCH_STRENGTH_PUBLIC_SAFE.json');producer=[]
  for t in tasks:
   c=read(ROOT/p['semantics_root']/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'))
   for arm,rec in c.items():
    for v in 'ABC':producer.append({'arm':arm,**rec[v]})
  stats=[]
  for arm in SEMANTIC_ARMS[2:]:
   r=[x for x in strength if x['arm']==arm];a=[x for x in producer if x['arm']==arm];stats.append({'arm':arm,'median_alternative_cosine':float(np.median([x['median_source_vs_alternative_cosine'] for x in a])),'median_final_branch_cosine':float(np.median([x['median_final_branch_cosine_to_native'] for x in r])),'median_final_branch_relative_vector_change':float(np.median([x['median_final_branch_relative_vector_change'] for x in r])),'mean_fraction_final_branch_change_over_0p1':float(np.mean([x['fraction_final_branch_relative_change_over_0p1'] for x in r])),'max_relative_branch_norm_error':max(x['max_relative_branch_norm_error'] for x in a),'zero_alternative_norm_total':sum(x['zero_alternative_norm_count'] for x in a)})
  result['lesion_strength']=stats;dump(H/'ANALYSIS_SEMANTICS.json',result);dump(H/'ALL_SEMANTIC_SCENE_SCALARS_PUBLIC_SAFE.json',scenes);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
