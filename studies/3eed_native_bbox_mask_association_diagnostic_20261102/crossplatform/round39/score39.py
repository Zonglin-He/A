"""Postsealed all13-arm exposed-development task score; no parameter update."""
from common39 import *
import numpy as np,collections

def main():
 check_plan();out=BASE/'inference208_attempt001';verify(out);assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 records=[];negative=[];summary={};coverage={};scenes=[];raws=[]
 for r in rows():
  z=arrays(r);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));rc=read(out/'predictions_PRIVATE'/(r['id']+'.json'));old=read(ROOT/rc['previous_case_file']);raw=read(out/rc['call_ref']['file'])['result'];raws.append(raw);uq,inv=np.unique(p['queries'],return_inverse=True);v=ious(r,z['source_boxes'][uq])[inv];record={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,p['queries']))),'IoU':dict(zip(ARMS,map(float,v))),'readouts':dict(old['readouts'],**rc['readouts']),'bbox_parse':raw['parsed']};records.append(record)
  for a in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_compatible_fullmap','som_compatible_fullmap','bbox_fullmap']:
    if a!=ref and record['IoU'][a]<record['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(record['IoU'][a]-record['IoU'][ref])})
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl];summary[pl]={};coverage[pl]={}
  for a in ARMS:
   v=np.array([r['IoU'][a] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);summary[pl][a]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in rr)}
  for a in ARMS[3:]:
   ss=collections.Counter(r['readouts'][a]['status'] for r in rr);coverage[pl][a]={'n':len(rr),'mapped':sum(r['readouts'][a]['mapped'] for r in rr),'unknown_fallback':sum(not r['readouts'][a]['mapped'] for r in rr),'status_counts':dict(ss)}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   ss=[r for r in rr if r['scene_PRIVATE']==s];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in ARMS}})
 summary['balanced_mIoU_percent']={a:float(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']])) for a in ARMS};pairs=[('bbox_fullmap','source','bbox_vs_source'),('bbox_fullmap','point_compatible_fullmap','native_bbox_vs_compatible_point'),('bbox_fullmap','som_compatible_fullmap','native_bbox_vs_SoM'),('bbox_fullmap','legacy_G','native_bbox_vs_legacy_G'),('bbox_fullmap','bbox_oldsubset','bbox_mapping_coverage')];rng=np.random.default_rng(39);clusters={pl:[[r for r in records if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in records if r['platform']==pl})] for pl in ['drone','quad']};contrasts=[]
 for a,b,label in pairs:
  by={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in records if r['platform']==pl])*100) for pl in ['drone','quad']};dist=[]
  for _ in range(2000):
   dd=[]
   for pl,cc in clusters.items():
    rs=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];dd.append(np.mean([r['IoU'][a]-r['IoU'][b] for r in rs]))
   dist.append(100*np.mean(dd))
  contrasts.append({'contrast':label,'arm':a,'reference':b,'platform_gain_pp':by,'balanced_gain_pp':float(np.mean(list(by.values()))),'descriptive_scene_bootstrap95_pp':np.quantile(dist,[.025,.975]).tolist(),'few_exposed_scenes_not_confirmatory':True})
 costs={'observations':208,'new_calls':208,'reused_preflight_calls':12,'images_per_new_call':1,'input_tokens':sum(r['input_tokens'] for r in raws),'image_visual_tokens':sum(sum(r['image_visual_tokens']) for r in raws),'generated_tokens':sum(r['generated_tokens'] for r in raws),'seconds_generate':sum(r['seconds_generate'] for r in raws),'parsed_status_counts':dict(collections.Counter(r['parsed']['status'] for r in raws)),'invalid_reasons':dict(collections.Counter(r['parsed']['reason'] for r in raws if r['parsed']['reason'])),'no_model_perfect_hit_gate_or_extra_synthetic_generations':True};b=summary['balanced_mIoU_percent'];gate=all(b['bbox_fullmap']>b[a] for a in ['source','point_compatible_fullmap','som_compatible_fullmap']);decision={'state':'COMPLETED_EXPOSED_G_ONLY_INTERFACE_NO_TTA_ADOPTION','above_source_point_SoM_overall':gate,'bbox_vs_source_pp':b['bbox_fullmap']-b['source'],'bbox_vs_point_pp':b['bbox_fullmap']-b['point_compatible_fullmap'],'bbox_vs_legacy_pp':b['bbox_fullmap']-b['legacy_G'],'may_replace_precision_default':False,'identity_reliability':'pending comparison with original unchanged presealed24 annotations; provisional assistant identity not gold','A208_is_exposed_development':True,'no_FINAL_T_U_or_student_updates':True,'single_output_template_change_not_RefCOCO_reproduction':True,'coordinate_uncertainty_and_center_mask_limitation_preserved':True,'all_unknowns_and_negatives_retained':True}
 for name,data in [('TASK_SUMMARY',summary),('ALL_ROWS_PRIVATE',records),('ALL_NEGATIVES',negative),('COVERAGE_UNKNOWN',coverage),('SCENE_ROWS',scenes),('FACTOR_CONTRASTS',contrasts),('COSTS',costs),('DECISION',decision)]:dump(EXP/(name+'.json'),data)
 print('R39_BALANCED',b);print('R39_COSTS',costs);print('R39_DECISION',decision)
if __name__=='__main__':main()
