"""Postsealed all13-arm exposed-development task score; no parameter update."""
from common40 import *
import numpy as np,collections

def main():
 check_plan();out=BASE/'predictions208_attempt001';verify(out);assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 records=[];negative=[];summary={};coverage={};scenes=[];raws=[]
 for r in rows():
  z=arrays(r);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));rc=read(out/'cases_PRIVATE'/(r['id']+'.json'));old39=read(PREVIOUS/'predictions_PRIVATE'/(r['id']+'.json'));old38=read(ROOT/old39['previous_case_file']);uq,inv=np.unique(p['queries'],return_inverse=True);v=ious(r,z['source_boxes'][uq])[inv];record={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,p['queries']))),'IoU':dict(zip(ARMS,map(float,v))),'readouts':dict(old38['readouts'],**old39['readouts'],**rc['readouts']),'bbox_parse':rc['bbox_parsed_original'],'assignment':rc['assignment'],'subgroups_defined_without_GT':rc['subgroups_defined_without_GT']};records.append(record)
  for a in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_compatible_fullmap','som_compatible_fullmap','bbox_fullmap','bbox_IoU_fullmap']:
    if a!=ref and record['IoU'][a]<record['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(record['IoU'][a]-record['IoU'][ref])})
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl];summary[pl]={};coverage[pl]={}
  for a in ARMS:
   v=np.array([r['IoU'][a] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);summary[pl][a]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in rr)}
  for a in ARMS[3:]:
   ss=collections.Counter(r['readouts'][a]['status'] for r in rr);coverage[pl][a]={'n':len(rr),'mapped':sum(r['readouts'][a]['mapped'] for r in rr),'unknown_fallback':sum(not r['readouts'][a]['mapped'] for r in rr),'status_counts':dict(ss)}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   ss=[r for r in rr if r['scene_PRIVATE']==s];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in ARMS}})
 summary['balanced_mIoU_percent']={a:float(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']])) for a in ARMS};pairs=[('bbox_IoU_fullmap','source','bboxIoU_vs_source'),('bbox_IoU_fullmap','bbox_fullmap','rectangle_extent_vs_center'),('bbox_IoU_fullmap','point_compatible_fullmap','bboxIoU_vs_point'),('bbox_IoU_fullmap','som_compatible_fullmap','bboxIoU_vs_SoM'),('bbox_IoU_fullmap','legacy_G','bboxIoU_vs_legacy'),('bbox_IoU_fullmap','bbox_IoU_oldsubset','bboxIoU_mapping_coverage')];rng=np.random.default_rng(40);clusters={pl:[[r for r in records if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in records if r['platform']==pl})] for pl in ['drone','quad']};contrasts=[]
 for a,b,label in pairs:
  by={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in records if r['platform']==pl])*100) for pl in ['drone','quad']};dist=[]
  for _ in range(2000):
   dd=[]
   for pl,cc in clusters.items():
    rs=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];dd.append(np.mean([r['IoU'][a]-r['IoU'][b] for r in rs]))
   dist.append(100*np.mean(dd))
  contrasts.append({'contrast':label,'arm':a,'reference':b,'platform_gain_pp':by,'balanced_gain_pp':float(np.mean(list(by.values()))),'descriptive_scene_bootstrap95_pp':np.quantile(dist,[.025,.975]).tolist(),'few_exposed_scenes_not_confirmatory':True})
 subgroups=[]
 for group in ['overall','center_unknown','center_IoU_mask_conflict']:
  for pl in ['drone','quad']:
   rr=[r for r in records if r['platform']==pl and (group=='overall' or r['subgroups_defined_without_GT'][group])];N=len(rr);d=np.array([r['IoU']['bbox_IoU_fullmap']-r['IoU']['bbox_fullmap'] for r in rr]);subgroups.append({'subgroup':group,'platform':pl,'n':N,'mean_IoU_percent':{a:float(np.mean([r['IoU'][a] for r in rr])*100) if N else None for a in ['source','bbox_fullmap','bbox_IoU_fullmap','point_compatible_fullmap','legacy_G']},'bboxIoU_vs_center_mean_gain_pp':float(d.mean()*100) if N else None,'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'unchanged':int((abs(d)<=1e-10).sum()),'posthoc_mechanism_description_not_rule_selection':True})
 costs={'observations':208,'new_VLM_calls':0,'new_GPU_calls':0,'new_TTA_or_training_steps':0,'reuse':'sameR39parsedvalues/grid/sourceboxes/masks; no new inference','originalR39actualcalls':209,'originalR39costs_retained':'research/round39/experiment/COSTS.json'};b=summary['balanced_mIoU_percent'];decision={'state':'COMPLETED_CACHED_STANDARD_ASSIGNMENT_CONTROL_NO_TTA','IoU_vs_center_pp':b['bbox_IoU_fullmap']-b['bbox_fullmap'],'IoU_vs_source_pp':b['bbox_IoU_fullmap']-b['source'],'IoU_vs_point_pp':b['bbox_IoU_fullmap']-b['point_compatible_fullmap'],'IoU_vs_legacy_pp':b['bbox_IoU_fullmap']-b['legacy_G'],'all13_oldcontrols_and_original_bbox_values_unchanged':True,'A208_is_exposed_development':True,'new_VLM_GPU_TTA_calls':0,'not_a_new_algorithm':True,'no_default_replacement_or_TTA_adoption':True,'identity_review':'same unchangedpresealed24,joinonlyafterseal,notfeedbackfilter','all_unknowns_and_negatives_retained':True}
 for name,data in [('TASK_SUMMARY',summary),('ALL_ROWS_PRIVATE',records),('ALL_NEGATIVES',negative),('COVERAGE_UNKNOWN',coverage),('SCENE_ROWS',scenes),('FACTOR_CONTRASTS',contrasts),('MECHANISM_SUBGROUPS',subgroups),('COSTS',costs),('DECISION',decision)]:dump(EXP/(name+'.json'),data)
 print('R40_BALANCED',b);print('R40_DECISION',decision);print('R40_SUBGROUPS',subgroups)
if __name__=='__main__':main()
