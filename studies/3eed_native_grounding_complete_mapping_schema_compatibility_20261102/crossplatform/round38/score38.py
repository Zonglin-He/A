"""Postseal engineering factor score; recovery cohort is schema-defined, not GT."""
from common38 import *
import numpy as np,collections

def main():
 check_plan();out=BASE/'predictions208_attempt001';verify(out);assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 records=[];negative=[];summary={};coverage={};scenes=[]
 for r in rows():
  z=arrays(r);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));rc=read(out/'cases_PRIVATE'/(r['id']+'.json'));uq,inv=np.unique(p['queries'],return_inverse=True);v=ious(r,z['source_boxes'][uq])[inv];rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,p['queries']))),'IoU':dict(zip(ARMS,map(float,v))),'recovery':{i:rc['calls'][i]['recovery'] for i in ['point','som']},'readouts':rc['readouts']};records.append(rec)
  for a in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_strict_fullmap','point_compatible_fullmap','som_compatible_fullmap']:
    if a!=ref and rec['IoU'][a]<rec['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(rec['IoU'][a]-rec['IoU'][ref])})
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl];summary[pl]={};coverage[pl]={}
  for a in ARMS:
   v=np.array([r['IoU'][a] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);summary[pl][a]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in rr)}
  for i in ['point','som']:
   for parser in ['strict','compatible']:
    for m in ['fullmap','oldsubset']:
     a=i+'_'+parser+'_'+m;st=collections.Counter(r['readouts'][a]['status'] for r in rr);coverage[pl][a]={'n':len(rr),'mapped':sum(r['readouts'][a]['mapped'] for r in rr),'unknown_fallback':sum(not r['readouts'][a]['mapped'] for r in rr),'status_counts':dict(st)}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   rs=[r for r in rr if r['scene_PRIVATE']==s];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(rs),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in rs])*100) for a in ARMS}})
 summary['balanced_mIoU_percent']={a:float(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']])) for a in ARMS};recovery=[];contrasts=[]
 for i in ['point','som']:
  for pl in ['drone','quad']:
   rr=[r for r in records if r['platform']==pl and r['recovery'][i]['recovered']];a=i+'_compatible_fullmap';b=i+'_strict_fullmap';delta=np.array([r['IoU'][a]-r['IoU'][b] for r in rr]);recovery.append({'interface':i,'platform':pl,'schema_recovered_n':len(rr),'relative_IoU_mean_gain_pp':float(delta.mean()*100) if len(rr) else None,'sum_gain_pp':float(delta.sum()*100),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'unchanged':int((abs(delta)<=1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries'][b] for r in rr),'posthoc_explanatory_not_confirmation_or_feedback_filter':True})
 pairs=[('point_compatible_fullmap','point_strict_fullmap','point_schema_at_fullmap'),('point_compatible_oldsubset','point_strict_oldsubset','point_schema_at_oldsubset'),('point_compatible_fullmap','point_compatible_oldsubset','compatible_point_mapping_coverage'),('point_compatible_fullmap','som_compatible_fullmap','compatible_interface'),('point_compatible_fullmap','source','compatible_point_vs_source'),('point_compatible_fullmap','legacy_G','compatible_point_vs_legacy')];rng=np.random.default_rng(38);clusters={pl:[[r for r in records if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in records if r['platform']==pl})] for pl in ['drone','quad']}
 for a,b,label in pairs:
  by={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in records if r['platform']==pl])*100) for pl in ['drone','quad']};dist=[]
  for _ in range(2000):
   d=[]
   for pl,cc in clusters.items():
    rr=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];d.append(np.mean([r['IoU'][a]-r['IoU'][b] for r in rr]))
   dist.append(100*np.mean(d))
  contrasts.append({'contrast':label,'arm':a,'reference':b,'platform_gain_pp':by,'balanced_gain_pp':float(np.mean(list(by.values()))),'descriptive_scene_bootstrap95_pp':np.quantile(dist,[.025,.975]).tolist(),'few_exposed_scenes_not_confirmatory':True})
 parser_counts={}
 for i in ['point','som']:
  cc=[r['recovery'][i] for r in records];parser_counts[i]={'n':len(cc),'old_parse_failure':sum(c['old_status']=='parse_failure' for c in cc),'new_parse_failure':sum(c['new_status']=='parse_failure' for c in cc),'recovered':sum(c['recovered'] for c in cc),'singleton_unwrapped':sum(c['singleton_unwrapped'] for c in cc),'label_ignored':sum(c['label_ignored'] for c in cc)}
 b=summary['balanced_mIoU_percent'];decision={'state':'COMPLETED_CACHED_ENGINEERING_CANDIDATE_NOT_NEW_METHOD','original_R37_strict_results_unchanged':True,'compatible_point_vs_strict_pp':b['point_compatible_fullmap']-b['point_strict_fullmap'],'compatible_point_vs_source_pp':b['point_compatible_fullmap']-b['source'],'compatible_point_vs_legacy_pp':b['point_compatible_fullmap']-b['legacy_G'],'new_VLM_GPU_TTA_calls':0,'no_GT_based_parser_or_recovery_case_selection':True,'identity_comparison':'same presealed24 annotations, postseal strict/compatible join by independent reviewer; no annotation changes','A208_exposed_development_not_confirmation':True,'native_coordinate_unit_uncertainty_from_R37_preserved':True}
 dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'ALL_ROWS_PRIVATE.json',records);dump(EXP/'ALL_NEGATIVES.json',negative);dump(EXP/'COVERAGE_UNKNOWN.json',coverage);dump(EXP/'SCENE_ROWS.json',scenes);dump(EXP/'SCHEMA_RECOVERED_SUBSET.json',recovery);dump(EXP/'FACTOR_CONTRASTS.json',contrasts);dump(EXP/'PARSER_COUNTS.json',parser_counts);dump(EXP/'DECISION.json',decision);print('R38_DECISION',decision);print('R38_BALANCED',b);print('R38_PARSER',parser_counts)
if __name__=='__main__':main()
