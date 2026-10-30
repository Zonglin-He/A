"""Post-score descriptive target-OBB support proxy. Never a mask/identity ground truth.
No model, optimizers, candidate changes, new IoU or task-method feedback.
"""
from pathlib import Path
import argparse,json,hashlib,collections,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt,yaw_matrix

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def proxy(p,t):
 p=np.asarray(p).reshape(-1,3);d=(p.astype(float)-t[:3])@yaw_matrix(float(t[6]));a=np.abs(d);half=t[3:6].astype(float)/2
 return {'n':len(p),'inside_official_target_OBB_n':int((a<=half+1e-6).all(1).sum()),'inside_OBB_fraction':float((a<=half+1e-6).all(1).mean()) if len(p) else None,'inside_OBB_plus_fixed_0p1m_n':int((a<=half+.1).all(1).sum()),'inside_OBB_plus_fixed_0p1m_fraction':float((a<=half+.1).all(1).mean()) if len(p) else None}
def main(a):
 plan=read(a.plan);assert plan['program_sha256']==sha(__file__);assert sha(a.plan)==a.plan_sha256 and plan['postseal_descriptive_only'] and not plan['method_changes_allowed'];assert sha(a.predictions/'SEALED.json')==plan['prediction_seal_sha256'] and sha(a.scored/'SEALED.json')==plan['scored_seal_sha256'] and sha(a.scoring_index)==plan['scoring_index_sha256']
 assert not a.output.exists();a.output.mkdir(parents=True);trplan=read(a.predictions/'registered_plan.json');assert sha(trplan['frame_index'])==trplan['frame_index_sha256'];frameindex=read(trplan['frame_index']);trace=read(a.predictions/'trace.json');manifest={r['id']:r for r in read(a.predictions/'manifest_input.json')};idx={r['id']:r for r in read(a.scoring_index)};out=[]
 for tr in trace:
  sid=tr['id'];r=manifest[sid];ix=idx[sid];assert sha(r['pointpath'])==frameindex[tr['anchor_frame_alias']]['point_sha256']
  pp=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3];model=pp.copy()
  if r['platform']=='drone':model[:,2]+=np.float32(1.8)
  unique=np.unique(model,axis=0);assert sha(ix['meta_path'])==ix['metadata_container_sha256'];gt=read(ix['meta_path'])['ground_info'][ix['object_index']]['bbox_3d'];t=transform_gt(gt,r['platform'],ix.get('waymo_view_suffix'))
  d=tr['diagnostics'];g=d['grabcut']['geometry'];ids=np.asarray(g['support_ids_by_arm'][1],np.int64);assert np.all((ids>=0)&(ids<len(unique)));case={'id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'grabcut_support_proxy':proxy(unique[ids],t),'single_support_proxy':None,'MV_corroborating_anchor_support_proxy':None,'MV_admitted':('multiview_geometry' in d),'interpretation':'OBB spatial-support proxy only; outside may be another object, background, calibration error, partial/loose label or bad sourceprior; inside is not proven semantic instance membership.'}
  if 'single_geometry' in d:
   ids=np.asarray(d['single_geometry']['support_ids_by_arm'][1],np.int64);assert np.all((ids>=0)&(ids<len(unique)));case['single_support_proxy']=proxy(unique[ids],t)
  if 'multiview_geometry' in d:
   ids=sorted({int(i) for n in d['multiview']['neighbors'] for i in n['matched_anchor_raw_ids_unique']});assert all(0<=i<len(pp) for i in ids);case['MV_corroborating_anchor_support_proxy']=proxy(model[ids],t)
  out.append(case)
 summary={}
 for platform in ['drone','quad']:
  rr=[r for r in out if r['platform']==platform];stats={}
  for key in ['grabcut_support_proxy','single_support_proxy','MV_corroborating_anchor_support_proxy']:
   vals=[r[key] for r in rr if r[key] is not None and r[key]['n']>0];stats[key]={'nonempty_case_n':len(vals),'point_n':sum(x['n'] for x in vals),'inside_n':sum(x['inside_official_target_OBB_n'] for x in vals),'pointweighted_inside_fraction':sum(x['inside_official_target_OBB_n'] for x in vals)/sum(x['n'] for x in vals) if vals else None,'case_mean_inside_fraction':float(np.mean([x['inside_OBB_fraction'] for x in vals])) if vals else None,'all_points_outside_case_n':sum(x['inside_official_target_OBB_n']==0 for x in vals),'inside_any_case_n':sum(x['inside_official_target_OBB_n']>0 for x in vals),'fixed_collar_pointweighted_fraction':sum(x['inside_OBB_plus_fixed_0p1m_n'] for x in vals)/sum(x['n'] for x in vals) if vals else None,'different_own_support_populations_not_causal_comparison':True}
  matched=[r for r in rr if r['MV_corroborating_anchor_support_proxy'] is not None and r['MV_corroborating_anchor_support_proxy']['n']>0 and r['single_support_proxy'] is not None and r['single_support_proxy']['n']>0]
  delta=[r['MV_corroborating_anchor_support_proxy']['inside_OBB_fraction']-r['single_support_proxy']['inside_OBB_fraction'] for r in matched]
  stats['matched_nonempty_single_to_corroborated_anchor']={'n':len(matched),'mean_fraction_delta':float(np.mean(delta)) if delta else None,'better_n':sum(d>1e-8 for d in delta),'worse_n':sum(d< -1e-8 for d in delta),'same_n':sum(abs(d)<=1e-8 for d in delta),'not_semantic_or_pixelGT':True}
  summary[platform]=stats
 result={'state':'postseal_descriptive_official_target_OBB_support_proxy_not_semanticGT','n':len(out),'summary':summary,'rows_PRIVATE':out,'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'no_new_IoU_or_model_G_threshold_or_predictions_changed':True};(a.output/'RESULTS_PRIVATE.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'SUMMARY.json').write_text(json.dumps({k:v for k,v in result.items() if k!='rows_PRIVATE'},indent=2)+'\n');(a.output/'SEALED.json').write_text(json.dumps({'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'program_sha256':sha(__file__),'plan_sha256':sha(a.plan)},indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['plan','predictions','scored','scoring-index','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--plan-sha256',required=True);main(p.parse_args())
