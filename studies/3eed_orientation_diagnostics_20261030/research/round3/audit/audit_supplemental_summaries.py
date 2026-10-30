"""Audit supplemental summaries from independently validated sealed scalar rows only."""
import json,pathlib,hashlib,shutil
from collections import defaultdict
import numpy as np
R=pathlib.Path(__file__).resolve().parent;P=R.parents[2]/'crossplatform/round3/diagnostics'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 core=json.loads((R/'INDEPENDENT_RESULTS_AUDIT.json').read_text());assert core['status']=='passed'
 parent=json.loads((P/'RESULTS_PRIVATE.json').read_text());assert sha(P/'RESULTS_PRIVATE.json')==core['parent_result_sha256']
 pairs=json.loads((P/'PAIRED_COMPARISONS.json').read_text());pert=json.loads((P/'PERTURBATION_SUMMARY.json').read_text());by={(x['id'],x['arm']):x for x in parent['rows']}
 errors=[];pairchecks=[]
 for pair in pairs['pairs']:
  a,b=pair['a'],pair['b'];grouped=defaultdict(list);ds=[]
  for (sid,arm),row in by.items():
   if arm!=a:continue
   delta=row['iou']-by[sid,b]['iou'];ds.append(delta);grouped[row['platform'],row['scene']].append(delta)
  values={p:np.array([np.mean(v) for (q,s),v in sorted(grouped.items()) if q==p]) for p in ['waymo','drone','quad']}
  estimate=sum(v.mean()/3 for v in values.values());rng=np.random.default_rng(20261030);draws=np.zeros(10000)
  for p,v in values.items():draws+=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)/3
  ci=np.quantile(draws,[.025,.975]);errors.extend([abs(np.mean(ds)-pair['frame_mean_delta']),abs(estimate-pair['equalplatform_scene_macro_delta']),*abs(ci-pair['platformstratified_scene_bootstrap_95ci_descriptive'])])
  for s in pair['scene_deltas']:assert len(grouped[s['platform'],s['scene']])==s['n'];errors.append(abs(np.mean(grouped[s['platform'],s['scene']])-s['delta']))
  ds=np.array(ds);assert sum(ds>1e-10)==pair['positive'] and sum(ds< -1e-10)==pair['negative'] and sum(abs(ds)<=1e-10)==pair['same']
  pairchecks.append({'a':a,'b':b,'primary_scene_macro_delta':float(estimate),'frameweighted_delta':float(ds.mean()),'scene_bootstrap_95ci':ci.tolist()})
 perrors=[];pchecks=[];f={r['id']:r for r in json.loads((P.parent/'p2/FEATURES_PRIVATE.json').read_text())['rows']}
 for g in pert['groups']:
  rows=[x for x in parent['P2_controlled_perturbation_rows'] if x['perturbation']==g['perturbation']];groups=defaultdict(list)
  for x in rows:groups[x['id']].append(x)
  assert len(groups)==g['parents_with_trace'] and len(rows)==g['variants_not_independent_evidence'] and 54-len(groups)==g['missing_no_trace_parents']
  absent=set(f)-set(groups);assert all(f[s]['features']['unique_crop_count']<3 for s in absent)
  means=[]
  for p in g['parents']:
   rr=groups[p['id']];v=np.array([x['delta_unperturbed_probe'] for x in rr]);assert len(v)==p['variants'] and sum(not x['valid_PCA_yaw'] for x in rr)==p['invalid_variants'];means.append(v.mean());perrors +=[abs(v.mean()-p['mean_delta']),abs(v.min()-p['min_delta']),abs(v.max()-p['max_delta'])]
  perrors.append(abs(np.mean(means)-g['frame_mean_of_withinparent_variant_deltas']))
  pchecks.append({'group':g['perturbation'],'parents':len(groups),'absent_min3_support':len(absent),'variants_not_new_scenes':len(rows),'mean_withinparent_delta':float(np.mean(means))})
 assert len(pairchecks)==7 and len(pchecks)==5 and max(errors)<1e-12 and max(perrors)<1e-12
 report={'status':'passed','scope':'Recomputation from previously independent3Dconvex-validated scalar rows, no new target access or model','paired_summary_sha256':sha(P/'PAIRED_COMPARISONS.json'),'perturbation_summary_sha256':sha(P/'PERTURBATION_SUMMARY.json'),'paired_comparisons':pairchecks,'max_abs_paired_statistic_error':float(max(errors)),'perturbation_groups':pchecks,'max_abs_perturbation_aggregate_error':float(max(perrors)),'no_variants_treated_as_new_independent_parents_or_scenes':True}
 (R/'SUPPLEMENTAL_SUMMARY_AUDIT.json').write_text(json.dumps(report,indent=2));backup=R/'INDEPENDENT_RESULTS_AUDIT_CORE.json'
 if not backup.exists():shutil.copyfile(R/'INDEPENDENT_RESULTS_AUDIT.json',backup)
 core['supplemental_summary_audit']={'file':'SUPPLEMENTAL_SUMMARY_AUDIT.json','sha256':sha(R/'SUPPLEMENTAL_SUMMARY_AUDIT.json'),'status':'passed','paired_comparison_count':7,'perturbation_group_count':5,'max_abs_paired_statistic_error':report['max_abs_paired_statistic_error'],'max_abs_perturbation_aggregate_error':report['max_abs_perturbation_aggregate_error']}
 (R/'INDEPENDENT_RESULTS_AUDIT.json').write_text(json.dumps(core,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['paired_comparisons','perturbation_groups']},indent=2))
if __name__=='__main__':main()
