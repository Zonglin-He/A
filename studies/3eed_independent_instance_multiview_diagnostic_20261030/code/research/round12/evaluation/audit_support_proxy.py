"""Postseal support-proxy implementation/reduction review. No new GT/P/IoU reads.
Inputs are alreadylabelled support counts, sealed task scalars and predictiontrace.
"""
from pathlib import Path
import json,hashlib,math,argparse,sys,collections
import numpy as np
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D));from diagnose_support_postseal import proxy
ROOT=D.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(a):
 plan=read(a.plan);ss=read(a.support/'SEALED.json');pseal=read(a.predictions/'SEALED.json');scoreseal=read(a.scored/'SEALED.json');su=read(a.support/'RESULTS_PRIVATE.json');task=read(a.scored/'RESULTS_PRIVATE.json');rows=su['rows_PRIVATE'];tr=read(a.predictions/'trace.json');manifest=read(a.predictions/'manifest_input.json');paired=read(a.paired)
 assert plan['postseal_descriptive_only'] and not plan['method_changes_allowed'] and plan['program_sha256']==sha(D/'diagnose_support_postseal.py')==ss['program_sha256'] and ss['plan_sha256']==sha(a.plan)
 assert plan['prediction_seal_sha256']==sha(a.predictions/'SEALED.json')==su['prediction_seal_sha256'] and plan['scored_seal_sha256']==sha(a.scored/'SEALED.json')==su['scored_seal_sha256']
 # Close actual input-provenance gap in original analysis script without rerunning it.
 for base,seal in [(a.predictions,pseal),(a.scored,scoreseal),(a.support,ss)]:
  for n,h in seal['files'].items():p=(base/n).resolve();assert p.is_relative_to(base.resolve()) and sha(p)==h
 assert read(a.support/'SUMMARY.json')=={k:v for k,v in su.items() if k!='rows_PRIVATE'}
 assert len(rows)==len(tr)==len(manifest)==146
 lookup={r['id_PRIVATE']:r for r in rows};taskmap={r['id']:r for r in task['rows_PRIVATE']};trace={r['id']:r for r in tr};aliases={r['id']:i+1 for i,r in enumerate(manifest)}
 assert set(lookup)==set(taskmap)==set(trace)
 checks=0;maxerr=0.
 def cmp(x,y):
  nonlocal checks,maxerr
  checks+=1
  if isinstance(x,dict):
   for k,v in x.items():cmp(v,y[k])
  elif isinstance(x,list):
   assert len(x)==len(y)
   for v,z in zip(x,y):cmp(v,z)
  elif x is None or y is None:assert x is y
  elif isinstance(x,(float,int)) and isinstance(y,(float,int)):
   e=abs(float(x)-float(y));maxerr=max(maxerr,e);assert e<1e-12,(x,y)
  else:assert x==y
 for r in rows:
  for key in ('grabcut_support_proxy','single_support_proxy','MV_corroborating_anchor_support_proxy'):
   x=r[key]
   if x is None:continue
   n=x['n'];i=x['inside_official_target_OBB_n'];wide=x['inside_OBB_plus_fixed_0p1m_n'];assert 0<=i<=wide<=n
   cmp(x['inside_OBB_fraction'],i/n if n else None);cmp(x['inside_OBB_plus_fixed_0p1m_fraction'],wide/n if n else None)
  diag=trace[r['id_PRIVATE']]['diagnostics'];cmp(r['MV_admitted'],'multiview_geometry' in diag)
  if r['MV_admitted']:
   ids=set(i for v in diag['multiview']['neighbors'] for i in v['matched_anchor_raw_ids_unique']);cmp(r['MV_corroborating_anchor_support_proxy']['n'],len(ids))
 for p in ('drone','quad'):
  rr=[r for r in rows if r['platform']==p]
  for key in ('grabcut_support_proxy','single_support_proxy','MV_corroborating_anchor_support_proxy'):
   v=[r[key] for r in rr if r[key] is not None and r[key]['n']>0];N=sum(x['n'] for x in v);I=sum(x['inside_official_target_OBB_n'] for x in v)
   exp={'nonempty_case_n':len(v),'point_n':N,'inside_n':I,'pointweighted_inside_fraction':I/N if N else None,'case_mean_inside_fraction':float(np.mean([x['inside_OBB_fraction'] for x in v])) if v else None,'all_points_outside_case_n':sum(x['inside_official_target_OBB_n']==0 for x in v),'inside_any_case_n':sum(x['inside_official_target_OBB_n']>0 for x in v),'fixed_collar_pointweighted_fraction':sum(x['inside_OBB_plus_fixed_0p1m_n'] for x in v)/N if N else None};cmp(exp,su['summary'][p][key])
  matched=[r for r in rr if r['grabcut_support_proxy'] is not None and r['grabcut_support_proxy']['n']>0 and r['single_support_proxy'] is not None and r['single_support_proxy']['n']>0];delta=[r['single_support_proxy']['inside_OBB_fraction']-r['grabcut_support_proxy']['inside_OBB_fraction'] for r in matched]
  cmp({'n_common_nonempty':len(matched),'mean_case_fraction_delta':float(np.mean(delta)) if delta else None,'improve':sum(x>1e-8 for x in delta),'worsen':sum(x< -1e-8 for x in delta),'tie':sum(abs(x)<=1e-8 for x in delta),'original_points':sum(r['grabcut_support_proxy']['n'] for r in matched),'new_points':sum(r['single_support_proxy']['n'] for r in matched),'original_inside':sum(r['grabcut_support_proxy']['inside_official_target_OBB_n'] for r in matched),'new_inside':sum(r['single_support_proxy']['inside_official_target_OBB_n'] for r in matched)},paired['summary'][p])
  mvmatched=[r for r in rr if r['MV_corroborating_anchor_support_proxy'] is not None and r['MV_corroborating_anchor_support_proxy']['n']>0 and r['single_support_proxy'] is not None and r['single_support_proxy']['n']>0]
  mvdelta=[r['MV_corroborating_anchor_support_proxy']['inside_OBB_fraction']-r['single_support_proxy']['inside_OBB_fraction'] for r in mvmatched]
  cmp({'n':len(mvmatched),'mean_fraction_delta':float(np.mean(mvdelta)) if mvdelta else None,'better_n':sum(x>1e-8 for x in mvdelta),'worse_n':sum(x< -1e-8 for x in mvdelta),'same_n':sum(abs(x)<=1e-8 for x in mvdelta)},su['summary'][p]['matched_nonempty_single_to_corroborated_anchor'])
 # Analytic inverse-yaw and fixed local-axis collar test; all arrays synthetic.
 theta=np.pi/6;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]]);target=np.array([.4,-.3,1.,4,2,2,theta]);local=np.array([[1.9,.9,0],[2.05,0,0],[2.15,0,0],[0,0,1.15]]);q=proxy(local@R.T+target[:3],target);assert q['inside_official_target_OBB_n']==1 and q['inside_OBB_plus_fixed_0p1m_n']==2
 empty=proxy(np.empty((0,3)),target);assert empty['inside_OBB_fraction'] is None and empty['inside_OBB_plus_fixed_0p1m_fraction'] is None
 selected={}
 for sid,number in aliases.items():
  if number not in (116,117):continue
  x=lookup[sid]['MV_corroborating_anchor_support_proxy'];row=taskmap[sid];mv=row['arms']['multiview_instance_direct'];source=row['arms']['source'];ids=set(i for v in trace[sid]['diagnostics']['multiview']['neighbors'] for i in v['matched_anchor_raw_ids_unique']);assert x['n']==len(ids)
  selected['case_'+str(number)]={'platform':row['platform'],'anchor_corroborating_raw_point_n':x['n'],'inside_official_OBB_n':x['inside_official_target_OBB_n'],'inside_fixed_expanded_OBB_n':x['inside_OBB_plus_fixed_0p1m_n'],'source_IoU':source['iou'],'MV_IoU':mv['iou'],'IoU_delta':mv['iou']-source['iou'],'center_error_increase_m':mv['coordinate_errors']['center_L2_m']-source['coordinate_errors']['center_L2_m'],'point_scope':'Anchorcorroboration rawIDunion only, not allneighbor/context/face-gradient/RGB/prior information.'}
 assert selected['case_117']['anchor_corroborating_raw_point_n']==selected['case_117']['inside_official_OBB_n']==37 and selected['case_117']['IoU_delta']<0 and selected['case_117']['center_error_increase_m']>0
 assert selected['case_116']['anchor_corroborating_raw_point_n']==57 and selected['case_116']['inside_official_OBB_n']==0 and selected['case_116']['IoU_delta']==0
 out={'state':'PASS_R12_POSTSEAL_SUPPORT_PROXY_IMPLEMENTATION_AND_SCALAR_REDUCTION_REVIEW','analysis_program_sha256':sha(D/'diagnose_support_postseal.py'),'plan_sha256':sha(a.plan),'support_seal_sha256':sha(a.support/'SEALED.json'),'support_results_sha256':sha(a.support/'RESULTS_PRIVATE.json'),'paired_support_summary_sha256':sha(a.paired),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'auditor_sha256':sha(Path(__file__)),'finite_checks':checks,'max_numeric_difference':maxerr,'all_actual_analysis_inputs_rehashed_against_seals':True,'analytic_inverse_yaw_and_expanded_local_box_test_pass':True,'implementation_review':['row-vector delta@R_yaw is inverse rotation into targetlocal coordinates, not delta@R.T.','Drone points use nativefloat32 +1.8; R5GT adds +1.8 in rawmetadata dtype then truncatesfirst7D/float32. Same modelframe, not assumed identical arithmetic. NoGTyaw update to predictions.','GrabCut/single support IDs indexsorted unique modelXYZ exactlylike_G; MVmatched IDs indexoriginal rawrows thenmodelshift.','FixedexpandedOBB adds0.1toeach localhalfextent (fullsize+.2), notEuclidean0.1distance or semanticmaskGT.','Originalscript checksplan/predictionseal/scoredseal/indexhash beforetargetGT read, but didnotrecheckeveryconsumedtrace/manifest againstseal; thisreviewclosesactualinputbindings now without reexecution.','Originalanalysis containsnooptimizer/newIoU/predictionwrites; onlypostseal output directory.'],'matched_case_fraction_deltas':{p:paired['summary'][p] for p in ('drone','quad')},'selected_alias_only_case_evidence':selected,'claim_limits':['OBB enclosure is spatialproxy, not true pixel/point instance membership. Inside can include otherobjects/background; outside can reflectcalibration/looseGT/partialvisibility/sourceerror.','Ownsupportpopulations differ; onlypaired commonnonempty cases supporta pairedproxycontrast, stilldifferentpointsets andno causalpurityclaim.','SupportIDs includevisibleROI/mask orcorroboratinganchors, notnecessarilyallpointsenteringactualface terms; neighborpoints, normalcontext, RGBcontainment andsourceprior remain.','Case117 weakens a background-only explanation butdoesnotidentifywhichloss or prior causedharm. No currentmethodselection/threshold changes fromthese label-assisted diagnostics.','Currentcommentary mustsayposthoc/postseal descriptive, notindependentconfirmation ornewTTAresult.'],'no_new_GT_metadata_pointfiles_images_IoU_model_GPU_or_prediction_changes':True,'issues':[]};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'checks':checks,'sha256':sha(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('plan','predictions','scored','support','paired','output'):p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
