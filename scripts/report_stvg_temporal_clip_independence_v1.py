"""Independent numpy readback of six saved engineering stages; never task GT."""
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_clip_independence_v1 import D,P,EARLY

def metric(a,b):
 a=a.numpy().astype(np.float64);b=b.numpy().astype(np.float64);d=a-b
 return dict(max_relative=float(np.max(abs(d)/np.maximum(1,abs(b)))),max_absolute=float(abs(d).max()),RMS=float(np.sqrt(np.mean(d*d))),relative_L2=float(np.linalg.norm(d.ravel())/max(1e-30,np.linalg.norm(b.ravel()))),exact=bool(np.array_equal(a,b)))

def main():
 p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_STAGE_BARRIER.json');v=read(D/'ENGINEERING_RESULTS.json');assert b['plan_sha256']==sha(D/'PLAN_LOCK.json')and b['no_GT']and v['no_GT']and v['task_predictions']==0;assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in b['files']);assert len(b['files'])==6 and not(D/'GT_EXPOSURE.json').exists();z={r:load(D/'stages'/f'{r}.pt')for r in ['Native0','Native1','Batch','Repeat','OtherPermutation','Single']};checks={};e=load(EARLY/'early_inputs'/f"{p['record']['alias']}.pt")
 for off in (0,1):
  pr=z['Native'+str(off)]['projected'];h,w=e['original_views'][off]['info']['fea_map_size'];want=e['captured_encoder_inputs'][off]['src'][-h*w:].permute(1,2,0).reshape(pr.shape);checks['Native'+str(off)+'_exact']=bool(np.array_equal(pr.numpy(),want.numpy()))
 for stage in ['raw','projected']:
  for name,a,bb in [('same_batch',z['Batch'][stage],z['Repeat'][stage]),('first_clip_independence',z['OtherPermutation'][stage][:8],z['Batch'][stage][:8]),('single_versus_batch',z['Batch'][stage][:8],z['Single'][stage])]:
   key=name+'_'+stage;checks[key]=metric(a,bb)
   for k,x in checks[key].items():
    y=v['checks'][key][k];assert x==y if isinstance(x,bool)else abs(x-y)<1e-12
 for k in ['Native0_exact','Native1_exact']:assert checks[k]==v['checks'][k]
 assert z['Batch']['input_sha256']==z['Repeat']['input_sha256']and all(x==6 for x in v['actual_calls'].values());closed=False==v['checks']['original_R30_qualification_passed'];assert closed
 audit=dict(status='pass',scope='numpy readback of engineering endpoints, not teacher/task/OPD qualification',pins=len(p['pins']),six_receipts_exact=True,independent_numerical_comparisons=6,Native_controls_checked=2,original_R30_remains_failed=True,GT_read=False,task_scoring=False)
 for f in ['ROOT_AUDIT.json','INDEPENDENT_AUDIT.json']:put(P/f,audit)
 put(P/'RESULTS_PUBLIC.json',dict(status='completed_engineering_audit_only',runtime=v['runtime'],audit_completed=True,independence_pass=v['independence_pass'],checks=checks,original_R30_qualification_passed=False,current_comparison_with_original_bound_passed=v['checks']['current_fullclip_comparison_with_original_bound_passed'],old_batch_centers_reproduced=v['checks']['old_saved_batch_centers_exact'],old_single_reproduced=v['checks']['old_saved_single_exact'],task_predictions=0,GT_read=False,actual_calls=v['actual_calls'],cost=read(D/'COST.json')))
 put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],data_role='same exposed RC001 input only; no task GT',excluded=['private pixels','captions','parent IDs','weights','GT','raw feature caches']))
 put(P/'DECISION_PUBLIC.json',dict(completed=True,independence_pass=v['independence_pass'],original_R30_qualification_passed=False,teacher_quality='not_tested',goal_achieved=False,no_automatic_precision_batch_or_task_rerun=True))
 lines=['# Fixed local-clip engineering audit, no task scores','','All six complete raw768/projection256 stages saved before numeric comparison, then independently read back using numpy. Same frozen Source, failed first batch input, AMP/batch8/clip8/center3 unchanged. Native original controls now retained. No fusion, temporal task prediction, new expert, training, student/spatial update or GT.','','```json',json.dumps(read(P/'RESULTS_PUBLIC.json'),indent=2),'```','','The original R30 max-error qualification remains failed. Same-shape repeatability and first-clip invariance under permutation of the other seven clips support these checked engineering properties only. They do not prove exhaustive sample independence, numerical root cause, harmless downstream error, teacher quality or temporal gain. Raw stages distinguish where the batch/single discrepancy is present, without a precision/kernel/batch sweep. No automatic old R30 resume or tolerance change. Independent confirmation and OPD remain untested.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');assert all(sha(R/f)==h for f,h in p['pins'].items());print(json.dumps(read(P/'DECISION_PUBLIC.json')),flush=True)
if __name__=='__main__':main()
