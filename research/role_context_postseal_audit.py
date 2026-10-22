"""Independent sealed-output diagnosis; no new prediction or model call."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.native_event_role_context_v1.run import D,P,TEXT,L,read,put,sha,B

def box_iou(a,b):
 a=np.asarray(a);b=np.asarray(b);lo=np.maximum(a[:,:2]-a[:,2:]/2,b[:,:2]-b[:,2:]/2);hi=np.minimum(a[:,:2]+a[:,2:]/2,b[:,:2]+b[:,2:]/2);inter=np.maximum(hi-lo,0).prod(1);return inter/np.maximum(a[:,2:].prod(1)+b[:,2:].prod(1)-inter,1e-30)
def main():
 start=time.perf_counter();plan=read(D/'plan.json');seal=read(D/'sealed.json');receipt=read(D/'score_receipt.json');assert receipt['seal_sha256']==sha(D/'sealed.json')and receipt['report_sha256']==sha(P/'report.json');assert sha(L/'role_context_input_preflight.json')==plan['grammar_sha256'];report=read(P/'report.json');rows=[]
 for row in report['rows']:
  k=row['query'];e=seal['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],map_location='cpu',weights_only=False);assert tr['native_bitwise']and all(tr['key_noop']);source_iou=None
  if e['supported']:
   cap=torch.load(B/'captures'/f'{k}.pt',map_location='cpu',weights_only=False);source_iou=float(box_iou(tr['argument_source_boxes'],cap['Source']).mean())
  stats=tr['statistics'];q0=tr['q']['NativeG6'];r=dict(query=k,supported=e['supported'],argument_to_Sourceactor_IoU_proxy=source_iou,extra_cells_mean=float(np.mean([x['extra']for x in stats]))if stats else None,actor_fraction_valid=float(np.mean([x['target']/x['valid']for x in stats]))if stats else None,pair_fraction_valid=float(np.mean([(x['target']+x['extra'])/x['valid']for x in stats]))if stats else None,reflection_overlap=float(np.mean([x['argument_reflected_overlap']for x in stats]))if stats else None,empty_ROI_fallbacks=sum(sum(x['fallbacks'])for x in stats),posterior_TV={a:float(abs(q-q0).sum()/2)for a,q in tr['q'].items()},dt_pp={a:100*(v['t']-row['values']['NativeG6']['t'])for a,v in row['values'].items()},dv_pp={a:100*(v['v']-row['values']['NativeG6']['v'])for a,v in row['values'].items()});rows.append(r)
 supported=[x for x in rows if x['supported']];a='NativeRoleContextFusion';groups={g:{m:float(np.mean([100*(x['values'][a][m]-x['values']['Source'][m])for x in report['rows']if x['duration_group']==g]))for m in ['v','t']}for g in ['short_le5s','long_gt5s']}
 out=dict(kind='independent_postseal_diagnosis_not_online_gate',code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),scored_report_sha256=sha(P/'report.json'),supported=len(supported),native64_bitwise=True,grammar_hash_valid=True,argument_Sourceactor_proxy_mean=float(np.mean([x['argument_to_Sourceactor_IoU_proxy']for x in supported])),argument_Sourceactor_proxy_below_half=sum(x['argument_to_Sourceactor_IoU_proxy']<.5 for x in supported),mean_extra_cells=float(np.mean([x['extra_cells_mean']for x in supported])),actor_fraction=float(np.mean([x['actor_fraction_valid']for x in supported])),pair_fraction=float(np.mean([x['pair_fraction_valid']for x in supported])),reflection_overlap=float(np.mean([x['reflection_overlap']for x in supported])),ROI_fallbacks=sum(x['empty_ROI_fallbacks']for x in rows),groups_vs_Source=groups,rows=rows,cost=seal['counts'],GPU_prediction_seconds=seal['seconds'],research_cost_not_deployment_latency=True,seconds=time.perf_counter()-start,new_model=0,new_STVG_predictions=0,qualified=False)
 put(P/'CONTEXT_AUDIT.json',out);print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2));print('best',[(x['query'],x['dt_pp'][a])for x in sorted(rows,key=lambda x:x['dt_pp'][a],reverse=True)[:4]]);print('worst',[(x['query'],x['dt_pp'][a])for x in sorted(rows,key=lambda x:x['dt_pp'][a])[:4]])
if __name__=='__main__':main()
