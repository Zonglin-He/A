"""ROOT-only completed-scalar review; no GT/media/models/forward or re-scoring."""
from pathlib import Path
import sys,argparse,json,hashlib,time,os
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/expert_evidence_controls_v1'))
from common import D,P,read,sha,atomic,identity,np,check,rows,ARMS
def direct_ci(rr,field):
 values=[];parents=[]
 for r in rr:
  a,b=r['values']['Full'][field],r['values']['SAM2MatchedMask'][field]
  if a is not None and b is not None:values.append(100*(a-b));parents.append(r['parent'])
 if not values:return dict(mean=None,CI95=None,queries=0)
 values=np.asarray(values);unique=sorted(set(r['parent']for r in rr));s=np.array([sum(values[i]for i,p in enumerate(parents)if p==q)for q in unique]);c=np.array([parents.count(q)for q in unique]);w=np.random.default_rng(20261022).multinomial(len(unique),[1/len(unique)]*len(unique),10000);den=w@c;boot=(w@s)[den>0]/den[den>0]
 return dict(mean=float(values.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),queries=len(values))
def main(ds):
 check();dest=D/'propagation'/ds;pub=P/'propagation'/ds;ack=read(dest/'reporting_receipt.json');export=read(dest/'export_receipt.json');assert ack['report_sha256']==sha(pub/'report.json')and export['audit_sha256']==sha(dest/'reporting_receipt.json')
 rr=read(dest/'metrics.json')['private_rows'];report=read(pub/'report.json');assert len(rr)==128 and len({x['parent']for x in rr})==128
 checks=0
 for arm in ARMS:
  for field,expected in report['means'][arm].items():
   vv=[r['values'][arm][field]for r in rr if r['values'][arm][field]is not None];actual=float(np.mean(vv))if vv else None
   assert actual==expected;checks+=1
 for group in ['all','short_le5s','long_gt5s']:
  subset=[r for r in rr if group=='all'or r['duration_group']==group]
  for field,key in [('v','delta_v_pp'),('t','delta_t_pp'),('event_unobserved_IoU','delta_event_unobserved_IoU_pp'),('event_observed_IoU','delta_event_observed_IoU_pp')]:
   actual=direct_ci(subset,field);expected=report['paired'][group]['statistics'].get(key)
   if expected is None:assert not subset;continue
   for k in ['mean','CI95']:
    if actual[k]is None:assert expected[k]is None
    else:assert np.allclose(actual[k],expected[k],rtol=0,atol=1e-11)
   assert actual['queries']==expected['queries'];checks+=1
 tails=[]
 for i,r in enumerate(rr,1):
  dt=100*(r['values']['Full']['t']-r['values']['SAM2MatchedMask']['t']);dv=100*(r['values']['Full']['v']-r['values']['SAM2MatchedMask']['v']);tails.append(dict(query=f'Q{i:05}',duration_group=r['duration_group'],delta_v_Full_minus_SAM2_pp=dv,delta_t_pp=dt,severe_t=dt< -5,severe_v=dv< -5,severe_union_once=dt< -5 or dv< -5,rescue_t=dt>5,rescue_v=dv>5,rescue_union_once=dt>5 or dv>5))
 atomic(pub/'ROOT_all_signed_tails.json',dict(rows=tails,all128_retained=True))
 cost=read(pub/'all_actual_query_costs.json');empty=read(pub/'cost_empty_summary.json');load=read(D/'propagation/model_load.json');root=dict(status='ROOT_whole128_all_signed_shortlong_tails_support_empty_cost_independently_reviewed',dataset=ds,queries=128,parents=128,independent_scalar_and_parentCI_checks=checks,all_time_intervals_equal=True,report_sha256=sha(pub/'report.json'),metrics_sha256=sha(dest/'metrics.json'),reporting_receipt_sha256=sha(dest/'reporting_receipt.json'),export_receipt_sha256=sha(dest/'export_receipt.json'),model_load_unique_PIDstart=load['worker'],one_shared_load_seconds=load['seconds'],new_Source=0,new_ReferDINO=0,new_optimizer=0,new_GT=0,old_Source_expert_acquisition_cost_nonzero=True,old_Source_acquisition_seconds_not_reconstructed=None,experimental_clocks_include_provenance_IO=True,nested_clocks_not_additive=True,not_deployment_latency=True,publication_complete=False,worker=identity(),time=time.time())
 allpair=report['paired']['all']['statistics'];v=allpair['delta_v_pp'];u=allpair['delta_event_unobserved_IoU_pp'];obs=allpair['delta_event_observed_IoU_pp']
 def direction(s):
  if s['CI95']is None:return 'unscorable'
  return 'positive CI'if s['CI95'][0]>0 else'negative CI'if s['CI95'][1]<0 else'CI crosses zero; direction unresolved'
 text=[f'# ROOT review: {ds}, original128 parents, matched ReferDINO mask evidence','',f'Whole128 predictions, exact teacher/RGB/input/Fulltime audit, official tube scores, separate normalized pixel-IoU/SHA/means/CI audit and signed export completed. ROOT independently recomputed {checks} mean/CI checks from the sealed scalar rows.','', '|Spatial strategy|tIoU|vIoU|R.3|R.5|Conditional dense unobserved IoU|Conditional observed IoU|','|---|---:|---:|---:|---:|---:|---:|']
 for a in ARMS:text.append('|'+a+'|'+'|'.join('null'if report['means'][a][k]is None else f"{100*report['means'][a][k]:.6f}"for k in ['t','v','R03','R05','event_unobserved_IoU','event_observed_IoU'])+'|')
 text+=['',f"Full minus SAM2 vIoU {v['mean']:+.6f} pp,95%CI {v['CI95']}: {direction(v)}.",f"Conditional unobserved difference {u['mean']} pp,95%CI {u['CI95']}; observed difference {obs['mean']} pp,95%CI {obs['CI95']}.",'',f"Full v improves {sum(t['delta_v_Full_minus_SAM2_pp']>1e-10 for t in tails)}, harms {sum(t['delta_v_Full_minus_SAM2_pp']< -1e-10 for t in tails)}, unchanged {sum(abs(t['delta_v_Full_minus_SAM2_pp'])<=1e-10 for t in tails)} of128. Severe t/v declines uniononce {sum(t['severe_union_once']for t in tails)}; rescues uniononce {sum(t['rescue_union_once']for t in tails)}."]
 for group in ['short_le5s','long_gt5s']:
  q=report['paired'][group];text.append(f"{group}: {q['queries']} queries; Full-SAM2 statistics {json.dumps(q['statistics'],sort_keys=True)}")
 text+=['',f"Empty/input/cost: {json.dumps(empty,sort_keys=True)}",f"One model load shared by both domains: {load['seconds']:.6f}s at PID/start {load['worker']}; do not add to every query or sum nested clocks. Source and ReferDINO acquisition is inherited, not zero; no claim of deployment-speed superiority.",'','Exact same original eight ReferDINO selected masks and physicaltimes, same Source predictions/roster and common Full event. SAM2.1 Hiera-L encodes the full sampled video and adds a model. This controls evidence and time, not representation/knowledge/model count/compute. Full remains the unmodified soft-anchor implementation; this does not validate an intended hard-anchor variant. No target configuration selection; panel already exposed.','', 'The overall paired difference does not separately identify shared-correction, bidirectional transport and DP contributions. The completed candidate×decoder controls retain that role. All negative outcomes, empty masks and unsupported conditional queries remain. No automatic full expansion or old Source-prompt full resumption.']
 (pub/'ROOT_ANALYSIS.md').write_text('\n'.join(text)+'\n');atomic(dest/'ROOT_receipt.json',root);atomic(pub/'ROOT_receipt.json',root);print(ds,json.dumps(dict(v=v,unobserved=u,independent_checks=checks)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['hc2','vidstg'],required=True);main(a.parse_args().dataset)
