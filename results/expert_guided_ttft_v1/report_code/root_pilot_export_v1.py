"""ROOT scalar/trace/cost interpretation; no inference, updates or new GT reads."""
from pathlib import Path
import os,sys,time,csv,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[3];S=R/'studies/expert_guided_ttft_v1';sys.path[:0]=[str(S),str(R),str(R/'tools')]
import target_pilot_v1 as t
def cluster(rows,fields):
 if not rows:return dict(queries=0,parents=0,statistics={})
 parents=sorted({r['parent']for r in rows});ix={p:i for i,p in enumerate(parents)};n=len(parents);s=np.zeros((n,len(fields)));counts=np.zeros_like(s)
 for r in rows:
  for j,k in enumerate(fields):
   if r.get(k)is not None:s[ix[r['parent']],j]+=r[k];counts[ix[r['parent']],j]+=1
 w=np.random.default_rng(20261022).multinomial(n,np.full(n,1/n),10000);den=w@counts;boot=np.divide(w@s,den,out=np.full_like(den,np.nan),where=den>0);out={}
 for j,k in enumerate(fields):
  vals=[r[k]for r in rows if r.get(k)is not None];b=boot[:,j];b=b[np.isfinite(b)];out[k]=dict(mean=float(np.mean(vals))if vals else None,CI95=np.quantile(b,[.025,.975]).tolist()if len(b)else None,queries=len(vals),parents=int((counts[:,j]>0).sum()))
 return dict(queries=len(rows),parents=n,statistics=out)
def paired(rr,a,b):
 out=[]
 for r in rr:
  dt=100*(r['values'][a]['t']-r['values'][b]['t']);dv=100*(r['values'][a]['v']-r['values'][b]['v']);out.append(dict(parent=r['parent'],delta_t_pp=dt,delta_v_pp=dv,improved_v_percent=100.*(dv>1e-10),harmed_v_percent=100.*(dv< -1e-10),severe_t_percent=100.*(dt< -5),severe_v_percent=100.*(dv< -5),severe_union_percent=100.*(dt< -5 or dv< -5),rescue_union_percent=100.*(dt>5 or dv>5)))
 return cluster(out,list(out[0])[1:])if out else cluster([],[])
def main():
 p=t.check();d=t.D;pub=t.P;outer=t.read(d/'controller_actual_exit.json');assert outer['actual_tool_wait']and outer['exit_code']==0;complete=t.read(d/'controller_complete.json');assert len(complete['exits'])==6 and all(e['actual_subprocess_wait']and e['exit_code']==0 for e in complete['exits'].values());both={};costs=[];traces=[];checks=0
 for ds in t.c.TARGETS:
  ack=t.read(d/ds/'reporting_receipt.json');assert ack['metrics_sha256']==t.sha(d/ds/'metrics.json')and ack['report_sha256']==t.sha(pub/ds/'report.json');rr=t.read(d/ds/'metrics.json')['private_rows'];rep=t.read(pub/ds/'report.json');assert len(rr)==128 and len({r['parent']for r in rr})==128
  for a in t.ARMS:
   for field,want in rep['means'][a].items():
    vals=[r['values'][a][field]for r in rr if r['values'][a][field]is not None];got=float(np.mean(vals))if vals else None;assert (got is None and want is None)or abs(got-want)<1e-12;checks+=1
  for group,pairs in rep['paired_by_duration'].items():
   qq=[r for r in rr if group=='all'or r['duration_group']==group]
   for name,want in pairs.items():
    a,b=name.split('-');got=paired(qq,a,b);assert got['queries']==want['queries']and got['parents']==want['parents']
    for f,w in want['statistics'].items():
     g=got['statistics'][f];assert abs(g['mean']-w['mean'])<2e-12 and np.max(np.abs(np.asarray(g['CI95'])-w['CI95']))<2e-12;checks+=1
  newpairs={};tails={}
  for a,b in [('TTFT','Current'),('TTFT','SourceTR'),('TTFT','Source')]:
   pp=paired(rr,a,b);unobs=[dict(parent=r['parent'],delta_U_pp=100*(r['values'][a]['unobserved_actual_sIoU']-r['values'][b]['unobserved_actual_sIoU']))for r in rr if r['values'][a]['unobserved_actual_sIoU']is not None and r['values'][b]['unobserved_actual_sIoU']is not None];pp['unobserved']=cluster(unobs,['delta_U_pp']);newpairs[a+'-'+b]=pp
   dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]);dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);tails[a+'-'+b]=dict(improved_v=int((dv>1e-10).sum()),unchanged_v=int((abs(dv)<=1e-10).sum()),harmed_v=int((dv< -1e-10).sum()),severe_v_gt5pp=int((dv< -5).sum()),rescue_v_gt5pp=int((dv>5).sum()),severe_v_ge10pp=int((dv<=-10).sum()),severe_t_or_v_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_t_or_v_union_once=int(((dt>5)|(dv>5)).sum()))
  domaincost=[];trace=[]
  for i,r in enumerate(rr,1):
   e=t.checked(ds,r['id']);tr=t.read(d/e['trace_path']);assert tr['optimizer_steps']==p['configs'][ds]['steps']and tr['reset_all_parameters_and_Source_boxes_exact']and tr['checkpoint_cold_forward_exact'];assert tr['parameter_delta_L2']>0 and tr['spatial_changed_frames']>0;assert all(np.isfinite(s['loss'])and s['gradient_norm_before_clip']>0 for s in tr['trace'])
   z=dict(dataset=ds,query=f'Q{i:03}',Source_seconds=e['Source_seconds'],query_wall_seconds=e['query_wall_seconds'],shared_update_stream_seconds=tr['shared_stream_total_seconds'],worker=e['worker']);domaincost.append(z);costs.append(z)
   q=dict(dataset=ds,query=f'Q{i:03}',steps=tr['optimizer_steps'],pseudo_loss_before=tr['pseudo_loss_before'],pseudo_loss_after=tr['pseudo_loss_after'],parameter_delta_L2=tr['parameter_delta_L2'],spatial_changed_frames=tr['spatial_changed_frames'],unobserved_changed_frames=tr['unobserved_changed_frames'],sampled_frames=tr['all_sampled_frames'],active_gradient_parameter_tensors=tr['trace'][-1]['gradient_parameter_tensors'],active_gradient_parameter_scalars=tr['trace'][-1]['gradient_parameter_scalars'],reset_exact=True,cold_state_exact=True);trace.append(q);traces.append(q)
  efficiency=dict(queries=128,mean_active_experimental_query_wall_seconds=float(np.mean([x['query_wall_seconds']for x in domaincost])),mean_Source_forward_and_observer_seconds=float(np.mean([x['Source_seconds']for x in domaincost])),mean_conditional_update_stream_seconds=float(np.mean([x['shared_update_stream_seconds']for x in domaincost])),Source_model_load=t.read(d/ds/'model_load.json'),cost_scope='Actual query wall includes reference reads, decoding, transfers, readonly Source observers, raw checkpoints, SHA, adaptation diagnostics/checkpoint cold reads, prototype/readout and prediction commit. Conditional update stream includes transfers/state diagnostics/checkpoints; nested clocks overlap and must not be added. Not audit-free matched deployment timing.')
  both[ds]=dict(queries=128,parents=128,config=p['configs'][ds],means=rep['means'],paired=newpairs,tails=tails,short_long=rep['paired_by_duration'],efficiency=efficiency,optimization_checks=dict(episodes=len(trace),all_actual_gradients_finite_nonzero=True,all_parameter_update_and_allframe_changes_observed=True,all_exact_resets_and_cold_replays=True,pseudo_loss_increased_queries=sum(x['pseudo_loss_after']>x['pseudo_loss_before']for x in trace)),limitations=['Fixed128 exposed parent panel, not full target lists or unknown confirmation.','Configs selected solely on source validation36 per direction before target predictions.','SourceTR has no expert; its contrast includes added expert information and parameter updates, not a same-information adaptation attribution.','Current Full and TTFT use same8raw expert observations, different complete inference/optimization pipelines and own refined intervals. Not isolated propagation comparison.','Original Current soft E/expert anchor choice unchanged; not validation of an intended hard-anchor variant.','No complete matched deployment timing for TTFT measured.'])
  t.atomic(pub/ds/'ROOT_report.json',both[ds])
  with(pub/ds/'all_signed_scalars.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');fields=['v','t','R03','R05','unobserved_actual_sIoU'];w.writerow(['query','parent','duration_group']+[a+'_'+k for a in t.ARMS for k in fields]+['TTFT-Current_v_pp','TTFT-Current_t_pp'])
   pm={x:f'P{i:03}'for i,x in enumerate(sorted({r['parent']for r in rr}),1)}
   for i,r in enumerate(rr,1):w.writerow([f'Q{i:03}',pm[r['parent']],r['duration_group']]+[None if r['values'][a][k]is None else 100*r['values'][a][k]for a in t.ARMS for k in fields]+[100*(r['values']['TTFT'][k]-r['values']['Current'][k])for k in ['v','t']])
  text=['# ROOT target-pilot interpretation','',f'{ds}: original128 queries/128 distinct parents. Both complete source-selected TTFT prediction and saved-state/official/manual/input/SHA/means audits passed.','', '| Method | tIoU | vIoU | R.3 | R.5 | Native unobserved IoU |','|---|---:|---:|---:|---:|---:|']
  for a in t.ARMS:text.append('|'+a+'|'+'|'.join(f"{100*rep['means'][a][k]:.6f}"for k in ['t','v','R03','R05','unobserved_actual_sIoU'])+'|')
  for name,v in newpairs.items():
   x=v['statistics']['delta_v_pp'];text+=['',f"{name}: vIoU {x['mean']:+.6f} pp, pointwise10000 parent-bootstrap95%CI {x['CI95']}. Full signed outcomes and short/long groups retained."]
  text+=['','Unobserved IoU uses GT-supported native sampled unobserved frames over the video, equal query means; not the conditional dense event metric from the candidate-decode experiment.','',*both[ds]['limitations']];(pub/ds/'ANALYSIS.md').write_text('\n'.join(text)+'\n')
  t.atomic(d/ds/'ROOT_receipt.json',dict(status='ROOT_all_absolute_paired_CIs_positive_negative_shortlong_traces_actualcost_interpreted',report_sha256=t.sha(pub/ds/'ROOT_report.json'),official_audit_sha256=t.sha(d/ds/'reporting_receipt.json'),new_Source=0,new_expert=0,new_GT=0,worker=t.identity()))
 t.atomic(pub/'all_actual_query_costs.json',dict(rows=costs,nested_not_additive=True,not_deployment=True,root_bridge_wait=t.read(d/'ROOT_handoff_delay_v1.json'),all6_actual_child_waits=t.read(d/'controller_complete.json')['exits']))
 with(pub/'all_actual_optimizer_diagnostics.tsv').open('w')as f:
  w=csv.DictWriter(f,list(traces[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(traces)
 t.atomic(pub/'ROOT_summary.json',both);t.atomic(d/'ROOT_completion.json',dict(status='both128_TTFT_pilot_scored_audited_ROOT_interpreted_pending_publication_full_target_not_started',queries=256,independent_absolute_and_pair_CI_checks=checks,summary_sha256=t.sha(pub/'ROOT_summary.json'),new_Source=0,new_expert=0,new_GT=0,full_target_complete=False,matched_deployment_timing_complete=False,public_GitHub_synchronized=False,worker=t.identity(),time=time.time()))
 text=['# Expert-guided test-time fine-tuning: fixed128 target panels','', 'Both domains completed sealed predictions, independent GPU replay of all256 adapted weights, official scoring and separate CPU input/feature/interval/manual-score/means audit. ROOT independently reconstructed all absolute means and192 grouped paired fields/CIs. Configurations were selected only on source-domain validation before target prediction.','', '| Domain | Source vIoU | SourceTR vIoU | TTFT vIoU | Current Full vIoU | TTFT-Full pp (95%CI) |','|---|---:|---:|---:|---:|---|']
 for ds,z in both.items():
  q=z['paired']['TTFT-Current']['statistics']['delta_v_pp'];text.append('|'+ds+'|'+'|'.join(f"{100*z['means'][a]['v']:.6f}"for a in t.ARMS)+f"|{q['mean']:+.6f} {q['CI95']}|")
 text+=['','TTFT has higher HC vIoU with a positive pointwise CI relative to Current Full. VidSTG mean is higher but its CI crosses zero. These outcomes do not support claiming Current Full is more accurate than this baseline in both panels. Full target lists and actual matched deployment timing remain unfinished; do not extrapolate pilot ranks or relabel experimental query wall as deployment latency.','', 'The44.263877-minute VidSTG bridge wait was a ROOT handoff delay, not model computation. It remains in raw stage wall and is explicitly separated in cost reporting.','', 'All data/source calibration/target code and positive/negative summaries in this stage are currently LOCAL; no new GitHub publication receipt exists. Original published evidence remains unchanged.'];(pub/'ANALYSIS.md').write_text('\n'.join(text)+'\n')
 print(json.dumps(dict(ROOT_complete=True,scalar_and_CI_checks=checks,summary={ds:z['paired']['TTFT-Current']['statistics']['delta_v_pp']for ds,z in both.items()})),flush=True)
if __name__=='__main__':main()
