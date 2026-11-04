"""Root scalar-only verification after whole-domain independent audits; no GT read."""
from pathlib import Path
import os,json,hashlib,time,csv
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
R=Path(__file__).resolve().parents[3];D=R/'outputs/shared_time_spatial_recalc_v2';P=R/'results/shared_time_spatial_recalc_v2'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def boot(rows,a,b,key):
 parents=sorted({r['parent']for r in rows});index={p:i for i,p in enumerate(parents)};s=np.zeros(len(parents));n=s.copy();delta=[]
 for r in rows:
  x,y=r['values'][a][key],r['values'][b][key]
  if x is None or y is None:continue
  z=100*(x-y);j=index[r['parent']];s[j]+=z;n[j]+=1;delta.append(z)
 rng=np.random.default_rng(20261022);w=rng.multinomial(len(parents),[1/len(parents)]*len(parents),size=10000);den=w@n;v=(w@s)[den>0]/den[den>0]
 return dict(mean=float(np.mean(delta)),CI95=np.quantile(v,[.025,.975]).tolist(),queries=len(delta))
def main():
 start=time.perf_counter();allreports={};verified=0;exits={f.stem:read(f)for f in (D/'exits').glob('*.json')};assert len(exits)==6 and all(x['exit_code']==0 and x['actual_subprocess_wait']for x in exits.values())
 # The original tool session is no longer addressable after context handover.
 # Preserve that unknown outer wait; all six real child waits and final state exist.
 outer=dict(tool_session=37292,actual_tool_wait=False,exit_code=None,reason='write_stdin returned Unknown process id after context handover; no inferred exit code',controller_process_absent=not Path('/proc/2315029').exists(),successful_final_state=read(D/'state.json')['status'])
 write(D/'controller_external_wait_unavailable.json',outer)
 for ds,count in [('hc2',1902),('vidstg',10303)]:
  audit=read(D/ds/'audit_receipt.json');score=read(D/ds/'score_receipt.json');report=read(P/ds/'report.json');rows=read(D/ds/'metrics.json')['private_rows'];assert len(rows)==count==audit['queries'];assert audit['report_sha256']==sha(P/ds/'report.json')and audit['metrics_sha256']==sha(D/ds/'metrics.json')and score['seal_sha256']==sha(D/ds/'sealed.json')
  checks={};tails={}
  for label,rr,table in [('all',rows,report['paired'])]+[(g,[r for r in rows if r['duration_group']==g],x['paired'])for g,x in report['timegroups'].items()]:
   checks[label]={}
   for a,b in [('Full','PureExpertLinear'),('Full','SharedCorrection')]:
    pair=a+'-'+b;checks[label][pair]={}
    for key in ['t','v','event_unobserved_IoU','event_observed_IoU']:
     z=boot(rr,a,b,key);ref=table[pair][key];assert z['queries']==ref['queries']and abs(z['mean']-ref['mean'])<1e-10 and np.max(abs(np.array(z['CI95'])-ref['CI95']))<1e-10;verified+=1;checks[label][pair][key]=z
    delta=np.array([[100*(r['values'][a][k]-r['values'][b][k])for k in ['t','v']]for r in rr]);v=delta[:,1];loss=-v[v< -1e-10]
    tails[label+'/'+pair]=dict(queries=len(rr),improved=int((v>1e-10).sum()),unchanged=int((abs(v)<=1e-10).sum()),harmed=int((v< -1e-10).sum()),severe_tv_union_once_gt5pp=int((delta< -5).any(1).sum()),rescued_tv_union_once_gt5pp=int((delta>5).any(1).sum()),v_decline_ge10pp=int((v<=-10).sum()),v_harm_median_pp=float(np.median(loss))if len(loss)else None,v_harm_p90_pp=float(np.quantile(loss,.9))if len(loss)else None)
  arms=['PureExpertLinear','SharedCorrection','Full'];support={}
  for r in rows:
   for k in ['t','event_GT_frames','event_observed_frames','event_unobserved_frames']:assert len({r['values'][a][k]for a in arms})==1
  for k in ['event_GT_frames','event_observed_frames','event_unobserved_frames']:support[k]=dict(total=sum(r['values']['Full'][k]for r in rows),queries_with_support=sum(r['values']['Full'][k]>0 for r in rows),queries_without_support=sum(r['values']['Full'][k]==0 for r in rows))
  cost={k:x for k,x in exits.items()if k.startswith(ds+'_')};out=dict(queries=count,parents=report['parents'],means=report['means'],paired_and_group_CIs_independently_recomputed=checks,signed_tails=tails,identical_support_all3=support,cost=cost,scope='Cached CPU reconstruction/replay plus separate official scoring/audit. Prior Source/expert acquisition is nonzero; not deployment timing; concurrent stage times must not be summed as suite elapsed.',audit_sha256=sha(D/ds/'audit_receipt.json'),all_original_Full_arrays_and_vIoU_exact=True,soft_anchor_contract_unchanged=True)
  write(P/ds/'ROOT_report.json',out);allreports[ds]=out
 lines=['# Full shared-time spatial comparison: independent recalculation','', 'Both domains use the complete target lists, raw identical eight expert observations and one immutable Current Full interval per query. All three tIoUs and scored annotated-frame supports are exactly equal per query. PureExpertLinear is raw pixel-corner interpolation on every input video frame with nearest-expert tails. SharedCorrection is freshly reconstructed from Source/raw8 residual consensus. Full reruns the original final DP on unchanged cached candidate/unary/link sufficient state; no new transport fit or unavailable raw grid is claimed.','', '|Dataset|Path|tIoU|vIoU|R.3|R.5|Event-unobserved IoU|','|---|---|---:|---:|---:|---:|---:|']
 for ds,r in allreports.items():
  for arm,m in r['means'].items():lines.append('|'+ds+'|'+arm+'|'+'|'.join(f'{100*m[k]:.6f}'for k in ['t','v','vIoU_ge_03','vIoU_ge_05','event_unobserved_IoU'])+'|')
 lines+=['','HC Full exceeds pure interpolation by +1.756325 pp vIoU; Vid Full is lower by −0.246251 pp. Full exceeds direct shared-corrected E by +0.859295/+0.609417 pp. Different dataset outcomes do not establish sampling imbalance: every within-domain comparison has the exact same complete query list and interval/support. The corrected numbers do not turn Vid into a positive spatial result.','', 'Event-unobserved IoU uses the same dense annotated frames inside common predicted∩GT event, excluding exact eight expert times. HC 1732/1902 and Vid 8241/10303 queries have nonempty support; empty-support conditional values are null, while all queries remain in primary vIoU. This replaces the previous incompatible dense-pure versus sampled-native unobserved comparison; it must not be compared numerically with all-video/native-grid means.','', 'The old baseline wrappers had two distinct issues already preserved: incomplete dense input tails and a historical Source_interval alias corresponding to MBR rather than native Source. This controlled table explicitly binds actual Full_interval and needs neither Source time nor the proposed temporal branch for its interpolation path. The raw-corner interpolation kernel contains no E/transport/DP.','', 'Full still allows E instead of raw expert boxes at observed frames. That unresolved hard-anchor design/implementation contract is not repaired here; output equality at anchors is not a property of this comparison. Numeric reproduction does not validate an intended hard-anchor method.','', 'All signed changes, severe tails, short/long groups and pointwise 10000 parent-cluster CIs are in ROOT_report.json and the scalar TSVs. Cohorts were exposed before this audit; there is no parameter selection or equivalence claim. 36615 official arm/query checks, 97640 independent array checks and all six actual child wait receipts passed. Outer tool37292 became unavailable after context handover, so its external wait/exit is explicitly unknown rather than fabricated zero. All Source/expert/GPU/model/optimizer calls in this recalculation are zero; inherited acquisition costs are nonzero.','']
 (P/'ANALYSIS.md').write_text('\n'.join(lines));write(P/'README.md.json',dict(status='complete_full_scored_audited_ROOT_interpreted_pending_publication',new_Source=0,new_expert=0,new_GPU=0,no_Full_method_fix=True))
 write(D/'ROOT_completion.json',dict(status='complete_full_scored_audited_ROOT_interpreted_PENDING_publication',queries=12205,official_checks=36615,all6actualchild_wait0=True,outer_actual_wait=outer,CI_independent_checks=verified,root_reports_sha256={ds:sha(P/ds/'ROOT_report.json')for ds in allreports},analysis_sha256=sha(P/'ANALYSIS.md'),audit_sha256={ds:sha(D/ds/'audit_receipt.json')for ds in allreports},elapsed_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GPU=0))
 print(json.dumps(dict(status='ROOT_export_complete',CI_checks=verified)))
if __name__=='__main__':main()
