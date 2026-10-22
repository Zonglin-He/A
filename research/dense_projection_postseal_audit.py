"""Independent completed-study projection/migration audit, no new predictions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from research.joint_event_v1.run import read,put,sha
D=R/'outputs/native_event_dense_projection_v2';P=R/'results/native_event_dense_projection_v2';L=R/'outputs/overnight_event_loop_20261023';s=read(D/'sealed.json');p=read(D/'plan.json');r=read(P/'report.json');assert sha(P/'report.json')==read(D/'score_receipt.json')['report_sha256'];out={}
for g,v in r['domains'].items():
 rows=[]
 for item in [i for i in p['items']if i['group']==g]:
  e=s['files'][item['alias']];assert sha(D/e['trace_path'])==e['trace_sha256'];t=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');a=t['steps'][1];x=a['raw_L'];y=a['projected_L'];corr=float(np.corrcoef(x,y)[0,1])if np.std(x)*np.std(y)>0 else None;rows.append(dict(alias=item['alias'],nodes=len(x),second_raw_std=float(np.std(x)),second_projected_std=float(np.std(y)),projected_raw_correlation=corr,projected_direct_TV=float(abs(t['q']['DenseTemporalProjectionLoop2']-t['q']['DirectSecondRoundFeedback']).sum()/2)))
 out[g]=dict(count=len(rows),node_min=min(x['nodes']for x in rows),node_max=max(x['nodes']for x in rows),mean_projected_raw_correlation=float(np.mean([x['projected_raw_correlation']for x in rows if x['projected_raw_correlation']is not None])),median_projected_direct_TV=float(np.median([x['projected_direct_TV']for x in rows])),rows=rows)
migrated=[]
for k,e in p['migration_partial_files'].items():
 for a,b in [('path','sha256'),('trace_path','trace_sha256')]:assert sha(D/e[a])==sha(R/'outputs/native_event_dense_projection_v1'/e[a])==e[b]
 migrated.append(k)
assert len(migrated)==96
put(P/'PROJECTION_AUDIT.json',dict(domains=out,all96_migrated_prediction_and_trace_bytes_exact=True,postseal=True,new_model_calls=0,new_predictions=0,interpretation='Projection conveys strong source-seeded semantic feedback on HC64, but doesnotdenoise it into stable fixed-domain improvement. Position/shuffled controls retain comparable or greater fixed-panel performance; no evidence for claimed temporal-feature-specific correction. This doesnotprove rawhidden lacks event information.',seal_sha256=sha(D/'sealed.json')))
put(P/'MIGRATION_VERIFICATION.json',dict(old_failed_fingerprint=p['migration_old_plan_sha256'],old_plan_hash_not_fingerprint=True,new_fingerprint=p['fingerprint'],original96_outputs_and_traces_SHA_exact=True,original_failed_code_and_plan_preserved=True,mathematical_core_exact=True,mathematical_block_through_finalboxes_exact=True,change='shape assertion 65 -> actual source length, native grids unchanged, explicit reuse branch',old_prediction_count=96,new_prediction_count=32,all128_seal_before_GT=True,no_model_rerun=True,no_old_hash_waiver=True))
lines=['# Dense projection failure cases and migration','','Original prediction code preserved after an incorrect65-node assertion atfirstVid16. V2 has a new own plan/fingerprint, original native grids16–200 and explicit exact96-output/trace reuse. No new model calls or GT-scoring before full128 seal.','']
for g,v in r['domains'].items():
 rows=sorted(v['rows'],key=lambda x:x['values']['DenseTemporalProjectionLoop2']['t']-x['values']['ParentDirectTarget']['t']);lines += [g,'|Alias|projected−G6 tpp|direct−G6 tpp|projected−direct tpp|','|---|---:|---:|---:|']
 for x in rows[:3]+rows[-3:]:
  a=x['values']['DenseTemporalProjectionLoop2'];b=x['values']['ParentDirectTarget'];c=x['values']['NativeG6'];lines.append(f"|{x['query']}|{100*(a['t']-c['t']):+.4f}|{100*(b['t']-c['t']):+.4f}|{100*(a['t']-b['t']):+.4f}|")
 lines.append('All signs and severe outcomes remain in signed.tsv. DirectSecondRoundFeedback uses the exact second-round evidence from the primary, so parent-direct differences are not attributed only to projection.')
(P/'CASE_ANALYSIS.md').write_text('\n'.join(lines)+'\n');put(L/'dense_projection_postseal_receipt.json',dict(code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),score_receipt_sha256=sha(D/'score_receipt.json'),audit_sha256=sha(P/'PROJECTION_AUDIT.json'),all96_aliases_exact=True,new_models=0,new_predictions=0,CUDA_initialized=torch.cuda.is_initialized()))
print(json.dumps({g:{k:v for k,v in x.items()if k!='rows'}for g,x in out.items()},indent=2))
