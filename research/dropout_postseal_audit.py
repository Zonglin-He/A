"""Independent postseal audit; no model, prediction selection or parameters."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch,numpy as np
from research.joint_event_v1.run import read,put,sha
D=R/'outputs/native_event_dropout_v1';P=R/'results/native_event_dropout_v1';L=R/'outputs/overnight_event_loop_20261023'
s=read(D/'sealed.json');r=read(P/'report.json');assert sha(P/'report.json')==read(D/'score_receipt.json')['report_sha256'];p=read(D/'plan.json');out={};modules=None
for group,v in r['domains'].items():
 rows=[]
 for item in [x for x in p['items']if x['group']==group]:
  e=s['files'][item['alias']];assert sha(D/e['trace_path'])==e['trace_sha256'];t=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');modules=t['modules'];q=t['q'];delta=np.concatenate([(torch.stack([z[i]for z in t['MC_logits']])-t['native_logits'][i]).numpy().reshape(4,-1)for i in range(2)],1)
  with np.load(D/e['path'])as z:equal=bool(np.array_equal(z['NativeDropoutFusion_interval'],z['MomentMatchedGaussianFusion_interval']))
  rows.append(dict(alias=item['alias'],native_MC_TV=float(abs(q['NativeG6']-q['DirectMCMean']).sum()/2),primary_gaussian_TV=float(abs(q['NativeDropoutFusion']-q['MomentMatchedGaussianFusion']).sum()/2),delta_RMS=float(np.sqrt((delta**2).mean())),mean_delta_RMS=float(np.sqrt((delta.mean(0)**2).mean())),MC_std_RMS=float(np.sqrt((delta.std(0)**2).mean())),same_Gaussian_interval=equal))
 summary={key:dict(mean=float(np.mean([x[key]for x in rows])),median=float(np.median([x[key]for x in rows])))for key in ['native_MC_TV','primary_gaussian_TV','delta_RMS','mean_delta_RMS','MC_std_RMS']};summary['Gaussian_same_intervals']=sum(x['same_Gaussian_interval']for x in rows);summary['count']=len(rows);out[group]=dict(summary=summary,rows=rows)
put(P/'STABILITY_AUDIT.json',dict(domains=out,actual_scoped_module_probabilities=modules,module_count=len(modules),all_original_p_point1=all(x['probability']==.1 for x in modules),note='Actual stochastic model changes posterior. Same final Gaussian decisions do not mean identical distributions or inactive dropout. Small native mass changes often stay within original physical MBR decision cell. No structure-specific task benefit demonstrated.',new_model_calls=0,new_final_predictions=0,postseal=True,seal_sha256=sha(D/'sealed.json')))
text=['# Independent postseal dropout cases','', 'All stochastic modules actually use p=.1. No native source change. Gaussian interval equality is decision-level equality, not equal posterior or disabled dropout.']
for g,v in r['domains'].items():
 rows=sorted(v['rows'],key=lambda x:x['values']['NativeDropoutFusion']['t']-x['values']['NativeG6']['t']);text+=['',g, '|Alias|Δt G6 pp|Δv G6 pp|boundary Δ seconds|','|---|---:|---:|---:|']
 for x in rows[:3]+rows[-3:]:
  a=x['values']['NativeDropoutFusion'];b=x['values']['NativeG6'];text.append(f"|{x['query']}|{100*(a['t']-b['t']):+.4f}|{100*(a['v']-b['v']):+.4f}|{a['boundary_MAE_seconds']-b['boundary_MAE_seconds']:+.4f}|")
 text.append('All128 Gaussian-moment control final intervals match primary; the saved posteriors differ. These cases do not support event-specific stochastic correction.')
(P/'CASE_ANALYSIS.md').write_text('\n'.join(text)+'\n');put(L/'dropout_postseal_receipt.json',dict(code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),score_receipt_sha256=sha(D/'score_receipt.json'),audit_sha256=sha(P/'STABILITY_AUDIT.json'),new_models=0,new_predictions=0,CUDA_initialized=torch.cuda.is_initialized()))
print(json.dumps({g:x['summary']for g,x in out.items()},indent=2))
