"""Separate postseal numerical audit; no new predictions, GT reads or model calls."""
import sys,json,os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research.joint_event_v1.run import read,put,sha
import torch,numpy as np
from sklearn.covariance._shrunk_covariance import _oas

def entropy(q):
 v=q[q>0];return float(-(v*np.log(v)).sum())
def main():
 torch.set_num_threads(1);D=R/'outputs/native_event_covariance_v1';P=R/'results/native_event_covariance_v1';plan=read(D/'plan.json');s=read(D/'sealed.json');report=read(P/'report.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');rows=[];maxerr=0.
 for item in plan['items']:
  k=item['alias'];e=s['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');root=R/'outputs'/('native_event_semantic_prototype_hc64_v1'if item['source']=='first'else'native_event_semantic_recheck_v1');pe=item['entry'];assert sha(root/pe['trace_path'])==pe['trace_sha256'];old=torch.load(root/pe['trace_path'],weights_only=False,map_location='cpu');p=old['semantic_p']['motion']if item['source']=='first'else old['semantic_p'];prior=tr['q']['NativeG6'];base_steps=old['prototype_steps']['real']if item['source']=='first'else old['prototype_steps'];a=tr['steps'][0];ratios=[];rhos=[];conditions=[]
  for i,st in enumerate(a['statistics']):
   ref,rho=_oas(p[np.arange(len(p))%2!=i]);maxerr=max(maxerr,float(abs(ref-st['covariance']).max()));rhos.append(st['shrinkage']);conditions.append(float(np.linalg.cond(st['covariance'])))
  l0=float(np.std(base_steps[0]['L']));L=float(np.std(a['L']));rows.append(dict(query=k,group=item['group'],class_count=p.shape[1],median_OAS_shrinkage=float(np.median(rhos)),median_covariance_condition=float(np.median(conditions)),firststep_frame_evidence_std=L,parent_Bernoulli_firststep_std=l0,evidence_std_ratio=L/l0 if l0 else None,TV=float(abs(tr['q']['CovariancePrototypeLoop2']-prior).sum()/2),max_posterior_mass=float(tr['q']['CovariancePrototypeLoop2'].max()),Native_entropy=entropy(prior),Covariance_entropy=entropy(tr['q']['CovariancePrototypeLoop2']),Parent_entropy=entropy(tr['q']['ParentBernoulli'])))
 out=dict(scope='POSTSEALnumericalaudit usingsealedfeaturesandoutputs only;0newGT/0model/0predictions',rows=rows,max_real_covariance_reference_error=maxerr,summary={})
 for g in ['HC64','hc2_recheck','vidstg_recheck']:
  rr=[x for x in rows if x['group']==g];out['summary'][g]={m:float(np.median([x[m]for x in rr if x[m]is not None]))for m in rr[0]if m not in ['query','group']}
 put(P/'NUMERICAL_AUDIT.json',out);put(D/'numerical_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),output_sha256=sha(P/'NUMERICAL_AUDIT.json'),new_model_calls=0,new_predictions=0,new_GT_reads=0));print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
