"""Source-only semantic-coordinate contribution audit; no candidate/GT scoring."""
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from research.joint_event_v1.run import read,put,sha
from research.native_event_semantic_prototype_v1.core import evidence,membership
A=R/'outputs/target_semantic128_collect_v3';P=R/'results/target_semantic128_shrink_v1'
def main():
 start=time.perf_counter();s=read(A/'sealed.json');plan=read(A/'plan.json');rows=[];error=0
 for item in plan['items']:
  e=s['files'][item['alias']];assert sha(A/e['path'])==e['sha256'];x=torch.load(A/e['path'],map_location='cpu',weights_only=False);p=x['semantic_p']['TargetContext'];q=x['q0'];L,stats=evidence(p,q);terms=np.zeros_like(p)
  for v in stats:
   held=np.arange(len(p))%2==v['heldout_parity'];a,b=v['positive'],v['negative'];terms[held]=p[held]*np.log(a/b)+(1-p[held])*np.log((1-a)/(1-b))
  error=max(error,float(abs(terms.sum(1)-L).max()));z=np.log(p)-np.log1p(-p);relevance=1/(1+np.exp(-z.mean(0)));mass=np.abs(terms).sum(0);low=relevance<.1;rows.append(dict(query=item['alias'],dataset=item['dataset'],split=item['split'],coordinates=len(relevance),fraction_clipconfidence_below_point1=float(low.mean()),fraction_absolute_L_from_lowconfidence=float(mass[low].sum()/mass.sum())if mass.sum()else None,clipconfidence_mean=float(relevance.mean()),clipconfidence_max=float(relevance.max()),clipconfidence_approx_from_clippedp=True))
 assert error<1e-12;out=dict(kind='SOURCEONLY signalcomposition audit_NOTnewcandidate/GT score',code_sha256=sha(Path(__file__)),input_seal_sha256=sha(A/'sealed.json'),max_L_recovery_error=error,summary={ds:{k:float(np.median([x[k]for x in rows if x['dataset']==ds and x[k]is not None]))for k in ['fraction_clipconfidence_below_point1','fraction_absolute_L_from_lowconfidence','clipconfidence_mean','clipconfidence_max']}for ds in ['hc2','vidstg']},rows=rows,new_model=0,new_GT_read=0,new_predictions=0,seconds=time.perf_counter()-start);put(P/'CLASS_CONTRIBUTION_AUDIT.json',out);print(out['summary'])
if __name__=='__main__':main()
