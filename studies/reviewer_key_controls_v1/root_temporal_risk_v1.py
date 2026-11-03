"""Separate native-readout risk from the actual visual-posterior update."""
from common_v1 import *
from statistics_v1 import pair_stats
import numpy as np
def run(ds):
 dest=D/'temporal'/ds;aud=read(dest/'reporting_receipt.json');assert aud['metrics_sha256']==sha(dest/'metrics.json')and aud['all_arrays_cold_rederived_exact'];rr=read(dest/'metrics.json')['private_rows'];out={}
 comparisons=[('SourceJointMAP','Source'),('SourceTR','SourceJointMAP'),('ExpertInitTR','SourceJointMAP'),('ExpertInitTR','SourceTR')]
 for baseline in ['Source','SourceJointMAP']:
  for lo,hi,label in [(0,.2,'0_02'),(.2,.5,'02_05'),(.5,.8,'05_08'),(.8,1.00000000001,'08_10')]:
   rows0=[r for r in rr if lo<=r['values'][baseline]['t']<hi];groups={}
   for a,b in comparisons:
    v=pair_stats(rows0,a,b);changes=[100*(r['values'][a]['t']-r['values'][b]['t'])for r in rows0];v['exact_t_counts']=dict(improved=sum(x>1e-10 for x in changes),harmed=sum(x< -1e-10 for x in changes),equal=sum(abs(x)<=1e-10 for x in changes),harmed_over5=sum(x< -5 for x in changes));groups[a+'-'+b]=v
   out[baseline+'/'+label]=groups
 p=P/'temporal_evidence'/ds;atomic(p/'readout_vs_visual_update_risk.json',dict(groups=out,definition='SourceJointMAP-Source isolates the readout change. SourceTR-SourceJointMAP and ExpertInitTR-SourceJointMAP use the same legal jointMAP decoder and isolate appearance posterior updates. ExpertInitTR-SourceTR tIoU compares the two evidence regions; vIoU for that pair also changes source/E boxes, so use ExpertInitTR-FixedE_SourceTR for fixed-space v attribution.',labels_only_offline=True,query_equal_parent_CI=True,new_model=0,new_GT_scoring=0))
 from full_score import cluster
 for name,g in out.items():
  base,label=name.split('/');bounds={'0_02':(0,.2),'02_05':(.2,.5),'05_08':(.5,.8),'08_10':(.8,1.00000000001)};lo,hi=bounds[label];rows0=[r for r in rr if lo<=r['values'][base]['t']<hi]
  if not rows0:continue
  for a,b in comparisons:
   old=cluster(rows0,a,b,[('t',100)])['t'];new=g[a+'-'+b]['statistics']['delta_t_pp'];assert abs(old['mean']-new['mean'])<1e-12 and np.allclose(old['CI95'],new['CI95'],rtol=0,atol=1e-12)
 atomic(p/'readout_risk_supplement_receipt.json',dict(status='root_same_readout_temporal_risk_supplement_independent_cluster_check',code_sha256=sha(Path(__file__)),prediction_audit_sha256=sha(dest/'reporting_receipt.json'),report_sha256=sha(p/'readout_vs_visual_update_risk.json'),worker=identity(),new_model=0,new_GT_scoring=0));print(ds,'readout-controlled risk exported')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=COUNTS,required=True);run(ap.parse_args().dataset)
