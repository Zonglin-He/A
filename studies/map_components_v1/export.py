"""Post-audit paper tables/curves from already scored scalar results only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,csv,argparse,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'outputs/map_components_v1';P=R/'results/map_components_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def export(task,ds):
 receipt=D/f'{task}_{ds}_reporting_receipt.json';assert receipt.exists(),'Whole task/domain score and exact replay audit required'
 proof=read(receipt);score=D/f'{task}_{ds}_score_receipt.json';assert proof['score_sha256']==sha(score)and proof['all_arrays_cold_replayed_exact']and proof['all_means_recomputed']
 folder=P/task/ds;report=folder/'report.json';res=read(report);made=[]
 if task=='components':
  path=folder/'table3.tsv'
  with path.open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['configuration','queries','vIoU_percent','R_03_percent','R_05_percent','delta_v_vs_original_Source_pp'])
   for a in ['Source','EightReplacement','SharedCorrection','SpatialInference','FullMAP']:
    m=res['means'][a];w.writerow([a,res['queries'],100*m['v'],100*m['vIoU_ge_03'],100*m['vIoU_ge_05'],100*(m['v']-res['means']['Source']['v'])])
  made.append(path)
  path=folder/'final_DP_control.json';path.write_text(json.dumps(dict(queries=res['queries'],fixed='Same candidates, validity, auxiliary path, q1 and final MAP interval.',DP=res['means']['FullMAP'],independent=res['means']['FinalIndependent'],paired=res['paired']['FullMAP-FinalIndependent'],no_new_GT_scoring=True),indent=2)+'\n');made.append(path)
 else:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  with(folder/'budget_curve.tsv').open()as f:rr=list(csv.DictReader(f,delimiter='\t'))
  fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
  for schedule in ['Hybrid','Uniform']:
   rows=sorted([r for r in rr if r['schedule']==schedule],key=lambda r:int(r['nominal_K']))
   axes[0].plot([int(r['nominal_K'])for r in rows],[float(r['vIoU_percent'])for r in rows],marker='o',label=schedule)
   axes[1].plot([float(r['mean_original_expert_forward_seconds'])for r in rows],[float(r['vIoU_percent'])for r in rows],marker='o',label=schedule)
  axes[0].set(xlabel='Nominal observation budget K',ylabel='vIoU (%)',xticks=[2,4,8,16]);axes[1].set(xlabel='Original expert forward time (s/query)',ylabel='vIoU (%)')
  for ax in axes:ax.grid(alpha=.25);ax.legend()
  fig.suptitle(f'{ds}: fixed confirmation cohort, MAP + recomputed spatial DP')
  path=folder/'budget_curve.svg';fig.savefig(path,metadata={'Date':None});made.append(path)
  private=D/'figures';private.mkdir(exist_ok=True);fig.savefig(private/f'{ds}_budget_curve.png',dpi=180);fig.savefig(private/f'{ds}_budget_curve.pdf');plt.close(fig)
 result=dict(status='post_audit_scalar_export_complete',queries=res['queries'],task=task,dataset=ds,report_sha256=sha(report),audit_sha256=sha(receipt),files={p.name:sha(p)for p in made},new_model=0,new_GT_scoring=0)
 (folder/'paper_export_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',required=True,choices=['components','budget']);a.add_argument('--dataset',required=True,choices=['hc2','vidstg']);v=a.parse_args();export(v.task,v.dataset)
