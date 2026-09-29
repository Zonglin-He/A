"""Post-hoc readout scale from sealed scalars; no new updates or inference."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write
from scripts.run_tastvg_sparse_online_capture_v1 import OUT

def run():
    rows=[r for r in read(OUT/'analysis/ROWS.json') if not r['expert']];stats=[]
    for r in rows:
        base=np.array(r['base_score']);arr=np.array(r['arrival_scores']);e=np.array(r['teacher_scores']);k=r['selected']['Full Rerank'];pp=[(i,j) for i in range(len(e)) for j in range(len(e)) if e[i]>e[j]+1e-12]
        agree=lambda ss:float(np.mean([1. if ss[i]>ss[j]+1e-12 else .5 if abs(ss[i]-ss[j])<=1e-12 else 0. for i,j in pp])) if pp else None
        stats.append(dict(position=r['position'],full_selected=k,native_score_gap=float(base[0]-base[k]),inherited_residual_advantage=float((arr[k]-base[k])-(arr[0]-base[0])),native_expert_pair_agreement=agree(base),online_expert_pair_agreement=agree(arr)))
    non=[r for r in stats if r['full_selected']!=0];aa=[r for r in stats if r['native_expert_pair_agreement'] is not None]
    z=dict(posthoc_existing_outputs_only=True,no_new_inference_or_updates=True,nonexpert=24,full_prefers_nonnative=len(non),inherited_residual_favors_that_candidate=sum(r['inherited_residual_advantage']>0 for r in non),median_native_gap=float(np.median([r['native_score_gap'] for r in non])),median_residual_advantage=float(np.median([r['inherited_residual_advantage'] for r in non])),mean_native_pair_agreement=float(np.mean([r['native_expert_pair_agreement'] for r in aa])),mean_online_pair_agreement=float(np.mean([r['online_expert_pair_agreement'] for r in aa])),rows=stats)
    path=OUT/'analysis/READOUT_DIAGNOSTIC.json'
    if path.exists():assert read(path)==z
    else:write(path,z)
    print({k:v for k,v in z.items() if k!='rows'})

if __name__=='__main__':run()
