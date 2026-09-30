"""Export sealed anonymous scalar readouts; no new inference or GT access."""
import sys,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
from vg_tta.tastvg_paper48_metrics_v1 import source_summary
from scripts.audit_tastvg_paper48_public_v1 import run as audit

def run(panel):
    assert panel in ['P1','P2','P3_b0','P3_b25','P3_b100','P5']
    b=ROOT/'artifacts/tastvg_paper48_v1'/panel
    assert read(b/'COMPLETION.json')['status']=='completed' and read(b/'AUDIT.json')['status']=='pass'
    rows=read(b/'ROWS.json');metrics=['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT','sIoU_sampled','vIoU_sampled']
    if panel=='P5':metrics=metrics[:5]  # HC2 reports official dense metrics only.
    fields=['parent','order','condition','arrival','expert_scheduled','quartile','query_type','updated','drift']+[a+'_'+k for a in ['Frozen','Ours','delta'] for k in metrics]
    assert all(set(r)==set(fields) for r in rows)
    with (b/'SCALARS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    if panel=='P2':
        keys=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT']]
        severity={}
        for s in [1,5,10]:
            rr=[r for r in rows if r['condition'].endswith('_'+str(s))]
            severity[str(s)]={sub:source_summary([r for r in rr if sub=='all' or not r['expert_scheduled']],keys) for sub in ['all','nonexpert']}
        write(b/'SEVERITY.json',severity)
    result=audit(b);write(b/'PUBLIC_SCALAR_AUDIT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':run(sys.argv[1])
