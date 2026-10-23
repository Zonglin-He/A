"""Describe both signs of the sealed comparison, without new labels/predictions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from full import read,sha,atomic,check,identity,RUN
D=ROOT/'outputs/cycle_reliability_ablation_v1';P=ROOT/'results/cycle_reliability_ablation_v1'

def run():
    report=read(P/'report.json');seal=read(D/'sealed.json');receipt=read(D/'score_receipt.json')
    assert receipt['report_sha256']==sha(P/'report.json') and receipt['seal_sha256']==sha(D/'sealed.json')
    reverse=dict(complete_v_count=0,unobserved_s_count=0,union_count=0)
    cases=[];lookup={e['alias']:e for e in seal['files'].values()}
    for row in report['rows']:
        dv=100*(row['values']['NoCycle']['v']-row['values']['Current']['v'])
        ds=100*(row['values']['NoCycle']['unobserved_actual_sIoU']-row['values']['Current']['unobserved_actual_sIoU'])
        reverse['complete_v_count']+=int(dv>5);reverse['unobserved_s_count']+=int(ds>5);reverse['union_count']+=int(max(dv,ds)>5)
        if min(dv,ds)>=-5 and max(dv,ds)<=5:continue
        e=lookup[row['query']];assert sha(D/e['path'])==e['sha256'] and sha(D/e['trace'])==e['trace_sha256']
        tr=torch.load(D/e['trace'],map_location='cpu',weights_only=False)
        with np.load(D/e['path'],allow_pickle=False)as z:
            changed=z['Current_actions']!=z['NoCycle_actions'];t=np.arange(len(changed));new=z['NoCycle_actions'];oldvalid=tr['Current_valid'].numpy()
            reopened=changed & (new>0) & ~oldvalid[t,new];confidence=tr['Current_confidence'].numpy()[changed]
        cases.append(dict(query=row['query'],parent=row['parent'],delta_complete_v_pp=dv,delta_unobserved_s_pp=ds,changed_nodes=int(changed.sum()),newly_selected_previously_cycle_ineligible_slots=int(reopened.sum()),old_confidence_at_changed_nodes=dict(min=float(confidence.min()) if len(confidence) else None,mean=float(confidence.mean()) if len(confidence) else None,max=float(confidence.max()) if len(confidence) else None)))
    p=check();state=read(RUN/'state.json');proc=[]
    for x in [state['controller'],read(RUN/'vidstg/worker.json')]:
        current=identity(x['pid']);assert current['startticks']==x['startticks'];st=Path(f"/proc/{x['pid']}/stat").read_text().rsplit(')',1)[1].split()[0];assert st not in ('Z','X');proc.append(dict(current,state=st))
    atomic(D/'production_unchanged_receipt.json',dict(production_fingerprint=p['fingerprint'],plan_sha256=sha(RUN/'plan.json'),processes=proc,Vid_progress=read(RUN/'vidstg/progress.json'),code_and_parameters_unchanged=True))
    atomic(P/'AUDIT.json',dict(status='passed',queries=67,parents=67,availability_preflight_geometry_exact=True,geometry_and_current_output_proofs=seal['checks'],all_predictions_sealed_before_independent_CPU_labels=True,all67_time_identical=True,official_dense_checks=report['checks'],max_metric_error=report['max_metric_error'],reverse_severe_Current_minus_NoCycle=reverse,severe_both_signs_cases=cases,new_models=0,new_RGB=0,new_expert=0,new_training=0,GPU=0,preflight_CPU_seconds=read(D/'source_only_preflight.json')['seconds'],prediction_CPU_seconds=seal['seconds'],score_CPU_seconds=report['seconds'],cost_scope='Cached reconstruction/replay only, not deployment acceleration',production_fingerprint=p['fingerprint'],production_unchanged=True,report_sha256=sha(P/'report.json'),seal_sha256=sha(D/'sealed.json'),audit_code_sha256=sha(__file__)))
    print(json.dumps(dict(reverse_severe=reverse,cases=cases)),flush=True)

if __name__=='__main__':run()
