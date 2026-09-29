"""O3 task readback from previously audited O2 candidate metrics, after online seal."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_conditional_opd_o3_v1 import OUT,OLD,verify,MODES
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS
ARMS=['Budgeted Rerank','Linear Slow',*MODES]
COMPS=[('Linear Slow','Budgeted Rerank'),('Conditional Pairwise','Budgeted Rerank'),('Conditional Reverse-KL','Budgeted Rerank'),('Conditional Reverse-KL','Conditional Pairwise'),('Conditional Pairwise','Linear Slow'),('Conditional Reverse-KL','Linear Slow')]


def summarize(rr):
    return dict(n=len(rr),arms={a:{m:stats([r['arms'][a][m] for r in rr]) for m in METRICS} for a in ARMS},comparisons={a+' - '+b:{m:stats([r['arms'][a][m]-r['arms'][b][m] for r in rr]) for m in METRICS} for a,b in COMPS},positive={a+' - '+b:{m:sum(r['arms'][a][m]-r['arms'][b][m]>1e-12 for r in rr) for m in METRICS} for a,b in COMPS},negative={a+' - '+b:{m:sum(r['arms'][a][m]-r['arms'][b][m]<-1e-12 for r in rr) for m in METRICS} for a,b in COMPS},harms_gt5pp={a+' - '+b:{m:sum(r['arms'][a][m]-r['arms'][b][m]<-.05 for r in rr) for m in METRICS} for a,b in COMPS})


def run():
    p=verify();assert read(OUT/'STATE_AUDIT.json')['status']=='pass';assert sha(OUT/'ONLINE.json')==read(OUT/'ONLINE_BARRIER.json')['sha256'];rows=read(OUT/'ONLINE.json');old={r['position']:r for r in read(OLD/'analysis/ROWS.json')}
    for r in rows:
        o=old[r['position']];assert r['base_score']==o['base_score'];r['selected']['Linear Slow']=o['selected']['Online Slow-Fast'];r['candidate_metrics']=o['candidate_metrics'];r['arms']={a:r['candidate_metrics'][r['selected'][a]] for a in ARMS};r['linear_arrival_scores']=o['arrival_scores'];r['teacher_scores']=o['teacher_scores']
        if r['expert']:assert len(set(r['selected'].values()))==1
    groups={c:[r for r in rows if r['condition']==c] for c in p['conditions']};macro=[]
    for parent in [r['parent'] for r in groups[p['conditions'][0]]]:
        rr=[r for r in rows if r['parent']==parent];assert len(rr)==5
        macro.append(dict(parent=parent,expert=rr[0]['expert'],arms={a:{m:float(np.mean([r['arms'][a][m] for r in rr])) for m in METRICS} for a in ARMS}))
    groups['macro']=macro;s={c:{g:summarize(rr if g=='all' else [r for r in rr if r['expert']==(g=='expert')]) for g in ['nonexpert','expert','all']} for c,rr in groups.items()}
    diag={}
    for c in p['conditions']:
        rr=groups[c];nn=[r for r in rr if not r['expert']];dd={}
        for a in MODES:
            er=[r for r in rr if r['expert']];ranks=[]
            for r in nn:
                base=np.array(r['base_score']);res=np.array(r['arrival_scores'][a])-base;ranks.append(dict(position=r['position'],residual_range=float(np.ptp(res)),base_range=float(np.ptp(base)),selected=r['selected'][a]))
            dd[a]=dict(changed=sum(r['selected'][a]!=r['selected']['Budgeted Rerank'] for r in nn),loss_down=sum(r['diagnostics'][a]['after']['total']<r['diagnostics'][a]['before']['total'] for r in er),current_loss_down=sum(r['diagnostics'][a]['after']['current']<r['diagnostics'][a]['before']['current'] for r in er),replay_loss_up=sum(r['diagnostics'][a]['after']['replay']>r['diagnostics'][a]['before']['replay']+1e-12 for r in er),readouts=ranks)
        diag[c]=dd
    write(OUT/'ROWS.json',rows);write(OUT/'MACRO_ROWS.json',macro);write(OUT/'SUMMARY.json',s);write(OUT/'DIAGNOSTICS.json',diag)
    write(OUT/'SCORE_RECEIPT.json',dict(cached_GT_derived_metrics_used=True,no_new_GT_file_read=True,O2_rows_sha256=p['old_files']['analysis/ROWS.json'],new_GPU=0,new_model_forwards=0,new_expert_calls=0,old_inputs_unchanged=True,time=time.time()))
    for c in [*p['conditions'],'macro']:print(c,s[c]['nonexpert']['comparisons'])

if __name__=='__main__':run()
