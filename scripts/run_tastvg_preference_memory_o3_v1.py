"""CPU-only O3 preference memory with staged teacher and metric access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from vg_tta.tastvg_preference_memory_v1 import append,select,state_hash
OLD=ROOT/'artifacts/tastvg_sparse_online_o1_v1';OUT=ROOT/'artifacts/tastvg_preference_memory_o3_v1'
ARMS=['Budgeted Rerank','Fast + Preference Memory','Full Rerank']


def prepare():
    assert not (OUT/'LOCK.json').exists();p=read(OLD/'LOCK.json');files={}
    for barname in ['CAPTURE_BARRIER.json','TEACHER_ONLINE_BARRIER.json','TEACHER_FULL_BARRIER.json']:
        for f,h in read(OLD/barname)['files'].items():assert sha(OLD/f)==h;files[f]=h
    f='analysis/ROWS.json';h=read(OLD/'COMPLETION.json')['files'][f];assert sha(OLD/f)==h;files[f]=h
    rows=[dict(position=r['position'],parent=r['ordinal'],condition=r['condition'],expert=r['expert']) for r in p['rows']]
    pins={f:sha(ROOT/f) for f in ['vg_tta/tastvg_preference_memory_v1.py','scripts/run_tastvg_preference_memory_o3_v1.py','protocols/tastvg_preference_memory_o3_v1.md','methods/CURRENT_METHOD.json']}
    write(OUT/'LOCK.json',dict(rows=rows,old_files=files,pins=pins,m=3,GT_read=False,time=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    for f,h in p['old_files'].items():assert sha(OLD/f)==h,f
    return p


def online():
    t=time.monotonic();torch.set_num_threads(4);p=verify();memory=[];rows=[];reads=[]
    for r in p['rows']:
        pos=r['position'];x=load(OLD/'capture'/f'{pos:03}.pt');phi=x['phi'].numpy();base=x['base_score'].tolist();before=state_hash(memory);size=len(memory)
        k,wins,pairs=select(memory,phi,base);budget=0
        if r['expert']:
            e=load(OLD/'teacher/online'/f'{pos:03}.pt');reads.append(pos);budget=e['selected'];chosen=budget
            memory=append(memory,phi,e['scores'],pos)
        else:chosen=k
        rows.append(dict(**r,base_score=base,arrival_wins=wins,arrival_selected=k,pairs=pairs,selected={'Budgeted Rerank':budget,'Fast + Preference Memory':chosen},arrival_memory_hash=before,after_memory_hash=state_hash(memory),arrival_memory_size=size,after_memory_size=len(memory),new_pairs=len(memory)-size,output_before_write=True))
    assert reads==list(range(1,33,4)) and not torch.cuda.is_initialized();verify();write(OUT/'ONLINE.json',rows)
    write(OUT/'ONLINE_BARRIER.json',dict(sha256=sha(OUT/'ONLINE.json'),positions=32,expert_reads=reads,nonexpert_teacher_reads=0,memory_writes=8,final_pairs=len(memory),GT_or_cached_metric_read=False,seconds=time.monotonic()-t,time=time.time()))


def full():
    verify();assert read(OUT/'STATE_AUDIT.json')['status']=='pass';bar=read(OUT/'ONLINE_BARRIER.json');assert sha(OUT/'ONLINE.json')==bar['sha256'];rows=read(OUT/'ONLINE.json')
    for r in rows:
        pos=r['position'];phase='online' if r['expert'] else 'full';e=load(OLD/'teacher'/phase/f'{pos:03}.pt');r['selected']['Full Rerank']=e['selected'];r['teacher_scores']=e['scores']
    write(OUT/'SEALED.json',rows);write(OUT/'PREDICTION_BARRIER.json',dict(sha256=sha(OUT/'SEALED.json'),online_barrier_sha256=sha(OUT/'ONLINE_BARRIER.json'),GT_or_cached_metric_read=False,time=time.time()))


def score():
    from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS
    p=verify();assert sha(OUT/'SEALED.json')==read(OUT/'PREDICTION_BARRIER.json')['sha256'];rows=read(OUT/'SEALED.json');old={r['position']:r for r in read(OLD/'analysis/ROWS.json')}
    for r in rows:
        q=old[r['position']];assert r['base_score']==q['base_score'] and r['teacher_scores']==q['teacher_scores'];r['candidate_metrics']=q['candidate_metrics'];r['arms']={a:r['candidate_metrics'][k] for a,k in r['selected'].items()}
    summary={}
    for name,rr in [('nonexpert',[r for r in rows if not r['expert']]),('all',rows),('expert',[r for r in rows if r['expert']])]:
        comps={a+' - Budgeted Rerank':{m:stats([r['arms'][a][m]-r['arms']['Budgeted Rerank'][m] for r in rr]) for m in METRICS} for a in ARMS[1:]}
        delta=lambda r,m:r['arms']['Fast + Preference Memory'][m]-r['arms']['Budgeted Rerank'][m]
        summary[name]=dict(n=len(rr),arms={a:{m:stats([r['arms'][a][m] for r in rr]) for m in METRICS} for a in ARMS},comparisons=comps,changed=sum(r['selected']['Fast + Preference Memory']!=r['selected']['Budgeted Rerank'] for r in rr),full_agreement={a:sum(r['selected'][a]==r['selected']['Full Rerank'] for r in rr) for a in ARMS[:2]},positive={m:sum(delta(r,m)>1e-12 for r in rr) for m in METRICS},negative={m:sum(delta(r,m)<-1e-12 for r in rr) for m in METRICS},harms_gt5pp={m:sum(delta(r,m)<-.05 for r in rr) for m in METRICS})
    write(OUT/'ROWS.json',rows);write(OUT/'SUMMARY.json',summary);verify();write(OUT/'SCORE_RECEIPT.json',dict(no_new_GT_file=True,cached_GT_metrics_used=True,old_inputs_unchanged=True,new_model_forwards=0,new_expert_calls=0,new_optimizer_steps=0,GPU_seconds=0,time=time.time()))
    print('PRIMARY',summary['nonexpert'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','online','full','score']);globals()[p.parse_args().stage]()
