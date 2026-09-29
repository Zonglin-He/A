"""CPU-only fixed-trajectory residual scale diagnostic, staged information read."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
OLD=ROOT/'artifacts/tastvg_sparse_online_o1_v1';OUT=ROOT/'artifacts/tastvg_o11_scale_readout_v1';ALPHAS=[1,8,16,32]


def prepare():
    p=read(OLD/'LOCK.json');cb=read(OLD/'CAPTURE_BARRIER.json');ob=read(OLD/'ONLINE_BARRIER.json');tb=read(OLD/'TEACHER_FULL_BARRIER.json');comp=read(OLD/'COMPLETION.json')
    assert sha(OLD/'analysis/ROWS.json')==comp['files']['analysis/ROWS.json']
    assert sha(OLD/'ONLINE_BARRIER.json')==tb['online_barrier_sha256']
    rows=[dict(position=r['position'],parent=r['ordinal'],condition=r['condition']) for r in p['rows'] if not r['expert']];assert len(rows)==24
    files={}
    for r in rows:
        pos=r['position']
        for folder,bar in [('capture',cb),('online',ob),('teacher/full',tb)]:
            rel=f'{folder}/{pos:03}.pt';assert sha(OLD/rel)==bar['files'][rel];files[rel]=bar['files'][rel]
    files['analysis/ROWS.json']=comp['files']['analysis/ROWS.json']
    pins={f:sha(ROOT/f) for f in ['protocols/tastvg_o11_scale_readout_v1.md','scripts/run_tastvg_o11_scale_readout_v1.py','methods/CURRENT_METHOD.json']}
    write(OUT/'LOCK.json',dict(rows=rows,alphas=ALPHAS,old_files=files,pins=pins,source_O1_commit='93eeabcb3a3ad5ce1e091f1efdd1ef8cab951e6b',GT_read=False,new_training=False,new_expert=False,new_GPU=False,time=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    for f,h in p['old_files'].items():assert sha(OLD/f)==h,f
    return p


@torch.no_grad()
def select():
    start=time.monotonic();torch.set_num_threads(4);p=verify();rows=[];error=0.;margin=float('inf');access=[]
    for r in p['rows']:
        pos=r['position'];x=load(OLD/'capture'/f'{pos:03}.pt');state=load(OLD/'online'/f'{pos:03}.pt');access.extend([f'capture/{pos:03}.pt',f'online/{pos:03}.pt'])
        assert not state['expert'];phi=x['phi'];w=state['arrival_state']['w'];base=x['base_score'];assert float(state['arrival_state']['b'])==0
        assert all(z.device.type=='cpu' for z in [phi,w,base]);residual=phi@w
        independent=np.array([math.fsum(float(a)*float(b) for a,b in zip(v,w.tolist())) for v in phi.tolist()]);err=float(np.max(np.abs(independent-residual.numpy())));error=max(error,err);assert err<1e-12
        variants={}
        for alpha in ALPHAS:
            scores=base+alpha*residual;k=int(scores.argmax());check=int(np.argmax(base.numpy()+alpha*independent));assert k==check
            top=np.sort(scores.numpy());gap=float(top[-1]-top[-2]);margin=min(margin,gap)
            if alpha==1:assert torch.equal(scores,state['arrival_scores']) and k==state['selected']['Online Slow-Fast']
            variants[str(alpha)]=dict(selected=k,scores=scores.tolist(),changed_from_frozen=k!=0,top_two_margin=gap)
        rows.append(dict(**r,native_score=base.tolist(),residual=residual.tolist(),arrival_state_sha256=state['arrival_hash'],variants=variants))
    assert not torch.cuda.is_initialized();verify();write(OUT/'SELECTIONS.json',rows)
    write(OUT/'SELECTION_BARRIER.json',dict(files={'SELECTIONS.json':sha(OUT/'SELECTIONS.json')},arrivals=24,readouts=96,alpha1_exact=True,teacher_read=False,GT_or_cached_metric_read=False,raw_tensor_access=access,max_independent_dot_error=error,minimum_top_two_margin=margin,seconds=time.monotonic()-start,time=time.time()))
    print('SEALED96 selections; alpha1 exactly O1; dot error',error)


def agreement():
    start=time.monotonic();p=verify();bar=read(OUT/'SELECTION_BARRIER.json');assert sha(OUT/'SELECTIONS.json')==bar['files']['SELECTIONS.json'];rows=read(OUT/'SELECTIONS.json');out=[]
    for r in rows:
        pos=r['position'];e=load(OLD/'teacher/full'/f'{pos:03}.pt');k=int(np.argmax(e['scores']));assert k==e['selected']
        out.append(dict(position=pos,full_selected=k,full_scores=e['scores'],native_agrees=k==0,variants={a:dict(agrees=z['selected']==k,toward_vs_alpha1=z['selected']==k and r['variants']['1']['selected']!=k,away_vs_alpha1=z['selected']!=k and r['variants']['1']['selected']==k) for a,z in r['variants'].items()}))
    write(OUT/'AGREEMENT.json',out);write(OUT/'AGREEMENT_BARRIER.json',dict(selection_barrier_sha256=sha(OUT/'SELECTION_BARRIER.json'),agreement_sha256=sha(OUT/'AGREEMENT.json'),GT_or_cached_metric_read=False,new_expert_calls=0,seconds=time.monotonic()-start,time=time.time()))
    print('AGREEMENT', {a:sum(r['variants'][a]['agrees'] for r in out) for a in ['1','8','16','32']})


def score():
    from scripts.analyze_tastvg_corruption_c0c1_v1 import stats,METRICS
    start=time.monotonic();p=verify();sb=read(OUT/'SELECTION_BARRIER.json');ab=read(OUT/'AGREEMENT_BARRIER.json');assert sha(OUT/'SELECTION_BARRIER.json')==ab['selection_barrier_sha256'];assert sha(OUT/'AGREEMENT.json')==ab['agreement_sha256'];assert sha(OUT/'SELECTIONS.json')==sb['files']['SELECTIONS.json']
    rows=read(OUT/'SELECTIONS.json');agreements=read(OUT/'AGREEMENT.json');old={r['position']:r for r in read(OLD/'analysis/ROWS.json')};out=[]
    for r,g in zip(rows,agreements):
        assert r['position']==g['position'];q=old[r['position']];assert not q['expert']
        assert q['base_score']==r['native_score'] and int(np.argmax(g['full_scores']))==q['selected']['Full Rerank']
        cm=q['candidate_metrics'];variants={a:{**z,**g['variants'][a], 'metrics':cm[z['selected']]} for a,z in r['variants'].items()};assert variants['1']['metrics']==q['arms']['Online Slow-Fast']
        out.append(dict(**{k:v for k,v in r.items() if k!='variants'},full_selected=g['full_selected'],full_scores=g['full_scores'],candidate_metrics=cm,frozen=cm[0],full=cm[g['full_selected']],variants=variants))
    summary={}
    for alpha in ALPHAS:
        a=str(alpha);summary[a]=dict(n=24,changed_from_frozen=sum(r['variants'][a]['changed_from_frozen'] for r in out),changed_from_alpha1=sum(r['variants'][a]['selected']!=r['variants']['1']['selected'] for r in out),full_agreement=sum(r['variants'][a]['agrees'] for r in out),toward_full=sum(r['variants'][a]['toward_vs_alpha1'] for r in out),away_full=sum(r['variants'][a]['away_vs_alpha1'] for r in out),
            metrics={m:stats([r['variants'][a]['metrics'][m] for r in out]) for m in METRICS},delta_alpha1={m:stats([r['variants'][a]['metrics'][m]-r['variants']['1']['metrics'][m] for r in out]) for m in METRICS},
            positive={m:sum(r['variants'][a]['metrics'][m]-r['variants']['1']['metrics'][m]>1e-12 for r in out) for m in METRICS},negative={m:sum(r['variants'][a]['metrics'][m]-r['variants']['1']['metrics'][m]<-1e-12 for r in out) for m in METRICS},harms_gt5pp={m:sum(r['variants'][a]['metrics'][m]-r['variants']['1']['metrics'][m]<-.05 for r in out) for m in METRICS})
    reference={name:dict(metrics={m:stats([r[field][m] for r in out]) for m in METRICS},full_agreement=sum((0 if field=='frozen' else r['full_selected'])==r['full_selected'] for r in out)) for name,field in [('Frozen','frozen'),('Full Rerank','full')]}
    write(OUT/'ROWS.json',out);write(OUT/'SUMMARY.json',summary);write(OUT/'REFERENCE.json',reference)
    assert not torch.cuda.is_initialized();verify()
    write(OUT/'SCORE_RECEIPT.json',dict(cached_metric_file_sha256=p['old_files']['analysis/ROWS.json'],no_new_GT_file_read=True,cached_GT_derived_metrics_used=True,old_inputs_unchanged=True,new_model_forwards=0,new_expert_calls=0,new_updates=0,GPU_seconds=0,seconds=time.monotonic()-start,time=time.time()))
    print('SUMMARY',summary)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','select','agreement','score']);x=a.parse_args();globals()[x.stage]()
