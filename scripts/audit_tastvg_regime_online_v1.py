"""Independent chronological SGD reconstruction; no GT or utility scoring."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_regime_online_capture_v1 import OUT,verify
from methods.decota_final_simplified_v1.tensors import state_hash


def run():
    p=verify();bar=read(OUT/'ONLINE_BARRIER.json');cb=read(OUT/'CAPTURE_BARRIER.json');w=np.zeros(768);b=0.;previous=None;maxerr=0.;maxloss=0.;reads=[];writes=0
    for r in p['rows']:
        if r['stream_position']==1:w=np.zeros(768);b=0.;previous=None
        pos=r['position'];rel=f'online/{pos:03}.pt';assert sha(OUT/rel)==bar['files'][rel];z=load(OUT/rel);relc=f'capture/{pos:03}.pt';assert sha(OUT/relc)==cb['files'][relc];x=load(OUT/relc)
        assert z['previous_record_sha256']==previous;assert state_hash(z['arrival_state'])==z['arrival_hash'];assert state_hash(z['after_state'])==z['after_hash']
        np.testing.assert_allclose(w,z['arrival_state']['w'].numpy(),rtol=0,atol=1e-12);assert float(z['arrival_state']['b'])==b==0.
        hidden=x['hidden'].numpy().astype(float);phi=np.stack([np.r_[hidden[s],hidden[e],x['hidden'][s:e+1].mean(0).numpy()] for s,e in [c['indices'] for c in x['candidates']]])
        np.testing.assert_array_equal(phi,x['phi'].numpy());ell=x['base_score'].numpy();assert int(ell.argmax())==0
        ss=ell+phi@w;np.testing.assert_allclose(ss,z['arrival_scores'].numpy(),atol=1e-12,rtol=0);assert int(ss.argmax())==z['before_selected']
        if r['expert']:
            assert z['teacher_read']==f'teacher/online/{pos:03}.pt';e=load(OUT/z['teacher_read']);reads.append(pos);rr=e['scores'];pp=[(i,j) for i in range(len(rr)) for j in range(len(rr)) if rr[i]>rr[j]+1e-12]
            grad=np.zeros(768)
            for i,j in pp:grad-=(phi[i]-phi[j])*np.exp(-np.logaddexp(0,ss[i]-ss[j]))
            if pp:grad/=len(pp)
            initial=float(np.mean([np.logaddexp(0,-(ss[i]-ss[j])) for i,j in pp])) if pp else 0.
            w-=p['lr']*grad;post=ell+phi@w;final=float(np.mean([np.logaddexp(0,-(post[i]-post[j])) for i,j in pp])) if pp else 0.
            maxloss=max(maxloss,abs(initial-z['diagnostics']['loss_before']),abs(final-z['diagnostics']['loss_after']));writes+=1
            assert z['selected']['Online Slow-Fast']==z['selected']['Budgeted Rerank']==int(np.argmax(rr))
        else:
            assert z['teacher_read'] is None and z['diagnostics'] is None and z['arrival_hash']==z['after_hash']
            assert z['selected']['Budgeted Rerank']==0 and z['selected']['Online Slow-Fast']==int(ss.argmax())
        err=np.max(np.abs(w-z['after_state']['w'].numpy()));maxerr=max(maxerr,float(err));assert err<1e-12;previous=sha(OUT/rel)
    assert reads==p['expert_positions'] and writes==20 and maxloss<1e-12
    write(OUT/'STATE_AUDIT.json',dict(status='pass',positions=80,expert_reads=reads,nonexpert_teacher_reads=0,updates=20,zero_state_native=80,previous_hash_chain=True,independent_numpy_SGD_max_absolute_error=maxerr,independent_loss_max_absolute_error=maxloss,features_exact=True,online_barrier_sha256=sha(OUT/'ONLINE_BARRIER.json'),time=time.time()))
    print('STATE AUDIT PASS',maxerr,maxloss)

if __name__=='__main__':run()
