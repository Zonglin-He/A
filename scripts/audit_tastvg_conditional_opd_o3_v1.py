"""Independent NumPy forward and analytic gradients for both O3 MLP students."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_conditional_opd_o3_v1 import OLD,OUT,verify,MODES
from methods.decota_final_simplified_v1.tensors import state_hash


def lossgrad(w,item,mode):
    x=item['phi'];base=item['base'];teacher=item['teacher'];pre=x@w['0.weight'].T+w['0.bias'];h=np.maximum(pre,0);s=base+(h@w['2.weight'].T).ravel()+w['2.bias'][0]
    if mode=='pairwise':
        pairs=[(i,j) for i in range(len(s)) for j in range(len(s)) if teacher[i]>teacher[j]+1e-12];ds=np.zeros(len(s));loss=0.
        for i,j in pairs:
            v=s[i]-s[j];loss+=np.logaddexp(0,-v);a=-np.exp(-np.logaddexp(0,v));ds[i]+=a;ds[j]-=a
        if pairs:loss/=len(pairs);ds/=len(pairs)
    else:
        lp=s-np.logaddexp.reduce(s);lq=teacher-np.logaddexp.reduce(teacher);p=np.exp(lp);loss=np.sum(p*(lp-lq));ds=p*(lp-lq-loss)
    dh=ds[:,None]*w['2.weight'];dz=dh*(pre>0)
    g={'0.weight':dz.T@x,'0.bias':dz.sum(0),'2.weight':(ds@h)[None,:],'2.bias':np.array([ds.sum()])}
    return float(loss),g,s


def run():
    start=time.monotonic();p=verify();bar=read(OUT/'ONLINE_BARRIER.json');assert sha(OUT/'ONLINE.json')==bar['sha256'];assert sha(OUT/'INITIAL.pt')==bar['initial_sha256'];initial=load(OUT/'INITIAL.pt');rows=read(OUT/'ONLINE.json');maxgrad=0.;maxstate=0.;maxloss=0.;writes=0;reads=[];first_layer_updates=0
    for r in rows:
        pos=r['position']
        if r['stream_position']==1:states={a:{k:v.numpy().copy() for k,v in initial.items()} for a in MODES};replay=[];previous=None
        f=OUT/'states'/f'{pos:03}.pt';assert sha(f)==r['state_file_sha256'];z=load(f);assert z['previous_record_sha256']==previous;previous=sha(f)
        x=load(OLD/'capture'/f'{pos:03}.pt');item=dict(position=pos,phi=x['phi'].numpy(),base=x['base_score'].numpy(),teacher=np.zeros(len(x['base_score'])))
        assert r['replay_before']==[x['position'] for x in replay]
        if r['expert']:
            e=load(OLD/'teacher/online'/f'{pos:03}.pt');reads.append(pos);item['teacher']=np.asarray(e['scores']);assert r['selected']['Budgeted Rerank']==e['selected']
        for a,mode in MODES.items():
            w=states[a];assert state_hash(z['arrival'][a])==r['arrival_hash'][a]
            for k in w:np.testing.assert_allclose(w[k],z['arrival'][a][k].numpy(),rtol=0,atol=1e-11)
            loss,g,s=lossgrad(w,item,mode);np.testing.assert_allclose(s,r['arrival_scores'][a],atol=1e-11,rtol=0)
            if r['expert']:
                assert r['selected'][a]==e['selected'];before=loss;pastloss=0.
                for past in replay:
                    l,gg,_=lossgrad(w,past,mode);pastloss+=l/len(replay)
                    for k in g:g[k]+=gg[k]/len(replay)
                di=r['diagnostics'][a];maxloss=max(maxloss,abs(di['before']['total']-before-pastloss))
                for k in g:
                    err=float(np.max(np.abs(g[k]-z['gradients'][a][k].numpy())));maxgrad=max(maxgrad,err);assert err<1e-10;w[k]-=.001*g[k]
                if np.linalg.norm(g['0.weight'])>0:first_layer_updates+=1
                curr,_,_=lossgrad(w,item,mode);past=sum(lossgrad(w,t,mode)[0] for t in replay)/len(replay) if replay else 0
                maxloss=max(maxloss,abs(di['after']['total']-curr-past));writes+=1
            else:
                assert r['selected'][a]==int(np.argmax(s));assert r['arrival_hash'][a]==r['after_hash'][a]
            for k in w:
                err=float(np.max(np.abs(w[k]-z['after'][a][k].numpy())));maxstate=max(maxstate,err);assert err<1e-11
            assert state_hash(z['after'][a])==r['after_hash'][a]
        if r['expert']:replay=(replay+[item])[-4:]
        assert r['replay_after']==[x['position'] for x in replay] and all(x['position']<pos for x in replay if x['position']!=pos)
    assert writes==40 and reads==[r['position'] for r in p['rows'] if r['expert']] and maxloss<1e-10
    write(OUT/'STATE_AUDIT.json',dict(status='pass',arrivals=80,models=2,reset_pairs=5,expert_reads=20,nonexpert_teacher_reads=0,SGD_updates=40,first_layer_nonzero_updates=first_layer_updates,independent_gradient_max_error=maxgrad,independent_state_max_error=maxstate,independent_loss_max_error=maxloss,no_future_replay=True,zero_native_identity=80,seconds=time.monotonic()-start,time=time.time()))
    print('NUMPY AUDIT PASS',maxgrad,maxstate,maxloss)

if __name__=='__main__':run()
