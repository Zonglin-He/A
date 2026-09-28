"""Independent NumPy cache projection and terminal cosine checks, CPU only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a0_fast_screen import D,GAP,P,load,labels_for
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins

def arr(x):return x.detach().float().numpy().astype('float64')
def flatnorm(x):
    import numpy as np
    return float(np.sqrt(np.sum(x*x,dtype=np.float64)))

def state_numpy(pred,trace,label):
    import numpy as np
    t=len(pred['frame_ids']);out=np.zeros((1,t,33));br=trace['branches']
    def soft(x):
        e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
    if br and 'time' in br[0]['logits']:
        p=soft(arr(br[0]['logits']['time']));out[0,:,:2]=p.T
        top=np.sort(p,axis=-1);out[0,:,5:7]=top[:,-1]-top[:,-2] if t>1 else 1
        out[0,:,7:9]=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/max(np.log(t),1);out[0,:,9]=1
    if pred['interval'] is not None:
        s,e=pred['interval'];out[0,s:e+1,2]=1;out[0,:,3]=s/max(t-1,1);out[0,:,4]=e/max(t-1,1)
    pos=list(pred['positions'])
    if pos:
        # Match the documented box conversion precision before FP64 statistics.
        boxes=pred['boxes_cxcywh'].detach().float().numpy();xy=np.concatenate((boxes[:,:2]-.5*boxes[:,2:],boxes[:,:2]+.5*boxes[:,2:]),-1)
        c=arr(br[1]['logits']['coordinate']);assert c.shape[-1]==152775
        ids=pred['readout']['raw_blocks'][:,1:5].numpy().astype('int64');p=soft(c)
        chosen=np.take_along_axis(p,ids[...,None],axis=-1)[...,0];ent=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(p.shape[-1])
        np.put_along_axis(p,ids[...,None],-1,axis=-1);margin=chosen-p.max(-1)
        valid=pred['geometry_valid'].numpy().astype(bool);known=np.asarray(label['box_valid'],dtype=bool)[pos]
        out[0,pos,10:14]=xy;out[0,pos,14:18]=chosen;out[0,pos,18:22]=margin;out[0,pos,22:26]=ent
        out[0,pos,26]=1;out[0,pos,27]=valid
        # Runtime subtracts FP32 evidence and boxes before masking.
        diff=(np.asarray(label['boxes_xyxy'],dtype=np.float32)[pos]-xy)*(known&valid)[:,None]
        out[0,pos,28:32]=diff;out[0,pos,32]=known
    return out

def cache(single=False):
    import numpy as np,torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4);start=time.monotonic()
    if not single:check_pins({str(D/p):h for p,h in read(D/'CACHE_SEAL.json')['files'].items()})
    U=arr(load(D/'BASIS.pt'));cfg=read(D/'CONFIG.json');errors=dict(projection_relative=0.,norm_relative=0.,state_max_abs=0.,CE_max_abs=0.)
    counts={};rowsout=[]
    for split,manifest in [('train','TRAIN128.json'),('dev','DEV64.json')]:
        rows=read(D/manifest)
        if single:rows=rows[:1] if split=='train' else []
        labels=labels_for(rows,split);defined=0
        for i,row in enumerate(rows):
            ep=D/'cache'/split/f'{i:04}';done=read(ep/'COMPLETE.json');check_pins({str(ep/p):h for p,h in done['files'].items()})
            data=load(ep/'CACHE.pt');target=arr(data['oracle_coeff256']);n=flatnorm(target);defined+=n>0
            assert data['z'].shape[-1]==128 and data['state33'].shape==(1,data['z'].shape[1],33)
            assert all(np.isfinite(arr(v)).all() for v in data.values() if isinstance(v,torch.Tensor))
            if split=='train':
                raw=load(ep/'ORACLE_RAW.pt');gt=arr(raw['gradients']['event']);gs=arr(raw['gradients']['spatial'])
                nt,ns=flatnorm(gt),flatnorm(gs);bal=gt/max(nt,1e-300)+gs/max(ns,1e-300)
                coeff=bal@U;cn=flatnorm(coeff)
                expected=-cfg['radius']*data['stock_norm']*coeff/max(cn,1e-300)
                rel=flatnorm(target-expected)/max(n,1e-300);errors['projection_relative']=max(errors['projection_relative'],rel);assert rel<3e-6
                trace=load(ep/'BASE_TRACE.pt')
                for branch,j,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
                    obj=raw['objectives'][branch]
                    if 'missing' in obj:continue
                    x=arr(trace['branches'][j]['logits'][kind]);y=obj['targets'].numpy().astype('int64');mask=obj['valid'].numpy().astype(bool)
                    top=x.max(-1);lse=top+np.log(np.exp(x-top[...,None]).sum(-1));ce=lse-np.take_along_axis(x,y[...,None],-1)[...,0]
                    err=abs(float(ce[mask].mean())-obj['CE']);errors['CE_max_abs']=max(errors['CE_max_abs'],err);assert err<1e-5
                del raw,gt,gs,bal,coeff,expected
            else:
                old=GAP/'episodes'/f"{row['gap_index']:04}";trace=load(old/'BASE_TRACE.pt')
                oldcoef=load(P/'coefficients'/f"{row['gap_index']:04}"/'ORACLE_COEFFICIENTS.pt');assert torch.equal(oldcoef,data['oracle_coeff256'])
                inp=read(old/'INPUT.json');cur=read(ep/'INPUT.json');assert cur['support']==inp['support'] and cur['preprocess']==inp['preprocess']
            if n:
                err=abs(n/(cfg['radius']*data['stock_norm'])-1);errors['norm_relative']=max(errors['norm_relative'],err);assert err<3e-6
            pred=load(ep/'B1.pt');expected_state=state_numpy(pred,trace,labels[row['key']]);err=float(np.abs(expected_state-arr(data['state33'])).max())
            errors['state_max_abs']=max(errors['state_max_abs'],err);assert err<3e-6
            rowsout.append(dict(split=split,index=i,oracle_defined=bool(n),stock_norm=data['stock_norm'],state_error=err))
            if i%32==0:print('CACHE_AUDIT',split,i+1,flush=True)
        counts[split]=dict(queries=len(rows),defined=defined,undefined=len(rows)-defined)
    name='ROOT_FIRST_CACHE_READBACK.json' if single else 'ROOT_CACHE_READBACK.json'
    write(D/name,dict(status='passed',counts=counts,errors=errors,rows=rowsout,CPU_seconds=time.monotonic()-start,
        limitation='Stock norm/features are physical worker extraction plus support hashes; no second PTD forward in root. Full train gradients and oracle projection/state independently NumPy checked. Dev oracle exact sealed reuse.',fresh_read=False))
    print('CACHE_AUDIT_PASSED',counts,errors,flush=True)

def direction():
    import numpy as np,torch
    from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer
    from vg_tta.optimizer_checkpoint import restore_optimizer
    torch.set_num_threads(4);start=time.monotonic();check_pins({str(D/p):h for p,h in read(D/'FIT_SEAL.json')['files'].items()})
    saved=load(D/'FINAL.pt');m=StateAwareDirectionMixer(load(D/'BASIS.pt'));m.load_state_dict(saved['mixer'])
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0);restore_optimizer(opt,saved['optimizer'])
    assert all(int(v['step'])==200 for v in opt.state.values());assert torch.equal(saved['mixer']['basis'],saved['initial']['basis'])
    hist=read(D/'HISTORY.json');assert len(hist)==200
    import random
    rng=random.Random(20260928);order=[];occ=[]
    for step,h in enumerate(hist,1):
        ix=[]
        for j in range(4):
            if not order:order=list(range(128));rng.shuffle(order)
            ix.append(order.pop())
        assert h['step']==step and h['indices']==ix and set(h['counters'])=={step};occ+=ix
    assert len(occ)==800 and len(set(occ))==128
    report=read(D/'DIRECTION_REPORT.json');values={};error=0.;records=[]
    for split,n in [('train',128),('dev',64)]:
        cos=[]
        for i in range(n):
            a=arr(load(D/'terminal_coefficients'/split/f'{i:04}.pt'));b=arr(load(D/'cache'/split/f'{i:04}'/'CACHE.pt')['oracle_coeff256'])
            den=flatnorm(a)*flatnorm(b);x=float((a*b).sum()/den) if den else None
            old=report['results'][split]['rows'][i]['cosine'];assert (x is None)==(old is None)
            if x is not None:error=max(error,abs(x-old));cos.append(x)
            records.append(dict(split=split,index=i,cosine=x))
        values[split]=dict(mean=float(np.mean(cos)) if cos else None,median=float(np.median(cos)) if cos else None,defined=len(cos),undefined=n-len(cos))
        for k in ['mean','median']:assert abs(values[split][k]-report['results'][split][k])<1e-10
    med=values['dev']['median'];decision='native_dev64' if med>=.1 else ('capacity_or_optimization' if values['train']['median']<.3 else 'conditioning_or_generalization')
    assert error<1e-10 and decision==report['decision']
    write(D/'ROOT_DIRECTION_READBACK.json',dict(status='passed',decision=decision,summary=values,max_cosine_error=error,records=records,actual_steps=200,query_occurrences=800,optimizer_integer_keys_and_live_binding=True,CPU_seconds=time.monotonic()-start,fresh_read=False))
    print('DIRECTION_AUDIT_PASSED',decision,values,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['first','cache','direction']);a=p.parse_args()
    direction() if a.action=='direction' else cache(a.action=='first')
