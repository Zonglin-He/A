"""Post-seal evaluation and isolated, explicitly GT-supervised P0 diagnostics."""
import argparse,collections,hashlib,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.ptd_spatial_adapter_ab_v1 import OUT,verify,path,digest
from scripts.decota_matrix_common_v1 import read,write,save,load,sha
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter,anchor_indices,boxes_from_tokens,sample,fit_A,kl

def seal(rows,condition,views):
    files={}
    for r in rows:
        for v in views:
            f=path(r,condition,v);assert sha(f)==read(f.with_suffix('.json'))['sha'];files[str(f)]=sha(f)
    target=OUT/'barriers'/(condition+'_'+str(len(rows))+'_'+str(len(views))+'.json')
    write(target,dict(files=files,stage='predictions_sealed_before_label_read',time=time.time()))
    return sha(target)

def truth(row,labels):
    g=labels[row['key']];pos=row['parent_positions']
    assert len(g['boxes'])==len(row['parent_frame_ids'])
    return dict(boxes=np.asarray(g['boxes'])[pos],valid=np.asarray(g['valid'],bool)[pos],interval=g['interval'])

def iou(boxes,gt):
    b=np.asarray(boxes,float);g=np.asarray(gt,float)
    left=np.maximum(b[...,:2]-b[...,2:]/2,g[...,:2]-g[...,2:]/2)
    right=np.minimum(b[...,:2]+b[...,2:]/2,g[...,:2]+g[...,2:]/2)
    inter=np.maximum(right-left,0).prod(-1)
    return inter/np.maximum(np.maximum(b[...,2:],0).prod(-1)+g[...,2:].prod(-1)-inter,1e-12)

def evaluate(z,tokens,row,g):
    n=len(row['input']['frame_ids']);ids=np.asarray(row['input']['frame_ids']);present=np.zeros(n,bool);boxes=np.zeros((n,4))
    anchors=[];geometry_invalid=0
    if z['format_ok']:
        b,valid=boxes_from_tokens(torch.as_tensor(tokens));positions=np.asarray(z['positions']);boxes[positions]=b.numpy();present[positions]=True
        anchors=[z['positions'][j] for j in anchor_indices(len(positions))];geometry_invalid=int((~valid).sum())
    q=iou(boxes,g['boxes']);q[~present]=0
    # Independent scalar geometry implementation, including invalid zero boxes.
    for j in np.flatnonzero(g['valid']):
        x1,y1=boxes[j,:2]-boxes[j,2:]/2;x2,y2=boxes[j,:2]+boxes[j,2:]/2
        a1,b1=g['boxes'][j,:2]-g['boxes'][j,2:]/2;a2,b2=g['boxes'][j,:2]+g['boxes'][j,2:]/2
        ar=max(0,min(x2,a2)-max(x1,a1))*max(0,min(y2,b2)-max(y1,b1))
        v=ar/max(max(0,x2-x1)*max(0,y2-y1)+(a2-a1)*(b2-b1)-ar,1e-12)
        assert abs(v-q[j])<1e-9
    if z.get('interval') is None:
        v=0.;t=0.;inside=np.zeros(n,bool)
    else:
        s,e=z['interval'];a,b=ids[s],ids[e]+1;c,d=g['interval']
        inside=(ids>=a)&(ids<b);denom=((ids>=min(a,c))&(ids<max(b,d))).sum()
        v=float(q[g['valid']&inside].sum()/max(1,denom));t=max(0,min(b,d)-max(a,c))/max(1,max(b,d)-min(a,c))
    common=g['valid']&inside&present;anchor_mask=np.isin(np.arange(n),anchors)
    def avg(mask):return float(q[mask].mean()) if mask.any() else None
    return dict(v=v,t=float(t),s=avg(common),anchor=avg(common&anchor_mask),nonanchor=avg(common&~anchor_mask),
                gt_positions=int(g['valid'].sum()),common_positions=int(common.sum()),anchor_positions=int((common&anchor_mask).sum()),
                nonanchor_positions=int((common&~anchor_mask).sum()),coverage=float(common.sum()/max(1,g['valid'].sum())),
                format_ok=z['format_ok'],invalid_boxes=geometry_invalid,ious=q.tolist(),anchors=anchors)

def stats(values):
    v=np.asarray([x for x in values if x is not None],float)
    if not len(v):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260924);boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    sv=np.sort(v);cut=int(len(v)*.05);trim=sv[cut:len(v)-cut].mean()
    return dict(n=len(v),mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),median=float(np.median(v)),
                positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),trimmed5=float(trim),
                severe_loss_gt5pp=int((v<-.05).sum()),minimum=float(v.min()),maximum=float(v.max()))

def p0():
    p=verify();rows=[r for r in read(OUT/'INPUTS.json') if r['p0']];bar=seal(rows,'clean',['original'])
    assert sha(p['labels'])==p['labels_sha'];labels=read(p['labels']);rr=[];torch.set_num_threads(4);start=time.perf_counter()
    for r in rows:
        z=load(path(r,'clean'));g=truth(r,labels)
        if not z['format_ok']:rr.append(dict(key=r['key'],cohort=r['cohort'],skipped='original_format_failure'));continue
        h=z['h'].clone();base=z['logits'].clone();model=Adapter(h.shape[-1]);initial={k:v.clone() for k,v in model.state_dict().items()}
        assert torch.equal(base+model(h),base)
        anchors=anchor_indices(len(z['positions']));valid=[j for j in anchors if g['valid'][z['positions'][j]]]
        native=evaluate(z,base.argmax(-1),r,g)
        if not valid:rr.append(dict(key=r['key'],cohort=r['cohort'],skipped='no_GT_at_native_anchors',native=native));continue
        gt=torch.tensor(g['boxes'][np.asarray(z['positions'])[valid]],dtype=torch.float32)
        target=torch.cat((gt[:,:2]-gt[:,2:]/2,gt[:,:2]+gt[:,2:]/2),-1).mul(1000).round().clamp(0,1000).long()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.);history=[];grad=None
        for step in range(31):
            logits=base+model(h);ce=torch.nn.functional.cross_entropy(logits[valid].reshape(-1,1001),target.reshape(-1))
            if step in [0,1,3,30]:history.append(dict(step=step,ce=float(ce.detach()),metrics=evaluate(z,logits.argmax(-1).detach(),r,g),tokens=logits.argmax(-1).detach().tolist()))
            if step==30:break
            opt.zero_grad();ce.backward()
            if step==0:grad=dict(up=float(model.up.weight.grad.norm()),down=float(model.down.weight.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        assert grad['up']>0
        clone=Adapter(h.shape[-1]);clone.load_state_dict(model.state_dict());assert torch.equal(clone(h),model(h))
        model.load_state_dict(initial);zeroopt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.)
        ce=torch.nn.functional.cross_entropy((base+model(h))[valid].reshape(-1,1001),target.reshape(-1));ce.backward();zeroopt.step()
        assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
        rr.append(dict(key=r['key'],cohort=r['cohort'],native=native,history=history,gradient=grad,zero_lr_exact=True,reload_exact=True,
                       supervised_anchors=len(valid),GT_trained_state_exported=False))
        print('P0',r['key'],'anchors',len(valid),'CE',history[0]['ce'],history[-1]['ce'],'anchorIoU',native['anchor'],history[-1]['metrics']['anchor'],flush=True)
    good=[r for r in rr if 'history' in r]
    summary={c:dict(sources=sum(r['cohort']==c for r in rr),fitted=sum(r['cohort']==c for r in good),
                      anchor_gain=stats([r['history'][-1]['metrics']['anchor']-r['native']['anchor'] for r in good if r['cohort']==c]),
                      nonanchor_gain=stats([r['history'][-1]['metrics']['nonanchor']-r['native']['nonanchor'] for r in good if r['cohort']==c and r['native']['nonanchor'] is not None])) for c in ['hcstvg1_test','vidstg_test']}
    passed=bool(good) and all(r['history'][-1]['ce']<r['history'][0]['ce'] for r in good) and np.mean([r['history'][-1]['metrics']['anchor']-r['native']['anchor'] for r in good])>0
    write(OUT/'P0.json',dict(status='completed',gate=bool(passed),summary=summary,rows=rr,seconds=time.perf_counter()-start,barrier_sha=bar,
                           scope='GT diagnostic only; adapter states destroyed; no backbone updates; nonanchor propagation not assumed'))

def signal(condition='clean'):
    p=verify();rows=[r for r in read(OUT/'INPUTS.json') if r['split']=='development'];bar=seal(rows,condition,['original','gamma0.9','gamma1.1'])
    # Seal genuine student candidate samples before opening GT.
    for r in rows:
        z=load(path(r,condition));f=OUT/'candidates'/condition/(r['key'].replace(':','_')+'.pt')
        if f.exists():continue
        if z['format_ok']:
            a=anchor_indices(len(z['positions']));sp=sample(z['logits'],a,int(digest(r['key']+'|'+condition+'|Bround1')[:12],16))
            save(f,dict(**sp,anchor_local=a,anchor_positions=[z['positions'][j] for j in a],key=r['key'],GT_used=False))
        else:save(f,dict(key=r['key'],skipped='original_format_failure',GT_used=False))
    write(OUT/'barriers'/('candidates_'+condition+'.json'),{str(f):sha(f) for f in (OUT/'candidates'/condition).glob('*.pt')})
    assert sha(p['labels'])==p['labels_sha'];labels=read(p['labels']);rr=[]
    for r in rows:
        z=load(path(r,condition));g=truth(r,labels)
        native=evaluate(z,z.get('base_tokens',[]),r,g);entry=dict(key=r['key'],source=r['source'],cohort=r['cohort'],native=native)
        if not z['format_ok']:entry['skipped']='original_format_failure';rr.append(entry);continue
        views=[z]+[load(path(r,condition,v)) for v in ['gamma0.9','gamma1.1']]
        if all(x['format_ok'] for x in views):
            assert all(x['positions']==z['positions'] and x['interval']==z['interval'] for x in views)
            prob=torch.stack([x['logits'].softmax(-1) for x in views]);q=prob.mean(0)
            fusion=evaluate(z,q.argmax(-1),r,g);js=(prob*(prob.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(-1).mean(0).mean(-1)
            entry.update(fusion=fusion,js=js.tolist(),consensus_changed_positions=int((q.argmax(-1)!=z['base_tokens']).any(-1).sum()))
            aa=native['anchors'];valid=[a for a in aa if g['valid'][a]]
            ni=np.asarray(native['ious'])[valid];fi=np.asarray(fusion['ious'])[valid]
            entry['A_anchor']=dict(n=len(valid),wrong_right=int(((ni<.5)&(fi>=.5)).sum()),right_wrong=int(((ni>=.5)&(fi<.5)).sum()),
                                   mean_gain=float((fi-ni).mean()) if len(ni) else None,changed=int((ni!=fi).sum()))
        else:entry['A_skipped']='view_native_format_failure'
        s=load(OUT/'candidates'/condition/(r['key'].replace(':','_')+'.pt'));pool=[]
        for i,pos in enumerate(s['anchor_positions']):
            if not g['valid'][pos]:continue
            ios=iou(s['boxes'][i],g['boxes'][pos]);native_iou=native['ious'][pos]
            pool.append(dict(position=pos,native_iou=native_iou,ious=ios.tolist(),valid=s['valid'][i].tolist(),oracle_gain=float(ios.max()-native_iou),
                             recall05=bool(ios.max()>=.5),native_wrong=bool(native_iou<.5)))
        entry['B_pool']=pool;rr.append(entry)
    ss={}
    for c in ['hcstvg1_test','vidstg_test']:
        rc=[r for r in rr if r['cohort']==c];aa=[r['A_anchor'] for r in rc if 'A_anchor' in r];pool=[v for r in rc for v in r.get('B_pool',[])];wrong=[v for v in pool if v['native_wrong']]
        ss[c]=dict(sources=len(rc),format_ok=sum(r['native']['format_ok'] for r in rc),
                   A=dict(anchor_gain=stats([r['mean_gain'] for r in aa]),wrong_right=sum(r['wrong_right'] for r in aa),right_wrong=sum(r['right_wrong'] for r in aa),
                          changed=sum(r['changed'] for r in aa),anchor_count=sum(r['n'] for r in aa),
                          dv=stats([r['fusion']['v']-r['native']['v'] for r in rc if 'fusion' in r]),
                          nonanchor_gain=stats([r['fusion']['nonanchor']-r['native']['nonanchor'] for r in rc if 'fusion' in r and r['native']['nonanchor'] is not None])),
                   B=dict(evaluable_anchors=len(pool),wrong_anchors=len(wrong),recall05=float(np.mean([v['recall05'] for v in pool])) if pool else None,
                          wrong_recall05=float(np.mean([v['recall05'] for v in wrong])) if wrong else None,
                          wrong_oracle_gain=float(np.mean([v['oracle_gain'] for v in wrong])) if wrong else None))
    am=[r['A_anchor']['mean_gain'] for r in rr if r.get('A_anchor',{}).get('mean_gain') is not None]
    ag=bool(am) and np.mean(am)>0 and sum(x['A']['wrong_right'] for x in ss.values())>0 and all(x['A']['anchor_gain']['mean'] is not None and x['A']['anchor_gain']['mean']>=-.005 for x in ss.values())
    wrong=[v for r in rr for v in r.get('B_pool',[]) if v['native_wrong']]
    bg=bool(wrong) and any(v['recall05'] for v in wrong) and np.mean([v['oracle_gain'] for v in wrong])>0
    write(OUT/('SIGNAL_'+condition+'.json'),dict(status='completed',A_signal_pass=bool(ag),B_pool_pass=bool(bg),summary=ss,rows=rr,barrier_sha=bar))
    print('SIGNAL',condition,'A',bool(ag),'Bpool',bool(bg),ss,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['p0','signal']);a.add_argument('--condition',default='clean');x=a.parse_args()
    p0() if x.action=='p0' else signal(x.condition)
