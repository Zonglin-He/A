"""Isolated five-round execution. No evaluator/GT in prediction stages."""
import argparse, collections, copy, fcntl, gc, hashlib, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
OUT=ROOT/'artifacts/decota_five_round_closure_v1'
F22=ROOT/'artifacts/decota_s2_system_v1'
GROUPS={'hcstvg1_test':'vid_to_hc','vidstg_test':'hc_to_vid'}


def digest(s):return hashlib.sha256(str(s).encode()).hexdigest()


def prepare():
    full=read(F22/'system_lock.json');old=read(ROOT/'artifacts/decota_shared_state_v1/LOCK.json')
    guard={'hcstvg1_test':[116,777,97,282,814],'vidstg_test':[5972,621,5191,8335,2219,987]}
    kkeys={f'{c}:{i:06d}' for c,ii in guard.items() for i in ii}|set(old['graph_keys'])
    # Union source aliases and identical media globally, not separately by query.
    parent={}
    def root(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=root(parent[x])
        return parent[x]
    def union(a,b):parent[root(b)]=root(a)
    for c,rr in full['rows'].items():
        for r in rr:union('source:'+r['input']['source'],'media:'+r['input']['video_sha256'])
    groups=collections.defaultdict(list)
    for c,rr in full['rows'].items():
        for r in rr:groups[root('source:'+r['input']['source'])].append((c,r))
    kgroups={g for g,rr in groups.items() if any(r['key'] in kkeys for c,r in rr)}
    roles={g:'K' for g in kgroups}
    for c in GROUPS:
        gg=sorted([g for g,rr in groups.items() if g not in roles and any(cc==c for cc,r in rr)],key=lambda g:digest('closure-v1/'+g))
        count=12 if len(gg)>=32 else 8
        assert len(gg)>=2*count
        for j,g in enumerate(gg):roles[g]='D_fit' if j<count else 'D_select' if j<2*count else 'D_seal'
    rows={};counts={};dev={}
    for c,rr in full['rows'].items():
        rows[c]=[]
        for r in rr:
            g=root('source:'+r['input']['source']);i=r['key'].split(':')[1]
            rows[c].append(dict(key=r['key'],source=r['input']['source'],group=g,role=roles[g],
                input=r['input'],input_unavailable=r['input_unavailable'],subject=r.get('parses',{}).get('subject',''),
                parses=r.get('parses',{}),F22_path=str(F22/'system'/c/(i+'.pt'))))
        dev[c]=[]
        for role in ('D_fit','D_select'):
            for g in sorted({r['group'] for r in rows[c] if r['role']==role}):
                cand=[r for r in rows[c] if r['group']==g and not r['input_unavailable']]
                # One deterministic query per source for bounded development;
                # ALL queries retain their source role for the final full pool.
                dev[c].append(min(cand,key=lambda r:digest('query/'+r['key']))['key'])
        counts[c]={role:dict(queries=sum(r['role']==role for r in rows[c]),
            sources=len({r['group'] for r in rows[c] if r['role']==role})) for role in ('K','D_fit','D_select','D_seal')}
    sources=['private-authorization-e87fbb5d62ea','private-authorization-606f3f30b0fb','private-authorization-0cca20cea6f9']
    files=['vg_tta/closure_replay_v1.py','scripts/run_closure_v1.py','vg_tta/shared_state_v1.py',
        'vg_tta/time_space_repair_v1.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=rows,dev_keys=dev,guard_keys=sorted(kkeys),graph_keys=old['graph_keys'],counts=counts,
        boundary=old['boundary'],spatial_lr=old['spatial_lr'],configs=full['configs'],
        attachments={x:sha(Path('./private_authorization_notes')/x/'pasted-text.txt') for x in sources},
        pins={f:sha(ROOT/f) for f in files},created=time.time(),max_rounds=5,
        dev_backward_max=60000,dev_DINO_max=4096,numerical_forwards_max=1000,
        role_policy='Union source and identical media; K includes prior four graph probes. One SHA-picked query/source in fit/select, all remaining queries preserve same source role.',
        GT_online=False,historical_exposure=True,production_write=False))
    print('LOCK',counts,flush=True)


def plan():return read(OUT/'LOCK.json')


def model_for(c):
    from scripts.run_decota_refine_v1 import student
    return student(read(ROOT/'artifacts/decota_refine_v1/lock.json'),'tastvg',GROUPS[c])


def capture(model,r):
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.decota_tastvg_episode_v1 import make_batch
    from vg_tta.shared_state_v1 import capture_shared
    f=load(r['F22_path']);receipt=read(Path(r['F22_path']).with_suffix('.json'))
    assert sha(r['F22_path'])==receipt['sha256']
    assert sha(r['input']['video_path'])==r['input']['video_sha256']
    frames,ids=decode(r['input']);batch=make_batch(frames,ids,r['input'],r['subject'],model)
    base,inputs,records,sc,action,views=capture_shared(model,batch)
    import torch
    assert torch.equal(base['raw_boxes'].float().cpu(),f['predictions']['F0']['boxes'])
    assert list(base['predicted_indices'])==f['predictions']['F0']['indices']
    return frames,ids,base,records,action,views,f


def numerical():
    import torch,numpy as np
    from vg_tta.closure_replay_v1 import ClosureReplay
    from vg_tta.shared_state_v1 import norm_of
    from vg_tta.time_space_repair_v1 import legal_logp,offset_evidence,boundary_evidence
    from vg_tta.foreground_runtime import state_digest
    p=plan();calls=0
    for c,rr in p['rows'].items():
        model=model_for(c);orig=state_digest(model)
        ordinary=next(r['key'] for r in rr if r['key'] in p['dev_keys'][c] and r['role']=='D_fit')
        for r in rr:
            if r['key'] not in p['graph_keys']+[ordinary]:continue
            dest=OUT/'round1'/f"{r['key'].replace(':','_')}.pt"
            if dest.exists():continue
            tick=time.time();frames,ids,base,records,action,views,f=capture(model,r);results={}
            sg=offset_evidence(boundary_evidence(action,ids,r['input']['fps']),records,ids,'cuda')
            anchors=f['predictions']['F4']['anchors'];eta=p['boundary'][c]['eta']
            for condition,precision,frozen in [('N0','mixed',False),('N1','fp32',False),('N2','fp32',True)]:
                it=ClosureReplay(model,views,len(ids),'shared',precision,frozen)
                def forward():
                    nonlocal calls
                    calls+=1;assert calls<=p['numerical_forwards_max'];return it.values()
                v=forward();ref=[legal_logp(z)[0].detach() for z in v['logits']]
                with torch.no_grad():repeat=forward()
                assert torch.equal(v['boxes'],repeat['boxes'])
                if condition=='N0':assert torch.equal(v['boxes'].cpu(),base['raw_boxes'].float().cpu())
                result=dict(boxes=v['boxes'].detach().cpu(),logits=[z.detach().cpu() for z in v['logits']],
                    gates=v['gates'],dtypes=dict(parameters={n:str(z.dtype) for n,z in it.named},H=[str(z.dtype) for z in v['H']],
                    logits=[str(z.dtype) for z in v['logits']],boxes=str(v['boxes'].dtype)),rows=[],proposals={})
                for kind in ('spatial','boundary','coverage'):
                    it.restore(it.initial);v=forward();loss=it.loss(v,kind,anchors,ref,sg,eta)
                    grads=torch.autograd.grad(loss,[z for n,z in it.named]);gn=norm_of(grads)
                    if gn==0:result['proposals'][kind]={'zero_gradient':True};continue
                    gs={n:g.detach() for (n,z),g in zip(it.named,grads)};lv=float(loss.detach())
                    if kind=='spatial':
                        rng=torch.Generator(device='cuda').manual_seed(20260913)
                        directions=[{n:g/gn for n,g in gs.items()}]
                        for j in range(2):
                            d={n:torch.randn(z.shape,device=z.device,generator=rng) for n,z in it.named};dn=norm_of(d.values());directions.append({n:z/dn for n,z in d.items()})
                        for di,d in enumerate(directions):
                            ad=sum(float((gs[n].double()*z.double()).sum()) for n,z in d.items())
                            for eps in (.0002,.002,.02):
                                vv=[];disp=[];changed=[];act=[];ext=[]
                                for sign in (-1,1):
                                    it.restore({n:z+sign*eps*d[n] for n,z in it.initial.items()})
                                    with torch.no_grad():w=forward();l=it.loss(w,kind,anchors,ref,sg,eta)
                                    vv.append(float(l));disp.append(norm_of([z-it.initial[n] for n,z in it.named]));changed.append(w['gates']!=v['gates'])
                                    act.append(float(torch.cat([(a!=b).reshape(-1) for a,b in zip(w['H'],v['H'])]).float().mean()))
                                    ext.append(w['extrema']!=v['extrema'])
                                fd=(vv[1]-vv[0])/(2*eps)
                                result['rows'].append(dict(direction=di,eps=eps,AD=ad,FD=fd,error=abs(fd-ad)/max(abs(fd),abs(ad),1e-8),losses=vv,
                                    base_loss=lv,actual_L2=disp,RMS=[z/math.sqrt(512) for z in disp],gate_changed=changed,activation_changed=act,ASA_extrema_changed=ext))
                    it.restore(it.initial)
                    opt=torch.optim.Adam([z for n,z in it.named],lr=.001,eps=1e-8 if kind=='spatial' else 1e-4)
                    for n,z in it.named:z.grad=gs[n].clone()
                    opt.step();after=it.state();d={n:after[n]-it.initial[n] for n in after}
                    gd=sum(float((gs[n].double()*z.double()).sum()) for n,z in d.items());curve=[]
                    for a in (0,.125,.25,.5,1):
                        it.restore(after if a==1 else {n:z+a*d[n] for n,z in it.initial.items()})
                        with torch.no_grad():w=forward();l=it.loss(w,kind,anchors,ref,sg,eta)
                        curve.append(dict(alpha=a,loss=float(l),gates_changed=w['gates']!=v['gates']))
                    result['proposals'][kind]=dict(g_dot_d=gd,L2=norm_of(d.values()),gradient_norm=gn,curve=curve,loss_dtype=str(loss.dtype))
                    del loss,v,w,grads;gc.collect()
                it.restore(it.initial);results[condition]=result;del it
            assert torch.equal(results['N1']['boxes'],results['N2']['boxes'])
            save(dest,dict(key=r['key'],results=results,seconds=time.time()-tick,GT_online=False))
            write(dest.with_suffix('.json'),dict(sha256=sha(dest),key=r['key']))
            print('ROUND1',r['key'],calls,'forwards',round(time.time()-tick,2),flush=True)
            assert state_digest(model)==orig
            del frames,base,records,action,views,f,results;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    write(OUT/'ROUND1_EXECUTION.json',dict(suffix_forwards=calls,GT_online=False,ended=time.time()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','numerical']);args=ap.parse_args()
    if args.stage=='prepare':return prepare()
    from scripts.run_decota_refine_v1 import configure
    configure();lease=open(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick=time.time();failure=None
    try:numerical()
    except BaseException as e:failure=repr(e);raise
    finally:
        write(OUT/'leases'/f'{time.time_ns()}.json',dict(stage=args.stage,seconds=time.time()-tick,failure=failure))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=='__main__':main()
