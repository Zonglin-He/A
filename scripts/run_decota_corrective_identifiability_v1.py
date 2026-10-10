"""Bounded CPU diagnostics; label-free barrier precedes diagnostic GT use."""
from __future__ import annotations
import argparse, collections, copy, hashlib, importlib.util, itertools, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/decota_corrective_identifiability_v1'
PARENT=ROOT/'artifacts/decota_final_simplification_v1'
PROTOCOL=ROOT/'protocols/decota_corrective_identifiability_v1.md'
DELTAS=[-8,-4,-2,0,2,4,8]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def prepare():
    p=read(PARENT/'LOCK.json');rows=[]
    for c,rr in p['rows'].items():
        dev=[r for r in rr if r['f44_role']=='development']
        assert len(dev)==len({r['group'] for r in dev})==32
        for r in dev:
            f=PARENT/'panel'/c/f"{r['ordinal']:06d}.pt";rec=read(f.with_suffix('.json'))
            assert rec['key']==r['key'] and sha(f)==rec['sha256']
            rows.append(dict(key=r['key'],source=r['source'],group=r['group'],cohort=c,path=str(f),sha256=sha(f)))
    files=['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json',str(PARENT.relative_to(ROOT)/'FINAL_CONFIG.json')]
    files += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    write('LOCK.json',dict(rows=rows,labels=p['labels'],labels_sha256=p['labels_sha256'],historical_exposure=True,
          runner_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),protected={f:sha(ROOT/f) for f in files},
          delta_indices=DELTAS,threads=2,spatial_GPU_status='not_started_pending_priority',created=time.time()))
def verify(p):
    assert p['runner_sha256']==sha(__file__) and p['protocol_sha256']==sha(PROTOCOL)
    for f,h in p['protected'].items():assert sha(ROOT/f)==h,f
def cfg(c):
    from methods.decota_final_simplified_v1.config import MethodConfig
    return MethodConfig.for_direction('vid_to_hc1' if c=='hcstvg1_test' else 'hc2_to_vid')
def native_indices(x):
    from methods.decota_final_simplified_v1.objectives import native_view_indices
    return [native_view_indices(z) for z in x['native_logits']]
def tiou(intervals,g):
    import numpy as np
    a=np.asarray(intervals,float);inter=np.maximum(0,np.minimum(a[...,1],g[1])-np.maximum(a[...,0],g[0]))
    return inter/(np.maximum(a[...,1],g[1])-np.minimum(a[...,0],g[0]))
def local_pairs(s,e,n):
    return [(s,e)]+sorted({(s+a,e+b) for a,b in itertools.product(DELTAS,repeat=2) if 0<=s+a<e+b<n}-{(s,e)})
def energies(boxes,t):
    import numpy as np
    b=np.asarray(boxes,float);t=np.asarray(t,float);v=np.diff(b,axis=-2)/np.diff(t)[...,None]
    speed=(v*v).sum(-1);ve=(speed*np.diff(t)).sum(-1)/(t[-1]-t[0])
    if len(t)<3:return ve,ve*0
    d=(np.diff(t)[:-1]+np.diff(t)[1:])/2;acc=np.diff(v,axis=-2)/d[...,None]
    ae=((acc*acc).sum(-1)*d).sum(-1)/d.sum()
    return ve,ae
def spatial_geometry(x):
    import numpy as np
    from scipy.stats import rankdata
    b=x['native_boxes'].numpy().astype(float);allb=[b.copy()];names=['native']
    for dx,dy,k in itertools.product([-.05,0,.05],[-.05,0,.05],[.9,1.,1.1]):
        if (dx,dy,k)==(0,0,1):continue
        z=b.copy();z[:,0]+=dx*b[:,2];z[:,1]+=dy*b[:,3];z[:,2:]*=k;allb.append(z);names.append(f'shift{dx},{dy}_scale{k}')
    for w in [3,5]:
        allb.append(np.array([b[max(0,i-w//2):min(len(b),i+w//2+1)].mean(0) for i in range(len(b))]));names.append(f'mean{w}')
    boxes=np.stack(allb);v,a=energies(boxes,np.array(x['frame_ids'])/x['input']['fps'])
    scores=dict(velocity=-v,acceleration=-a,rank_combination=-(rankdata(v,method='average')+rankdata(a,method='average'))/2)
    # Exact translational invariance can acquire roundoff; native wins numerical ties.
    selected={k:int(np.flatnonzero(s>=s.max()-1e-12)[0]) for k,s in scores.items()}
    return dict(boxes=boxes,names=names,scores=scores,selected=selected)
def temporal_candidates(x):
    import numpy as np, torch
    from methods.decota_final_simplified_v1.objectives import project
    acts=[a['raw_logits'] for a in x['teacher']];c=cfg(x['cohort'])
    ev=project(x['native_logits'],acts,x['records'],c)
    roll=project(x['native_logits'],[torch.roll(a,len(a)//2) for a in acts],x['records'],c)
    offs=[]
    for (s,e),rec,z,r in zip(native_indices(x),x['records'],ev['offsets'],roll['offsets']):
        pairs=local_pairs(s,e,len(rec['frame_ids']));lookup={tuple(v):i for i,v in enumerate(z['ij'].T.tolist())};inds=[lookup[v] for v in pairs]
        f=rec['frame_ids'];offs.append(dict(intervals=np.array([[f[a],f[b]+1] for a,b in pairs]),indices=pairs,
            scores=dict(structured=z['score'][inds].numpy(),evidence=(-z['cost'][inds]).numpy(),native=z['logp0'][inds].numpy(),time_shift=r['score'][inds].numpy())))
    combos=list(itertools.product(range(len(offs[0]['indices'])),range(len(offs[1]['indices']))));a,b=np.array(combos).T
    intervals=np.c_[np.minimum(offs[0]['intervals'][a,0],offs[1]['intervals'][b,0]),np.maximum(offs[0]['intervals'][a,1],offs[1]['intervals'][b,1])]
    scores={k:(offs[0]['scores'][k][a]+offs[1]['scores'][k][b])/2 for k in offs[0]['scores']}
    chosen={k:int(v.argmax()) for k,v in scores.items()}
    assert intervals[0].tolist()==x['predictions']['Frozen']['physical_interval']
    assert chosen['native']==0 or intervals[chosen['native']].tolist()==intervals[0].tolist()
    return dict(intervals=intervals,scores=scores,chosen=chosen,offsets=offs,evidence=ev)
def run():
    import torch
    torch.set_num_threads(2);p=read(OUT/'LOCK.json');verify(p);rows=[]
    for r in p['rows']:
        assert sha(r['path'])==r['sha256'];x=load(r['path'])
        rows.append(dict(key=r['key'],temporal=temporal_candidates(x),geometry=spatial_geometry(x)))
    f=OUT/'LABEL_FREE.pt';assert not f.exists();torch.save(rows,f)
    write('LABEL_FREE_BARRIER.json',dict(sha256=sha(f),queries=len(rows),no_GT_access_in_run=True,created=time.time()))
    print('LABEL_FREE_SEALED',len(rows),flush=True)
def aggregate(vals):
    import numpy as np
    a=np.array([v for v in vals if v is not None],float)
    if not len(a):return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260917);boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),ci95=np.quantile(boot,[.025,.975]).tolist(),
                positive=int((a>.001).sum()),negative=int((a<-.001).sum()),neutral=int((abs(a)<=.001).sum()),harm_gt5pp=int((a<-.05).sum()))
def ranking(scores,util,chosen):
    import numpy as np
    from scipy.stats import rankdata
    u=np.asarray(util);base=float(u[0]);head=float(u.max()-base);out={}
    for k,s in scores.items():
        idx=chosen[k];delta=float(u[idx]-base);pos=u>base+.001;neg=u<base-.001;au=None
        if pos.any() and neg.any():
            # Mann-Whitney including half credit for tied scores.
            a=np.asarray(s)[pos];b=np.asarray(s)[neg];au=float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))
        out[k]=dict(index=idx,baseline=base,selected=float(u[idx]),delta=delta,headroom=head,
             recovery=delta/head if head>1e-12 else None,correction_accuracy=float(delta>0),practical_win=float(delta>.001),candidate_correction_AUROC=au)
    return out
def temporal_mirror(x,gt,ev):
    import numpy as np
    g,h=gt['interval'];lo,hi=x['frame_ids'][0],x['frame_ids'][-1]+1;d=h-g
    if g<lo or h>hi or d<=0:return dict(status='GT_outside_observed_window')
    starts=np.arange(lo,hi-d+1);neg=np.c_[starts,starts+d];neg=neg[tiou(neg,[g,h])<=.1]
    if len(neg)==0:return dict(status='no_equal_length_low_overlap_interval')
    def val(ii):
        rr=[]
        for z in ev['offsets']:
            edges=z['cell_edges'].numpy();width=np.diff(edges);u=z['standardized_logits'].numpy()
            # Constant background term cancels in comparisons; partial cell overlap exact.
            inter=np.maximum(0,np.minimum(ii[:,1,None],edges[None,1:])-np.maximum(ii[:,0,None],edges[None,:-1]))
            rr.append((inter@u)/(edges[-1]-edges[0]))
        return np.mean(rr,axis=0)
    gs=float(val(np.array([[g,h]]))[0]);ns=val(neg);at=int(ns.argmax())
    return dict(status='scored',gt_score=gs,negatives=len(neg),hard_interval=neg[at].tolist(),hard_score=float(ns[at]),
                beats_hard=float(gs>ns[at]),beats_average_fraction=float(np.mean(gs>ns)),ties=int(np.sum(gs==ns)))
def make_head(state):
    spec=importlib.util.spec_from_file_location('official_correctiveness_mlp',ROOT/'external/TA-STVG/models/net_utils.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    head=m.MLP(256,256,2,2,dropout=.3).float().eval();head.load_state_dict({k.removeprefix('head.'):v for k,v in state.items()});return head
def head_logits(head,x):return [head(h)[-1] for h in x['temporal_cache']['inputs']]
def flat_grad(head):
    import torch
    return torch.cat([p.grad.detach().reshape(-1).double() for p in head.parameters()])
def temporal_gradient(x,gt,ev):
    import numpy as np,torch
    from methods.decota_final_simplified_v1.objectives import prediction,temporal_loss
    from methods.decota_final_simplified_v1.tensors import state_hash
    head=make_head(x['temporal_cache']['head_state']);source=copy.deepcopy(head.state_dict());oracle=copy.deepcopy(ev)
    quant=[]
    for z,rec in zip(oracle['offsets'],x['records']):
        f=np.array(rec['frame_ids']);ij=z['ij'].numpy();ivs=np.c_[f[ij[0]],f[ij[1]]+1];v=tiou(ivs,gt['interval']);at=int(v.argmax());z['target']=at;quant.append(float(v[at]))
    initial=head_logits(head,x);maxerr=max(float((a-b).abs().max()) for a,b in zip(initial,x['native_logits']))
    for a,b in zip(initial,x['native_logits']):assert torch.all((a-b).abs()<=2e-5+1e-6*b.abs()),maxerr
    pred=lambda zz:prediction(zz,x['spatial']['final']['boxes'],x['records'],x['frame_ids'])
    assert pred(initial)['physical_interval']==x['predictions']['Frozen']['physical_interval']
    grads={}
    for name,target in [('ssl',ev),('gt',oracle)]:
        head.zero_grad();temporal_loss(head_logits(head,x),target).backward();grads[name]=flat_grad(head)
    norms={k:float(v.norm()) for k,v in grads.items()};cos=float(torch.dot(grads['ssl'],grads['gt'])/(norms['ssl']*norms['gt'])) if min(norms.values())>1e-14 else None
    arms={};audit={};saved={}
    for name,target in [('ssl',ev),('gt',oracle)]:
        head.load_state_dict(source);opt=torch.optim.AdamW(head.parameters(),lr=cfg(x['cohort']).temporal_lr,eps=1e-4,weight_decay=0)
        losses=[]
        for step in range(1,6):
            opt.zero_grad();loss=temporal_loss(head_logits(head,x),target);loss.backward();opt.step();losses.append(float(loss))
            if step not in [1,5]:continue
            state=copy.deepcopy(head.state_dict());delta=torch.cat([(state[k]-source[k]).reshape(-1).double() for k in state])
            if step==1:
                # Positive means actual step locally decreases GT loss.
                audit[name+'_first_step_GT_descent']=float(-torch.dot(delta,grads['gt']))
            for eta in [1.,.25]:
                head.load_state_dict({k:source[k]+eta*(state[k]-source[k]) for k in state})
                tag=f'{name}_step{step}_eta{eta}';pr=pred(head_logits(head,x));arms[tag]=pr;saved[tag]=copy.deepcopy(head.state_dict())
            head.load_state_dict(state)
        audit[name+'_losses']=losses
    # Check the CPU rerun of the locked SSL trajectory against its saved final state.
    assert arms['ssl_step5_eta0.25']['physical_interval']==x['predictions']['Full_DeCoTA']['physical_interval']
    head.load_state_dict(source)
    return dict(cosine=cos,norms=norms,GT_discretization_tIoU=quant,max_native_logit_error=maxerr,audit=audit,arms=arms,states=saved,
                reset_exact=all(torch.equal(head.state_dict()[k],v) for k,v in source.items()),parameter_count=sum(p.numel() for p in head.parameters()))
def score():
    import numpy as np,torch
    from scripts.analyze_spatial10_components_v1 import checked_score
    from vg_tta.box_stability_diagnostics_v1 import overlap
    torch.set_num_threads(2);p=read(OUT/'LOCK.json');verify(p);assert sha(OUT/'LABEL_FREE.pt')==read(OUT/'LABEL_FREE_BARRIER.json')['sha256'];assert sha(p['labels'])==p['labels_sha256']
    labels=read(p['labels']);rows=[];states=[];zall={r['key']:r for r in load(OUT/'LABEL_FREE.pt')}
    for r in p['rows']:
        x=load(r['path']);g=labels[r['key']];z=zall[r['key']];t=z['temporal'];geom=z['geometry']
        # Collapse duplicate physical intervals without changing the selected physical output.
        unique,inv=np.unique(t['intervals'],axis=0,return_inverse=True);native=np.flatnonzero(np.all(unique==t['intervals'][0],axis=1))[0]
        order=np.r_[native,np.delete(np.arange(len(unique)),native)];uu=unique[order]
        ss={k:np.array([s[inv==i].max() for i in order]) for k,s in t['scores'].items()}
        cc={k:int(np.flatnonzero(np.all(uu==t['intervals'][a],axis=1))[0]) for k,a in t['chosen'].items()}
        tr=ranking(ss,tiou(uu,g['interval']),cc)
        valid=np.array(g['valid'],bool);truth=np.array(g['boxes']);sq=overlap(geom['boxes'],truth[None])[:,valid].mean(1)
        sr=ranking(geom['scores'],sq,geom['selected'])
        grad=temporal_gradient(x,g,t['evidence']);arms={}
        for k,a in grad.pop('arms').items():arms[k]=checked_score(a['boxes'],g,x['frame_ids'],a['indices'])[0]
        states.append(dict(key=r['key'],states=grad.pop('states')))
        base=checked_score(x['spatial']['final']['boxes'],g,x['frame_ids'],x['predictions']['Frozen']['indices'])[0]
        rows.append(dict(**r,temporal=tr,geometry=sr,geometry_names=geom['names'],mirror=temporal_mirror(x,g,t['evidence']),gradient=grad,adaptation=arms,baseline=base,
                         candidates=len(uu),raw_decompositions=len(t['intervals'])))
        print('SCORED',len(rows),r['key'],'grad',grad['cosine'],flush=True)
    f=OUT/'TEMPORAL_ORACLE_STATES.pt';assert not f.exists();torch.save(states,f);write('CPU_ROWS.json',rows)
    groups={}
    for c in sorted({r['cohort'] for r in rows}):
        rr=[r for r in rows if r['cohort']==c];d={}
        for family in ['temporal','geometry']:
            d[family]={}
            for signal in rr[0][family]:
                a=[r[family][signal] for r in rr];d[family][signal]={k:aggregate([q[k] for q in a]) for k in ['delta','recovery','headroom','correction_accuracy','candidate_correction_AUROC']}
                denom=sum(q['headroom'] for q in a);d[family][signal]['aggregate_headroom_recovery']=sum(q['delta'] for q in a)/denom if denom>0 else None
        d['gradients']=aggregate([r['gradient']['cosine'] for r in rr]);d['gradients']['positive_fraction']=sum(r['gradient']['cosine'] is not None and r['gradient']['cosine']>0 for r in rr)/len(rr)
        mm=[r['mirror'] for r in rr if r['mirror']['status']=='scored'];d['mirror']={k:aggregate([r[k] for r in mm]) for k in ['beats_hard','beats_average_fraction']}
        d['mirror']['unavailable']=collections.Counter(r['mirror']['status'] for r in rr if r['mirror']['status']!='scored')
        d['adaptation']={k:{m:aggregate([r['adaptation'][k][m]-r['baseline'][m] for r in rr]) for m in ['vIoU_corrected','tIoU','sIoU']} for k in rr[0]['adaptation']}
        groups[c]=d
    write('CPU_SUMMARY.json',groups);verify(p);print('CPU_COMPLETE',len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','score']);a=ap.parse_args();globals()[a.stage]()
