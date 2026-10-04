"""Independent CPU formulas, private receipt/state audit and public scalar audit.

No model or experimental loss helper is imported. This audits output-space
derivatives and parameter SGD, not the frozen model's complete Jacobian.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,json,hashlib,collections,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/tastvg_negative_evidence_v1'
PUB=ROOT/'results/tastvg_negative_evidence/2026-10-04'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
ARMS=['rank_native','rank_cross','negative_global','negative_local']
CONTRASTS=[(a,'pre') for a in ARMS]+[(a,'rank_native') for a in ARMS[1:]]+[('negative_local','negative_global')]
BLOCKS=('query','norm1','norm3','norm4')
BUNDLES={'vidstg':dict(lr=.033761698432507946,teacher_temperature=.34902548789596055,steps=1),
         'hc2':dict(lr=.006097133675874025,teacher_temperature=1.,steps=8)}


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,z):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),p
    p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
def key(c):return '/'.join(str(c[x]) for x in ('dataset','split','condition','order'))+f"/{c['arrival']:05}"
def shstate(s):
    h=hashlib.sha256()
    for n,v in sorted(s.items()):h.update(n.encode());h.update(np.ascontiguousarray(array(v)).tobytes())
    return h.hexdigest()


def summary(rows,fields):
    """Independent reconstruction of the paired source bootstrap, seed and draw order."""
    if not rows:return dict(cells=0,sources=0,metrics={})
    ids=sorted({r['source_id'] for r in rows})
    a=np.array([[np.mean([r[k] for r in rows if r['source_id']==i]) for k in fields] for i in ids])
    rng=np.random.default_rng(20261004)
    boot=np.concatenate([a[rng.integers(len(a),size=(100,len(a)))].mean(1) for _ in range(100)])
    ci=np.percentile(boot,[2.5,97.5],axis=0)
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=20261004,
        metrics={k:dict(mean=float(a[:,j].mean()),ci95=ci[:,j].tolist(),
            source_values={str(i):float(v) for i,v in zip(ids,a[:,j])},
            source_positive=int((a[:,j]>1e-12).sum()),source_negative=int((a[:,j]<-1e-12).sum()),
            cell_macro=float(np.mean([r[k] for r in rows]))) for j,k in enumerate(fields)})


def independent_tails(rows,a,b):
    d=np.array([r[a+'_v']-r[b+'_v'] for r in rows])
    return dict(gain=int((d>1e-12).sum()),harm=int((d< -1e-12).sum()),
        severe_harm_gt5pp=int((d< -.05).sum()),severe_harm_gt20pp=int((d< -.20).sum()),
        baseline_good_destroyed_at_03=sum(r[b+'_v']>.3 and r[a+'_v']<=.3 for r in rows),
        baseline_bad_rescued_at_03=sum(r[b+'_v']<=.3 and r[a+'_v']>.3 for r in rows),
        baseline_good_destroyed_at_05=sum(r[b+'_v']>.5 and r[a+'_v']<=.5 for r in rows),
        baseline_bad_rescued_at_05=sum(r[b+'_v']<=.5 and r[a+'_v']>.5 for r in rows))


class Checker:
    def __init__(self):self.count=collections.Counter();self.errors=collections.defaultdict(float)
    def close(self,a,b,kind='scalar',tol=1e-10,rtol=0):
        a,b=np.asarray(array(a)),np.asarray(array(b));assert a.shape==b.shape,(kind,a.shape,b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(),kind
        err=float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        assert np.allclose(a,b,atol=tol,rtol=rtol),(kind,err,tol)
        self.errors[kind]=max(self.errors[kind],err);self.count[kind]+=a.size
    def tree(self,a,b,kind='summary'):
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),(kind,set(a)^set(b))
            for k in a:self.tree(a[k],b[k],kind)
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):self.tree(x,y,kind)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):self.close(a,b,kind)
        else:assert a==b,(kind,a,b)


def public(directory=PUB):
    tick=time.monotonic();directory=Path(directory);ck=Checker()
    cfg=read(directory/'CONFIG.json');bar=read(directory/'PREDICTION_BARRIER.json')
    assert cfg['lambda_value']==cfg['student_temperature']==1 and cfg['parameters']==1792
    assert cfg['teacher_GT_used'] is False and cfg['production_promoted'] is False
    assert cfg['bootstrap_seed']==20261004 and cfg['bootstrap_draws']==10000
    assert bar['GT_read'] is False and bar['time']<bar['score_start_time']
    assert cfg['global_barrier_sha256']==bar['global_barrier_sha256']
    rows={s:read(directory/f) for s,f in [('current','ROWS.json'),('future','FUTURE_ROWS.json'),('reset_u','RESET_U_ROWS.json')]}
    assert len(rows['current'])==len(rows['future'])==288 and len(rows['reset_u'])==1152
    saved=read(directory/'SUMMARY.json');traces=read(directory/'STEP_DIAGNOSTICS.json')
    for scope,rr in rows.items():
        assert len({r['cell_key'] for r in rr})==len(rr)
        contrasts=[('reset_u','A'),('reset_u_fixedA','A')] if scope=='reset_u' else CONTRASTS
        for r in rr:
            for a,b in contrasts:
                for m in ('v','t','s'):ck.close(r[f'{a}_minus_{b}_{m}'],r[a+'_'+m]-r[b+'_'+m],'differences')
                d=r[a+'_v']-r[b+'_v'];ck.close([r[f'{a}_gross_gain_{b}'],r[f'{a}_gross_loss_{b}']],[max(d,0),max(-d,0)],'gross')
            if scope=='future':assert r['single_write_transfer'] and r['future_arrival']==r['arrival']+1
            if scope=='current':
                for a in ARMS:
                    for b in BLOCKS:ck.close(r[a+'_'+b+'_minus_pre_v'],r[a+'_'+b+'_v']-r['pre_v'],'block_differences')
        for ds in ('vidstg','hc2'):
            for sp in ('search','confirm'):
                q=[r for r in rr if r['dataset']==ds and r['split']==sp]
                predicates={'corrupt':lambda r:r['condition']!='clean','clean':lambda r:r['condition']=='clean','all':lambda r:True}
                if scope=='reset_u':predicates.update(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition']!='clean',nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition']!='clean')
                for name,select in predicates.items():
                    v=[r for r in q if select(r)];z=saved[scope][ds][sp][name];fields=list(z['metrics'])
                    rec=summary(v,fields)
                    for k in rec:ck.tree(rec[k],z[k],'bootstrap')
                    ck.tree({a+'_minus_'+b:independent_tails(v,a,b) for a,b in contrasts},z['negative_tails'],'tails')
                    for o in ('order1','order2'):ck.tree(summary([r for r in v if r['order']==o],fields),z['orders'][o],'orders')
                    if scope=='future':
                        target=[dict(r,source_id=r['target_source_id']) for r in v]
                        zz=saved['future_target_source_sensitivity'][ds][sp][name]
                        rec=summary(target,fields)
                        for k in rec:ck.tree(rec[k],zz[k],'target_source_bootstrap')
                        ck.tree(z['negative_tails'],zz['negative_tails'],'target_source_tails')
                        for o in ('order1','order2'):ck.tree(summary([r for r in target if r['order']==o],fields),zz['orders'][o],'target_source_orders')
    grouped=collections.defaultdict(list)
    for t in traces:
        grouped[(t['cell_key'],t['arm'])].append(t)
        shape=(-1,9) if t['arm']=='negative_local' else (9,)
        p0,q,pa=[np.asarray(t[k],float).reshape(shape) for k in ('p0','q','p_after')]
        e=np.asarray(t['e_jk'],float).reshape(-1,9);assert (e>=0).all() and (e<=1+1e-6).all()
        for p in (p0,q,pa):
            if p.size:ck.close(p.sum(-1),np.ones(p.shape[:-1]),'probability_normalization',2e-6)
        assert t['negative_GT_better_entries']<=t['known_negative_entries']<=t['negative_entries']
        assert t['observed_GT_frames']+t['unknown_or_outside_event_observations']==t['valid_observations']
        assert t['negative_entries']+t['undecided_entries']==8*t['valid_observations']
        ck.close(t['negative_evidence_mass'],e[:,1:].sum(),'evidence_mass')
        if t['arm'].startswith('negative'):
            ev=e if t['arm']=='negative_local' else (e.mean(0) if len(e) else np.zeros(9))
            expected=p0*np.exp(-ev)
            if expected.size:expected/=expected.sum(-1,keepdims=True)
            ck.close(expected,q,'public_negative_target',2e-6)
            if t['target_undecided_log_odds_max_error'] is not None:assert t['target_undecided_log_odds_max_error']<2e-6
            if t['arm']=='negative_local':assert t['output_gradient_unobserved_norm']==0.
    assert len(grouped)==1152
    current={r['cell_key']:r for r in rows['current']}
    for (k,a),tt in grouped.items():
        tt=sorted(tt,key=lambda t:t['step']);assert [t['step'] for t in tt]==list(range(len(tt)))
        assert len(tt)==current[k][a+'_steps']
        ck.close([tt[0]['loss'],tt[-1]['loss_after']],[current[k][a+'_loss_before'],current[k][a+'_loss_after']],'trace_endpoint')
    # CSV and a decision are optional until report(), but independently bound afterwards.
    for scope,filename in [('current','ROWS.csv'),('future','FUTURE_ROWS.csv'),('reset_u','RESET_U_ROWS.csv')]:
        if (directory/filename).exists():
            with (directory/filename).open() as f:csvrows=list(csv.DictReader(f))
            assert len(csvrows)==len(rows[scope]);by={r['cell_key']:r for r in rows[scope]}
            for r in csvrows:
                for k,v in r.items():
                    if isinstance(by[r['cell_key']][k],(int,float)) and not isinstance(by[r['cell_key']][k],bool):ck.close(float(v),by[r['cell_key']][k],'CSV')
    if (directory/'DECISION.json').exists():
        d=read(directory/'DECISION.json');assert d['production_promoted'] is False and d['full_online_negative_loss_established'] is False
        for a in ('negative_global','negative_local','rank_cross'):
            expected=all(saved['current'][ds]['confirm']['corrupt']['metrics'][a+'_minus_rank_native_v']['ci95'][0]>0 for ds in ('vidstg','hc2'))
            assert d['confirmation_positive_lower_bound_vs_rank_native'][a]==expected
    if (directory/'EXECUTION_DIAGNOSTICS.json').exists():
        dd=read(directory/'EXECUTION_DIAGNOSTICS.json')
        for ds in ('vidstg','hc2'):
            for sp in ('search','confirm'):
                for a in ARMS:
                    tt=[t for t in traces if t['dataset']==ds and t['split']==sp and t['arm']==a and t['condition']!='clean']
                    z=dd[ds][sp][a]
                    expected=dict(steps=len(tt),known_negative_entries=sum(t['known_negative_entries'] for t in tt),
                        negative_GT_better_entries=sum(t['negative_GT_better_entries'] for t in tt),
                        known_negative_evidence_mass=sum(t['known_negative_evidence_mass'] for t in tt),
                        negative_GT_better_evidence_mass=sum(t['negative_GT_better_evidence_mass'] for t in tt),
                        frozen_step_loss_decreased=sum(t['loss_after']<t['loss']-1e-12 for t in tt),
                        frozen_step_loss_increased=sum(t['loss_after']>t['loss']+1e-12 for t in tt),
                        frozen_step_loss_equal=sum(abs(t['loss_after']-t['loss'])<=1e-12 for t in tt),
                        frozen_step_loss_down_task_down=sum(t['loss_after']<t['loss']-1e-12 and t['post_prediction_v']<t['pre_prediction_v']-1e-12 for t in tt))
                    for k,v in expected.items():ck.close(v,z[k],'report_diagnostics')
                    for k,num,den in [('observed_negative_misfire_entry_fraction','negative_GT_better_entries','known_negative_entries'),
                                      ('observed_negative_misfire_weight_fraction','negative_GT_better_evidence_mass','known_negative_evidence_mass')]:
                        if expected[den]:ck.close(z[k],expected[num]/expected[den],'report_ratio')
                        else:assert z[k] is None
    for filename in ('FIGURE_MANIFEST.json','REPORT_MANIFEST.json'):
        if (directory/filename).exists():
            for f,h in read(directory/filename)['files'].items():assert sha(directory/f)==h;ck.count['public_manifest_files']+=1
    return dict(status='pass',checks=dict(ck.count),max_absolute_errors=dict(ck.errors),CPU_wall_seconds=time.monotonic()-tick,
        scope='Anonymous arithmetic, source pairing, bootstrap, tails, targets, diagnostic counts, optional CSV/decision; excludes private geometry, annotations and state reconstruction.',time=time.time())


def iou(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    inter=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1)
    union=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)+np.maximum(b[...,2:]-b[...,:2],0).prod(-1)-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def corners(b):
    b=np.asarray(array(b),float);return np.concatenate([b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2],-1)
def metric(pred,row,truth,span,ds,interval=None):
    """Literal independent dense formula: no extrapolation; HC clipped predictions."""
    ids=np.array(row['frame_ids']);b=corners(pred['boxes'])*np.array([row['input']['width'],row['input']['height']]*2)
    if ds=='hc2':b=np.maximum(b,0)
    f=np.array(sorted(truth));t=np.array([truth[int(j)] for j in f])
    z=np.stack([np.interp(f,ids,b[:,j]) for j in range(4)],-1);s=iou(z,t);s[(f<ids[0])|(f>ids[-1])]=0
    a,h=map(int,pred['physical_interval'] if interval is None else interval);g,k=span
    overlap=max(0,min(h,k)-max(a,g));union=h-a+k-g-overlap;assert union>0
    mask=(f>=max(a,g))&(f<min(h,k))
    return dict(v=float(s[mask].sum()/max(max(h,k)-min(a,g),1)),t=overlap/union,s=float(s.mean()))


def geometry(central,candidates,coeff,positions=None):
    """Independent differentiable box geometry; no project loss code is called."""
    import torch
    p=central.unsqueeze(0);q=candidates.detach()
    ap,bp=p[...,:2]-p[...,2:]/2,p[...,:2]+p[...,2:]/2
    aq,bq=q[...,:2]-q[...,2:]/2,q[...,:2]+q[...,2:]/2
    overlap=(torch.minimum(bp,bq)-torch.maximum(ap,aq)).clamp_min(0).prod(-1)
    union=p[...,2:].prod(-1)+q[...,2:].prod(-1)-overlap
    enclosure=(torch.maximum(bp,bq)-torch.minimum(ap,aq)).prod(-1)
    gl=1-overlap/union+(enclosure-union)/enclosure
    l1=(p-q).abs().sum(-1)
    if positions is None:return coeff[0]*l1.mean(-1)+coeff[1]*gl.mean(-1)
    return (coeff[0]*l1+coeff[1]*gl)[:,positions].T


def ranks(values):
    order=np.argsort(-values,kind='stable');out=np.empty(len(values));a=0
    while a<len(values):
        b=a+1
        while b<len(values) and values[order[a]]-values[order[b]]<=1e-12:b+=1
        out[order[a:b]]=(a+b-1)/2;a=b
    return out


def root():
    import torch
    torch.set_num_threads(2);tick=time.monotonic();ck=Checker()
    lock=read(BASE/'RUNTIME_LOCK.json');pins=dict(lock['pins'])
    for p in sorted((BASE/'revisions').glob('*.json')):pins.update(read(p)['pin_overrides'])
    for f,h in {**pins,**lock['inputs']}.items():assert sha(ROOT/f)==h,f;ck.count['runtime_pins']+=1
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==lock['CURRENT_METHOD_sha256']
    bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='sealed' and bar['GT_read'] is False
    for f,h in bar['files'].items():assert sha(BASE/f)==h,f;ck.count['sealed_files']+=1
    for ds in ('vidstg','hc2'):
        for stage in ('local','reset_u'):
            f=BASE/ds/(stage+'_RESOURCES.json');assert str(f.relative_to(BASE)) in bar['files']
            resource=read(f);assert resource['GT_read'] is False
            assert resource['new_backbone_calls']==resource['new_expert_calls']==0
    assert bar['time']<read(BASE/'GT_EXPOSURE.json')['time']<=read(BASE/'SCORE_COMPLETION.json')['time']
    labels={};plans={};cells=read(BASE/'COHORT.json')['cells'];assert len(cells)==1152
    for ds in ('vidstg','hc2'):
        plans[ds]=read(VIEW/ds/'PLAN.json');labels[ds]={}
        for sp in ('search','confirm'):
            p=POOL/ds/f'GT_LABELS_{sp}.json';assert sha(p)==lock['GT_inputs'][str(p.relative_to(ROOT))];labels[ds][sp]=read(p)
    def load(p):return torch.load(p,map_location='cpu',weights_only=False)
    def payload(stem):
        p=Path(str(stem)+'.pt');r=read(p.with_suffix('.json'))
        assert str(p.relative_to(BASE)) in bar['files'] and str(p.with_suffix('.json').relative_to(BASE)) in bar['files']
        assert r['GT_read'] is False and r['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        assert r['time']<bar['time'] and sha(p)==r['sha256'];ck.count['receipts']+=1
        z=load(p);assert z['GT_read'] is False;return z
    pubrows={scope:{r['cell_key']:r for r in read(PUB/name)} for scope,name in [('current','ROWS.json'),('future','FUTURE_ROWS.json'),('reset_u','RESET_U_ROWS.json')]}
    publicsteps={(r['cell_key'],r['arm'],r['step']):r for r in read(PUB/'STEP_DIAGNOSTICS.json')}
    initial={};previous={}
    def states(a,b,kind):
        expected={'spatial.query_residual'}|{f'spatial.layers.5.{n}.{t}' for n in ('norm1','norm3','norm4') for t in ('weight','bias')}
        assert set(a)==set(b)==expected and all(tuple(v.shape)==(256,) for v in a.values())
        for n in a:ck.close(a[n],b[n],kind,0)
    def sgd(pre,post,grads,lr):
        assert set(pre)==set(post)==set(grads)
        for n,v in pre.items():
            expected=(array(v).astype(np.float64)-lr*array(grads[n]).astype(np.float64)).astype(array(v).dtype)
            ck.close(expected,post[n],'SGD_coordinates',2e-7)
    def check_math(s,e,arm):
        cs=torch.stack([v['boxes'] for v in s['candidates']]);assert len(cs)==9
        center=s['pre_prediction']['boxes'].clone().requires_grad_(True);assert torch.equal(cs[0],center.detach())
        pos=np.flatnonzero(array(e['valid']).astype(bool));expert=corners(array(e['boxes'])[pos]);candidate=corners(array(cs)[:,pos])
        re=iou(candidate,expert[None]).T if len(pos) else np.empty((0,9))
        ev=np.maximum(re[:,:1]-re,0)
        ck.close(s['valid_positions'],pos,'valid_positions',0);ck.close(s['framewise_rewards'],re,'framewise_rewards',1e-6);ck.close(s['e_jk'],ev,'negative_evidence',1e-6)
        local=arm=='negative_local';d=geometry(center,cs,s['coefficients'],pos if local else None)
        lp=(-d).log_softmax(-1);lp0=lp.detach();en=torch.as_tensor(ev if local else (ev.mean(0) if len(ev) else np.zeros(9)),dtype=center.dtype)
        if arm.startswith('rank') and s['rewards'] is not None:
            ck.close(s['rewards'],re.mean(0),'rank_rewards',1e-10)
            lq=(-torch.as_tensor(ranks(re.mean(0)),dtype=center.dtype)/s['teacher_temperature']).log_softmax(0)
        else:lq=(lp0-en).log_softmax(-1) if (en!=0).any() else lp0.clone()
        no_op=(not len(pos) or ((en==0).all() and arm.startswith('negative')))
        loss=center.sum()*0 if no_op else (lp.exp()*(lp-lq.detach())).sum(-1).mean()
        grad=torch.autograd.grad(loss,center)[0]
        ck.close(s['distance'],d.detach(),'geometry_distances',3e-6,2e-6)
        ck.close(s['logp'],lp.detach(),'student_distribution',4e-6)
        ck.close(s['logp0'],lp0,'frozen_anchor',4e-6)
        ck.close(s['logq'],lq.detach(),'frozen_target',4e-6)
        ck.close(s['loss'],float(loss.detach()),'objective',5e-6,5e-6)
        ck.close(s['output_gradient'],grad,'independent_output_gradient',3e-5,5e-4)
        after=s['post_prediction']['boxes'];dp=geometry(after,cs,s['coefficients'],pos if local else None);la=(-dp).log_softmax(-1)
        afterloss=0. if no_op else float((la.exp()*(la-lq.detach())).sum(-1).mean())
        ck.close(s['logp_after'],la,'post_distribution',4e-6);ck.close(s['loss_after'],afterloss,'post_objective',6e-6,6e-6)
        if arm.startswith('negative') and len(pos):
            proximal=(lp.exp()*(lp-lp0+en)).sum(-1).mean()
            constant=torch.logsumexp(lp0-en,-1).mean()
            ck.close(float(loss.detach()),float((proximal+constant).detach()),'negative_KL_identity',2e-6)
        if no_op:
            assert np.count_nonzero(array(s['output_gradient']))==0
            states(s['pre_state'],s['post_state'],'explicit_noop_state')
        if local:
            unknown=np.ones(len(center),bool);unknown[pos]=False
            assert np.count_nonzero(array(s['output_gradient'])[unknown])==0;ck.count['local_gradient_support']+=1
        return pos,ev
    for c in cells:
        ds=c['dataset'];k=key(c);row=plans[ds]['rows'][c['parent']];g=labels[ds][c['split']][str(c['parent'])]
        truth={int(f):v for f,v in g['truth'].items()};oldpath=ROOT/c['old_payload'];assert sha(oldpath)==read(oldpath.with_suffix('.json'))['sha256'];old=load(oldpath)
        interval=[row['frame_ids'][old['final_indices'][0]],row['frame_ids'][old['final_indices'][1]]+1]
        z=payload(BASE/'reset_u'/k);rz=z['result'];stream=tuple(c[f] for f in ('dataset','split','condition','order'))
        if c['arrival']==0:
            if ds in initial:states(z['inherited_before_reset'],initial[ds],'stream_source_reset')
            else:initial[ds]=z['inherited_before_reset']
        else:states(z['inherited_before_reset'],previous[stream],'LN_inheritance_before_reset')
        for n,v in rz['pre_state'].items():
            ck.close(v,np.zeros_like(array(v)) if n=='spatial.query_residual' else z['inherited_before_reset'][n],'query_reset_transition',0)
        assert shstate(rz['pre_state'])==rz['pre_state_sha256'] and shstate(rz['post_state'])==rz['post_state_sha256']
        previous[stream]=rz['post_state'];state=rz['pre_state']
        for step in rz['update_steps']:
            states(step['pre_state'],state,'reset_inner_chain');u=step['update']
            if u is not None:
                assert u['lr']==BUNDLES[ds]['lr'];sgd(step['pre_state'],step['post_state'],u['gradients'],u['lr'])
                ck.close(np.exp(-np.asarray(array(u['distances'])))/np.exp(-np.asarray(array(u['distances']))).sum(),u['p'],'reset_student',2e-6)
                qq=np.exp(-ranks(np.asarray(step['rewards']))/BUNDLES[ds]['teacher_temperature']);qq/=qq.sum()
                ck.close(qq,u['q'],'reset_rank_target',2e-6)
            else:states(step['pre_state'],step['post_state'],'reset_no_update')
            state=step['post_state']
        states(state,rz['post_state'],'reset_final_chain')
        assert torch.equal(rz['output_prediction']['boxes'],rz['prediction']['boxes'])
        r=pubrows['reset_u'][k]
        for name,p,ii in [('A',old['slow'],interval),('reset_u',rz['output_prediction'],None),('reset_u_fixedA',rz['output_prediction'],interval)]:
            m=metric(p,row,truth,g['span'],ds,ii)
            for a,v in m.items():ck.close(v,r[name+'_'+a],'independent_dense_reset')
        if not c['scheduled']:continue
        x=payload(BASE/'local'/k);states(x['pre_state'],old['pre_state'],'same_A_prestate')
        assert x['old_payload_sha256']==sha(oldpath) and x['persistent_unchanged']
        er=x['evidence']['receipt'];assert sha(ROOT/er['path'])==er['sha256'] and er['pixel_sha256']==c['pixel_sha256']
        ex=load(ROOT/er['path']);ck.close(x['evidence']['boxes'],ex['boxes'],'cached_expert_boxes',0);ck.close(x['evidence']['valid'],ex['valid'],'cached_expert_valid',0)
        fc=x['future_cell'];assert fc['arrival']==c['arrival']+1 and not fc['scheduled']
        fold=load(ROOT/fc['old_payload']);states(fold['pre_state'],old['post_state'],'future_baseline_A_state')
        fr=plans[ds]['rows'][fc['parent']];fg=labels[ds][fc['split']][str(fc['parent'])];ft={int(i):v for i,v in fg['truth'].items()}
        fi=[fr['frame_ids'][fold['final_indices'][0]],fr['frame_ids'][fold['final_indices'][1]]+1]
        assert x['future_baseline']['physical_interval']==fi
        for scope,p,rr,t,sp in [('current',x['pre_prediction'],row,truth,g['span']),('future',x['future_baseline'],fr,ft,fg['span'])]:
            for m,v in metric(p,rr,t,sp,ds).items():ck.close(v,pubrows[scope][k]['pre_'+m],'independent_dense_pre')
        candidates0=x['arms']['rank_native']['steps'][0]['candidates']
        for arm in ARMS:
            a=x['arms'][arm];bundle=BUNDLES[('hc2' if ds=='vidstg' else 'vidstg') if arm=='rank_cross' else ds]
            assert a['config']==bundle and len(a['steps'])<=bundle['steps']
            assert len(a['steps'])==(bundle['steps'] if np.asarray(array(x['evidence']['valid'])).any() else 1)
            state=x['pre_state'];last=None
            for step in a['steps']:
                assert step['lr']==bundle['lr'] and step['teacher_temperature']==bundle['teacher_temperature']
                assert step['coefficients']==([5,3] if ds=='vidstg' else [5,4])
                states(step['pre_state'],state,'local_inner_chain');assert shstate(step['pre_state'])==step['pre_state_sha256'] and shstate(step['post_state'])==step['post_state_sha256']
                if step['step']==0:
                    for p,q in zip(step['candidates'],candidates0):ck.close(p['boxes'],q['boxes'],'first_step_shared_support',0)
                pos,ev=check_math(step,x['evidence'],arm);sgd(step['pre_state'],step['post_state'],step['gradients'],step['lr'])
                for n,v in step['gradients'].items():ck.close(np.linalg.norm(array(v).astype(float)),step['gradient_block_norms'][n],'gradient_norm',1e-9)
                gn=np.sqrt(sum(np.square(array(v).astype(float)).sum() for v in step['gradients'].values()))
                ck.close(step['eta_gradient_norm'],step['lr']*gn,'eta_gradient_norm',1e-9)
                if arm=='rank_native':
                    os=old['update_steps'][step['step']];states(step['pre_state'],os['pre_state'],'native_old_pre_parity');states(step['post_state'],os['post_state'],'native_old_post_parity')
                    if os['update']:
                        for n,v in step['gradients'].items():ck.close(v,os['update']['gradients'][n],'native_old_gradient_parity',0)
                publicstep=publicsteps[(k,arm,step['step'])]
                displacement=np.sqrt(sum(np.square(array(step['post_state'][n]).astype(float)-array(v).astype(float)).sum() for n,v in step['pre_state'].items()))
                ck.close(publicstep['parameter_displacement'],displacement,'actual_parameter_displacement',1e-10)
                for name in ('pre_prediction','post_prediction'):
                    first=interval
                    for m,v in metric(step[name],row,truth,g['span'],ds,first).items():ck.close(v,publicstep[name+'_'+m],'independent_dense_step')
                gt=[];known=[]
                for j in pos:
                    fid=row['frame_ids'][j];is_known=(g['span'][0]<=fid<g['span'][1]) and int(fid) in truth;known.append(is_known)
                    b=corners(np.stack([array(p['boxes'])[j] for p in step['candidates']]))*np.array([row['input']['width'],row['input']['height']]*2)
                    if ds=='hc2':b=np.maximum(b,0)
                    gt.append(iou(b,truth[int(fid)]) if is_known else np.zeros(9))
                gt=np.asarray(gt).reshape(-1,9);known=np.asarray(known,bool);better=gt[known,1:]>gt[known,:1]+1e-12;ee=ev[known,1:]
                ck.close([known.sum(),(ee>0).sum(),((ee>0)&better).sum(),ee.sum(),ee[better].sum()],
                    [publicstep['observed_GT_frames'],publicstep['known_negative_entries'],publicstep['negative_GT_better_entries'],publicstep['known_negative_evidence_mass'],publicstep['negative_GT_better_evidence_mass']],
                    'independent_negative_GT_misfire',2e-6)
                sg=np.array([metric(p,row,truth,g['span'],ds,interval)['s'] for p in step['candidates']])
                eg=ev.mean(0) if len(ev) else np.zeros(9);gb=sg[1:]>sg[0]+1e-12
                ck.close(sg,publicstep['full_event_GT_spatial_candidate_quality'],'independent_full_event_candidates')
                ck.close([(eg[1:]>0).sum(),((eg[1:]>0)&gb).sum(),eg[1:].sum(),eg[1:][gb].sum()],
                    [publicstep['global_negative_candidates'],publicstep['global_negative_GT_better_candidates'],publicstep['global_negative_evidence_mass'],publicstep['global_negative_GT_better_evidence_mass']],
                    'independent_global_negative_GT_misfire',2e-6)
                state=step['post_state'];last=step
            states(state,a['post_state'],'local_final_state');ck.close(a['post_prediction']['boxes'],last['post_prediction']['boxes'],'arrival_post_boxes',0)
            assert a['post_prediction']['physical_interval']==interval and a['future_prediction']['physical_interval']==fi
            if arm=='rank_native':
                states(a['post_state'],fold['pre_state'],'native_future_A_prestate')
                ck.close(a['future_prediction']['boxes'],fold['slow']['boxes'],'native_future_A_boxes',0)
            for scope,p,rr,t,sp in [('current',a['post_prediction'],row,truth,g['span']),('future',a['future_prediction'],fr,ft,fg['span'])]:
                for m,v in metric(p,rr,t,sp,ds).items():ck.close(v,pubrows[scope][k][arm+'_'+m],'independent_dense_local_future')
            assert set(a['block_counterfactual_predictions'])==set(BLOCKS)
            for b,p in a['block_counterfactual_predictions'].items():
                assert p['physical_interval']==interval
                for m,v in metric(p,row,truth,g['span'],ds).items():ck.close(v,pubrows['current'][k][arm+'_'+b+'_'+m],'independent_dense_blocks')
    assert not torch.cuda.is_initialized()
    out=dict(status='pass',checks=dict(ck.count),max_absolute_errors=dict(ck.errors),CPU_wall_seconds=time.monotonic()-tick,
        scope='Sealed receipts/runtime/private GT pins; same-prestate and native parity; nine-support first-step equality; independent IoU evidence/geometry/target/output gradient; SGD coordinates and lifecycle; independent dense metrics and GT-supported misfire counts.',
        limitations=['No model forward: parameter Jacobian and block-only replay outputs are not independently re-executed.',
                     'No-GT flags, code guard and receipts establish a protocol audit, not proof against arbitrary process tampering.',
                     'Future probe is one isolated write; write/target-source overlap may retain dependence beyond either one-way bootstrap.'],
        new_GPU_calls=0,new_model_calls=0,new_expert_calls=0,time=time.time())
    write(BASE/'ROOT_AUDIT.json',out);write(PUB/'ROOT_AUDIT.json',out);return out


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='root':result=root()
    else:
        directory=Path(sys.argv[1]) if len(sys.argv)>1 else PUB
        if len(sys.argv)>1 and sys.argv[1]=='public':directory=Path(sys.argv[2]) if len(sys.argv)>2 else PUB
        result=public(directory);write(directory/('PUBLIC_REPORT_AUDIT.json' if (directory/'PUBLIC_AUDIT.json').exists() else 'PUBLIC_AUDIT.json'),result)
    print(json.dumps(result,indent=2))
