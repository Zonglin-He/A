"""Independent R1 gradient/SGD/native/dense root audit, and public scalar audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PUB=ROOT/'results/tastvg_dta_oracle_r1/2026-10-04'
BASE=ROOT/'artifacts/tastvg_dta_oracle_r1_v1'
LAT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
PRE=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,z):Path(p).write_text(json.dumps(z,indent=2)+'\n')
def pre(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def cellkey(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])

def summary(rows,fields):
    source=collections.defaultdict(list)
    for r in rows:source[r['source_id']].append([r[f] for f in fields])
    if not rows:return dict(cells=0,sources=0,metrics={})
    ids=sorted(source);x=np.array([np.mean(source[s],0) for s in ids]);rng=np.random.default_rng(20261004)
    boot=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    ci=np.percentile(boot,[2.5,97.5],axis=0)
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=20261004,
       metrics={f:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),source_values={str(s):float(v) for s,v in zip(ids,x[:,j])},
        source_positive=int((x[:,j]>1e-12).sum()),source_negative=int((x[:,j]<-1e-12).sum()),
        cell_macro=float(np.mean([r[f] for r in rows]))) for j,f in enumerate(fields)})

def public_audit(directory=PUB):
    tick=time.monotonic();directory=Path(directory);counter=collections.Counter();maxerr=0.
    cfg=read(directory/'CONFIG.json');rows=read(directory/'ROWS.json');traces=read(directory/'ADAPTATION_TRACES_SCORED.json')
    sels=read(directory/'SOURCE_SELECTION.json');sr=read(directory/'SOURCE_VALIDATION_ROWS.json');saved=read(directory/'SUMMARY.json')
    seal=read(directory/'SOURCE_SELECTION_BARRIER.json');ps=read(directory/'PREDICTION_BARRIER.json')
    assert cfg['GT_used_in_target_teacher'] and cfg['target_selection'] is False and cfg['head_parameters']==514
    assert cfg['K']==3 and cfg['beta']==1 and cfg['new_GPU_calls']==0 and cfg['spatial_A_fixed']
    assert len(rows)==1152 and sum(r['expert_scheduled'] for r in rows)==len(traces)==288 and len(sr)==235
    assert seal['time']<ps['time'] and ps['GT_used_in_teacher'] and ps['GT_used_for_target_selection'] is False
    assert sha(directory/'SOURCE_VALIDATION_ROWS.json')==seal['source_rows_sha256']
    assert sha(directory/'SOURCE_SELECTION.json')==seal['selection_sha256']
    assert sha(directory/'ADAPTATION_TRACES.json')==ps['adaptation_traces_sha256']

    def close(a,b,tol=1e-10):
        nonlocal maxerr
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        d=float(np.max(np.abs(a-b))) if a.size else 0.;assert d<=tol,(d,tol)
        maxerr=max(maxerr,d);counter['numeric_scalars']+=a.size

    for ds in ['vidstg','hc2']:
        path=[];q=[r for r in sr if r['dataset']==ds]
        assert len(q)==cfg['source_validation_queries'][ds]*5
        for lr in cfg['lr_grid']:
            rr=[r for r in q if r['lr']==lr];assert len(rr)==cfg['source_validation_queries'][ds]
            path.append(dict(lr=lr,queries=len(rr),mean_after_tIoU=float(np.mean([r['after_t'] for r in rr])),mean_before_tIoU=float(np.mean([r['before_t'] for r in rr]))))
        for a,b in zip(path,sels[ds]['path']):
            assert a.keys()==b.keys()
            for k in a:close(a[k],b[k])
        at=min(range(5),key=lambda j:(-path[j]['mean_after_tIoU'],path[j]['lr']))
        assert at==sels[ds]['selected_index'] and seal['choices'][ds]==sels[ds]['selected_lr']==path[at]['lr']
        assert len({r['initial_head_sha256'] for r in q})==1
    index={r['cell_key']:r for r in rows};assert len(index)==1152
    for r in rows:
        for b in ['N','A8']:
            for m in ['v','t']:close(r[f'R1_minus_{b}_{m}'],r['R1_'+m]-r[b+'_'+m])
            d=r['R1_minus_'+b+'_v'];close(r['R1_gross_gain_'+b],max(d,0));close(r['R1_gross_loss_'+b],max(-d,0))
        if not r['expert_scheduled']:
            close([r['R1_v'],r['R1_t']],[r['A8_v'],r['A8_t']]);assert not r['GT_supervised']
    for q in traces:
        r=index[q['cell_key']];assert q['lr']==seal['choices'][q['dataset']] and q['episodic_reset'] and q['original_native_indices_exact']
        assert q['GT_supervised'] and q['A_state_pre_sha256']==r['A_state_pre_sha256'] and q['A_state_post_sha256']==r['A_state_post_sha256']
        assert len(q['trace'])==3
        for j,s in enumerate(q['trace']):
            assert s['step']==j+1;close(s['loss'],s['GT_KL']+s['prior_KL']);close(s['after_loss'],s['after_GT_KL']+s['after_prior_KL'])
            # SGD norm identity up to FP32 parameter-rounding error.
            close(s['step_displacement'],q['lr']*s['gradient_norm'],2e-6)
            assert s['bias_gradient_norm']<1e-5
            if j:close(s['loss'],q['trace'][j-1]['after_loss'])
        close([q['trace'][-1]['v'],q['trace'][-1]['t']],[r['R1_v'],r['R1_t']])
    for ds in saved:
        for sp, panels in saved[ds].items():
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp]
            predicates=dict(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition']!='clean',expert_clean=lambda r:r['expert_scheduled'] and r['condition']=='clean',
                expert_all=lambda r:r['expert_scheduled'],flow_corrupt=lambda r:r['condition']!='clean',flow_clean=lambda r:r['condition']=='clean',
                nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition']!='clean')
            for name,z in panels.items():
                q=[r for r in rr if predicates[name](r)];fields=list(z['metrics']);v=summary(q,fields)
                assert v['sources']==z['sources'] and v['cells']==z['cells']
                for k in fields:
                    for f in ['mean','ci95','cell_macro','source_positive','source_negative']:close(v['metrics'][k][f],z['metrics'][k][f])
                    for s in v['metrics'][k]['source_values']:close(v['metrics'][k]['source_values'][s],z['metrics'][k]['source_values'][s])
                for b,t in z['negative_tails'].items():
                    expected=dict(severe_harm=sum(r[f'R1_minus_{b}_v']<-.05 for r in q),harm=sum(r[f'R1_minus_{b}_v']<-1e-12 for r in q),
                       gain=sum(r[f'R1_minus_{b}_v']>1e-12 for r in q),baseline_good_destroyed_at_03=sum(r[b+'_v']>=.3 and r['R1_v']<.3 for r in q),
                       baseline_bad_rescued_at_03=sum(r[b+'_v']<.3 and r['R1_v']>=.3 for r in q))
                    assert expected==t;counter['tail_counts']+=len(t)
                assert z['objective_task_disagreement']==sum(r.get('loss_after',0)<r.get('loss_before',0) and r['R1_minus_N_v']<-1e-12 for r in q)
                for o,t in z['orders'].items():
                    v=summary([r for r in q if r['order']==o],fields);assert v['cells']==t['cells'] and v['sources']==t['sources']
                    for k in fields:
                        close(v['metrics'][k]['mean'],t['metrics'][k]['mean']);close(v['metrics'][k]['ci95'],t['metrics'][k]['ci95'])
    out=dict(status='pass',checks=dict(counter),max_absolute_error=maxerr,CPU_wall_seconds=time.monotonic()-tick,
        scope='Anonymous scalar arithmetic, source selection, counts, tails, order/source aggregation and 10000 paired bootstrap. No private weights/latents/annotations reconstructed.',time=time.time())
    return out

def root_audit():
    import torch
    from scipy.special import logsumexp
    torch.set_num_threads(2);tick=time.monotonic();count=collections.Counter();maxerrors=collections.defaultdict(float)
    cfg=read(PUB/'CONFIG.json');lock=read(BASE/'RUNTIME_LOCK.json');bs=read(BASE/'SOURCE_SELECTION_BARRIER.json');bar=read(BASE/'PREDICTION_BARRIER.json')
    for f,h in {**lock['code'],**lock['inputs']}.items():assert sha(ROOT/f)==h,f;count['immutable_hashes']+=1
    assert bs['time']<bar['time'] and bs['target_GT_teacher_started'] is False
    traces={r['cell_key']:r for r in read(PUB/'ADAPTATION_TRACES_SCORED.json')};source_rows=read(PUB/'SOURCE_VALIDATION_ROWS.json')
    allrows={r['cell_key']:r for r in read(PUB/'ROWS.json')};cohort=read(BASE/'COHORT.json')['cells']
    sources=read(LAT/'SOURCE_INPUTS.json');sgt=read(LAT/'SOURCE_GT.json')

    def close(a,b,kind,tol=1e-9):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        d=float(np.max(np.abs(a-b))) if a.size else 0.;assert d<=tol,(kind,d,tol)
        count[kind+'_scalars']+=a.size;maxerrors[kind]=max(maxerrors[kind],d)

    def load(f):return torch.load(f,map_location='cpu',weights_only=False,mmap=True)
    def decode(z,ids):
        out=[]
        for at in [list(range(j,len(ids),2)) for j in [0,1]]:
            a=z[at];s=a[:,0].log_softmax(0);e=a[:,1].log_softmax(0);n=len(at)
            mask=(torch.ones(n,n)*-1e32).tril(0);ij=int((mask+s[:,None]+e[None,:]).flatten().argmax())
            i,j=divmod(ij,n);assert i<j;out.append([at[i],at[j]])
        pair=[min(x[0] for x in out),max(x[1] for x in out)]
        return dict(indices=pair,physical_interval=[ids[pair[0]],ids[pair[1]]+1],offset_indices=out)

    def interval_t(a,g):
        ov=max(0,min(a[1],g[1])-max(a[0],g[0]));return ov/(a[1]-a[0]+g[1]-g[0]-ov)

    def check_fit(a,z,head,span,lr):
        ids=z['frame_ids'];hidden=z['hidden'];x=torch.relu(torch.nn.functional.linear(hidden.float(),head['0.weight'],head['0.bias']))
        close(a['states'][0]['weight'],head['1.weight'],'head_reset',0);close(a['states'][0]['bias'],head['1.bias'],'head_reset',0)
        w=head['1.weight'].clone();b=head['1.bias'].clone();zz=torch.nn.functional.linear(x,w,b)
        close(zz,a['logits'][0],'initial_logits',0)
        assert decode(zz,ids)==a['before']
        grids=[];priors=[];teachers=[];pairs=[]
        for at in [list(range(j,len(ids),2)) for j in [0,1]]:
            f=np.array([ids[i] for i in at],float);i,j=np.triu_indices(len(at),1);sig=float(np.median(np.diff(f)))
            q=-((f[i]-span[0])**2+(f[j]+1-span[1])**2)/(2*sig**2);q-=logsumexp(q)
            zv=zz[at].double().numpy();p=zv[i,0]+zv[j,1];p-=logsumexp(p)
            close(q,a['logq'][len(grids)],'GT_gaussian',1e-10);close(p,a['logp0'][len(grids)],'frozen_prior',1e-10)
            # torch median uses lower-middle for an even-length physical spacing;
            # grids in this fixed cohort have the same central spacing values.
            close(sig,a['sigma_frames'][len(grids)],'sigma',0)
            grids.append(at);pairs.append((i,j));priors.append(p);teachers.append(q)
        def objective(zz):
            terms=[];dq=[];da=[]
            for at,(i,j),q,p0 in zip(grids,pairs,teachers,priors):
                zv=zz[at].double().numpy();p=zv[i,0]+zv[j,1];p-=logsumexp(p)
                dq.append(float(np.sum(np.exp(q)*(q-p))));da.append(float(np.sum(np.exp(p0)*(p0-p))))
                diff=(2*np.exp(p)-np.exp(q)-np.exp(p0))/2
                dz=np.zeros_like(zv);np.add.at(dz[:,0],i,diff);np.add.at(dz[:,1],j,diff)
                terms.append((at,dz))
            gz=np.zeros((len(ids),2))
            for at,dz in terms:gz[at]=dz
            return float(np.mean(dq)),float(np.mean(da)),gz
        for k in range(3):
            q,anchor,gz=objective(zz);s=a['trace'][k];state=a['states'][k+1]
            close([q,anchor,q+anchor],[s['GT_KL'],s['prior_KL'],s['loss']],'objectives',1e-9)
            gw=gz.T@x.double().numpy();gb=gz.sum(0)
            close(gw,state['weight_gradient'],'analytic_gradient',2e-5);close(gb,state['bias_gradient'],'analytic_gradient',2e-5)
            expected_w=a['states'][k]['weight'].clone().add_(state['weight_gradient'],alpha=-lr)
            expected_b=a['states'][k]['bias'].clone().add_(state['bias_gradient'],alpha=-lr)
            close(expected_w,state['weight'],'SGD_arithmetic',0);close(expected_b,state['bias'],'SGD_arithmetic',0)
            # Independent analytic-gradient execution, not the stored gradient.
            w.add_(torch.from_numpy(gw).float(),alpha=-lr);b.add_(torch.from_numpy(gb).float(),alpha=-lr)
            close(w,state['weight'],'analytic_execution',2e-6);close(b,state['bias'],'analytic_execution',2e-6)
            stored=torch.nn.functional.linear(x,state['weight'],state['bias']);close(stored,a['logits'][k+1],'state_logits',0)
            zz=stored;q,anchor,_=objective(zz)
            close([q,anchor,q+anchor],[s['after_GT_KL'],s['after_prior_KL'],s['after_loss']],'post_objectives',1e-9)
            assert decode(zz,ids)==s['prediction'];count['native_decodes']+=1
        assert a['after']==a['trace'][-1]['prediction'] and a['spatial_changed'] is False and a['episodic_reset'] and a['GT_supervised']
        count['episodic_queries']+=1;count['independent_gradient_steps']+=3

    from scripts.tastvg_correction_views_common_v1 import oldcell
    from vg_tta.tastvg_oracle_event5_v1 import official
    for ds in ['vidstg','hc2']:
        head=load(BASE/ds/'HEAD.pt');cp=load(ROOT/cfg['checkpoints'][ds]['path'])['model_ema']
        for k in head:close(head[k],cp['temp_embed.layers.'+k],'checkpoint_head',0)
        targetplan=read(VIEW/ds/'PLAN.json');src=[r for r in sources[ds] if r['split']=='validation']
        assert not {r['source'] for r in src}&{r['source'] for r in targetplan['rows']}
        assert not {r['input']['video_sha256'] for r in src}&{r['input']['video_sha256'] for r in targetplan['rows']}
        for r in src:
            z=load(LAT/ds/'source_features'/f'{r["index"]:04}.pt');span=sgt[ds][str(r['index'])]
            for lr in cfg['lr_grid']:
                a=load(BASE/ds/'source_runs'/f'{r["index"]:04}_{lr:g}.pt');check_fit(a,z,head,span,lr)
                e=next(x for x in source_rows if x['dataset']==ds and x['source_id']==r['index'] and x['lr']==lr)
                close([interval_t(a['before']['physical_interval'],span),interval_t(a['after']['physical_interval'],span),interval_t(a['teacher_MAP']['physical_interval'],span)],
                      [e['before_t'],e['after_t'],e['teacher_MAP_t']],'source_metrics')
        labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cohort if c['dataset']==ds]:
            r=allrows[cellkey(c)];g=labels[c['split']][str(c['parent'])];row=targetplan['rows'][c['parent']];ids=row['frame_ids']
            old=oldcell(ds,c['split'],c['condition'],c['order'],c['arrival']);assert old['pre_sha']==c['pre_sha'] and old['post_sha']==c['post_sha']
            p=read(PRE/ds/'predictions'/f'{pre(c)}.json');A8=[ids[p['A_indices'][0]],ids[p['A_indices'][1]]+1]
            if c['scheduled']:
                f=BASE/ds/'target_runs'/f'{pre(c)}.pt';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                a=load(f);z=load(LAT/ds/'target_features'/f'{pre(c)}.pt');check_fit(a,z,head,g['span'],bs['choices'][ds])
                assert a['A_state_pre_sha256']==c['pre_sha'] and a['A_state_post_sha256']==c['post_sha']
                assert a['source_selection_barrier_sha256']==sha(BASE/'SOURCE_SELECTION_BARRIER.json')
                rf=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json');capture=load(POOL/ds/rf['cache'])
                assert a['before']['indices']==capture['prediction']['indices']==p['native_indices']
                for j,at in enumerate([list(range(k,len(ids),2)) for k in [0,1]]):
                    close(a['logits'][0][at],capture['prediction']['logits'][j].reshape(-1,2),'original_CUDA_CPU_logits',1e-4)
                intervals=dict(N=a['before']['physical_interval'],A8=A8,R1=a['after']['physical_interval'],teacher_MAP=a['teacher_MAP']['physical_interval'],GT_time=g['span'])
                for k,s in enumerate(traces[cellkey(c)]['trace']):
                    m=official(old['slow']['boxes'],row,{int(k):v for k,v in g['truth'].items()},g['span'],a['trace'][k]['prediction']['physical_interval'],ds)
                    close([m['v'],m['t']],[s['v'],s['t']],'step_dense_metrics')
            else:intervals=dict(N=A8,A8=A8,R1=A8,teacher_MAP=A8,GT_time=g['span'])
            for arm,interval in intervals.items():
                m=official(old['slow']['boxes'],row,{int(k):v for k,v in g['truth'].items()},g['span'],interval,ds)
                close([m['v'],m['t']],[r[arm+'_v'],r[arm+'_t']],'official_dense_metrics')
            count['A_immutable_rows']+=1
    assert count['episodic_queries']==523 and count['independent_gradient_steps']==1569 and count['A_immutable_rows']==1152
    out=dict(status='pass',checks=dict(count),maximum_errors=dict(maxerrors),CPU_wall_seconds=time.monotonic()-tick,
       gradient_validation='Every joint marginal gradient independently computed with NumPy and every three-step SGD state/logit/decode checked.',
       GT_scope='Explicit supervised oracle; no target hyperparameter selection',CUDA_initialized=torch.cuda.is_initialized(),time=time.time())
    write(PUB/'ROOT_AUDIT.json',out);write(BASE/'ROOT_AUDIT.json',out)
    return out

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='root':z=root_audit()
    else:
        d=Path(sys.argv[1]) if len(sys.argv)>1 else PUB;z=public_audit(d)
        p=d/'PUBLIC_AUDIT.json'
        if p.exists():assert z['checks']==read(p)['checks']
        else:write(p,z)
    print(json.dumps(z,indent=2))
