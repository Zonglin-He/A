"""Independent matched R2 selector/gradient/state/dense and public scalar audits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tastvg_dta_oracle_r1_v1 import read,write,sha,summary,pre,cellkey
PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04';BASE=ROOT/'artifacts/tastvg_dta_expert_r2_v1'
R1=ROOT/'artifacts/tastvg_dta_oracle_r1_v1';R1PUB=ROOT/'results/tastvg_dta_oracle_r1/2026-10-04'
LAT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1';POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
PRE=ROOT/'artifacts/tastvg_temporal_boundary_support_v1';VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
ARMS=['EDeploy','EOracle']

def public_audit(directory=PUB):
    tick=time.monotonic();directory=Path(directory);counter=collections.Counter();maxerr=0.
    cfg=read(directory/'CONFIG.json');rows=read(directory/'ROWS.json');saved=read(directory/'SUMMARY.json')
    deploy=read(directory/'DEPLOY_PREDICTION_BARRIER.json');oracle=read(directory/'ORACLE_PREDICTION_BARRIER.json');glob=read(directory/'GLOBAL_PREDICTION_BARRIER.json')
    assert cfg['head_parameters']==514 and cfg['K']==3 and cfg['beta']==1 and cfg['lr']=={'vidstg':.01,'hc2':.01}
    assert cfg['source_lr_inherited'] and not cfg['source_retuning'] and not cfg['target_tuning'] and cfg['spatial_A_fixed']
    assert not any(cfg[k] for k in ['mixture','PoE','confidence_weighting','gate','R2b_started','R3_started'])
    assert deploy['time']<oracle['time']<glob['time'] and deploy['pid']!=oracle['pid']
    assert deploy['GT_guard_active'] and not deploy['GT_read'] and oracle['GT_read'] and glob['all_predictions_sealed_before_dense']
    assert deploy['cells']==oracle['cells']==288 and glob['query_arm_adaptations']==576
    assert len(rows)==1152 and sum(r['expert_scheduled'] for r in rows)==288
    index={r['cell_key']:r for r in rows};assert len(index)==1152
    def close(a,b,tol=1e-10):
        nonlocal maxerr
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        d=float(np.max(np.abs(a-b))) if a.size else 0.;assert d<=tol,(d,tol)
        maxerr=max(maxerr,d);counter['numeric_scalars']+=a.size
    for r in rows:
        for arm in ARMS:
            assert r[arm+'_GT_supervised']==(r['expert_scheduled'] and arm=='EOracle')
            for b in ['N','A8','R1']:
                for m in ['v','t']:close(r[arm+'_minus_'+b+'_'+m],r[arm+'_'+m]-r[b+'_'+m])
                d=r[arm+'_minus_'+b+'_v'];close(r[arm+'_gross_gain_'+b],max(d,0));close(r[arm+'_gross_loss_'+b],max(-d,0))
            if not r['expert_scheduled']:
                for name in [arm,arm+'_direct',arm+'_map']:
                    close([r[name+'_v'],r[name+'_t']],[r['A8_v'],r['A8_t']])
        for m in ['v','t']:close(r['EOracle_minus_EDeploy_'+m],r['EOracle_'+m]-r['EDeploy_'+m])
    for arm in ARMS:
        assert sha(directory/(arm+'_TRACES.json'))==(deploy if arm=='EDeploy' else oracle)['trace_sha256']
        traces=read(directory/(arm+'_TRACES_SCORED.json'));assert len(traces)==288
        for q in traces:
            r=index[q['cell_key']];assert q['arm']==arm and q['lr']==.01 and q['episodic_reset'] and q['GT_supervised']==(arm=='EOracle')
            assert q['A_state_pre_sha256']==r['A_state_pre_sha256'] and q['A_state_post_sha256']==r['A_state_post_sha256']
            assert len(q['trace'])==3
            for j,s in enumerate(q['trace']):
                assert s['step']==j+1;close(s['loss'],s['teacher_KL']+s['prior_KL']);close(s['after_loss'],s['after_teacher_KL']+s['after_prior_KL'])
                close(s['step_displacement'],.01*s['gradient_norm'],2e-6);assert s['bias_gradient_norm']<1e-5
                if j:close(s['loss'],q['trace'][j-1]['after_loss'])
            close([q['trace'][-1]['v'],q['trace'][-1]['t']],[r[arm+'_v'],r[arm+'_t']])
    selections=read(directory/'TEACHER_SELECTION_SCORED.json');dsels={q['cell_key']:q for q in read(directory/'DEPLOY_SELECTION.json')}
    assert len(selections)==576 and len(dsels)==288
    for q in selections:
        assert q['GT_used_in_selection']==(q['arm']=='EOracle') and 0<=q['selected_index']<q['proposal_count']
        if q['arm']=='EDeploy':
            z=dsels[q['cell_key']];assert not z['GT_used'] and z['selected_index']==q['selected_index'] and z['proposal_count']==q['proposal_count'];close(z['selected_confidence'],q['confidence'])
    for ds in saved:
        for sp,panels in saved[ds].items():
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp]
            predicates=dict(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition']!='clean',expert_clean=lambda r:r['expert_scheduled'] and r['condition']=='clean',
                expert_all=lambda r:r['expert_scheduled'],flow_corrupt=lambda r:r['condition']!='clean',flow_clean=lambda r:r['condition']=='clean',nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition']!='clean')
            for name,z in panels.items():
                q=[r for r in rr if predicates[name](r)];fields=list(z['metrics']);v=summary(q,fields)
                assert v['sources']==z['sources'] and v['cells']==z['cells']
                for k in fields:
                    for f in ['mean','ci95','cell_macro','source_positive','source_negative']:close(v['metrics'][k][f],z['metrics'][k][f])
                    for s in v['metrics'][k]['source_values']:close(v['metrics'][k]['source_values'][s],z['metrics'][k]['source_values'][s])
                for arm in ARMS:
                    for b,t in z['negative_tails'][arm].items():
                        expected=dict(severe_harm=sum(r[arm+'_minus_'+b+'_v']<-.05 for r in q),harm=sum(r[arm+'_minus_'+b+'_v']<-1e-12 for r in q),gain=sum(r[arm+'_minus_'+b+'_v']>1e-12 for r in q),
                            baseline_good_destroyed_at_03=sum(r[b+'_v']>=.3 and r[arm+'_v']<.3 for r in q),baseline_bad_rescued_at_03=sum(r[b+'_v']<.3 and r[arm+'_v']>=.3 for r in q))
                        assert expected==t;counter['tail_counts']+=len(t)
                    assert z['loss_down_task_down'][arm]==sum(r.get(arm+'_loss_after',0)<r.get(arm+'_loss_before',0) and r[arm+'_minus_N_v']<-1e-12 for r in q)
                for o,t in z['orders'].items():
                    v=summary([r for r in q if r['order']==o],fields);assert v['cells']==t['cells'] and v['sources']==t['sources']
                    for k in fields:close(v['metrics'][k]['mean'],t['metrics'][k]['mean']);close(v['metrics'][k]['ci95'],t['metrics'][k]['ci95'])
    return dict(status='pass',checks=dict(counter),max_absolute_error=maxerr,CPU_wall_seconds=time.monotonic()-tick,
        scope='Anonymous paired arithmetic, traces, deploy/oracle timing, tails, orders and independently resampled 10000 source bootstrap; no private GT/weights.',time=time.time())

def root_audit():
    import torch
    from scipy.special import logsumexp
    torch.set_num_threads(2);tick=time.monotonic();count=collections.Counter();maxerrors=collections.defaultdict(float)
    cfg=read(PUB/'CONFIG.json');lock=read(BASE/'RUNTIME_LOCK.json');global_bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    for f,h in {**lock['code'],**lock['inputs'],**lock['oracle_label_inputs']}.items():assert sha(ROOT/f)==h,f;count['immutable_hashes']+=1
    bars={a:read(BASE/('DEPLOY_PREDICTION_BARRIER.json' if a=='EDeploy' else 'ORACLE_PREDICTION_BARRIER.json')) for a in ARMS}
    oracle_sel=read(BASE/'ORACLE_SELECTION_BARRIER.json');assert bars['EDeploy']['time']<oracle_sel['time']<bars['EOracle']['time']<global_bar['time']
    assert bars['EDeploy']['pid']!=bars['EOracle']['pid'] and bars['EDeploy']['GT_guard_active'] and not bars['EDeploy']['GT_read']
    assert sha(BASE/'ORACLE_SELECTION.json')==oracle_sel['selection_sha256']
    assert cfg['shared_R1_fit_code_sha256']==sha(ROOT/'vg_tta/tastvg_dta_oracle_v1.py')
    source_bar=read(R1/'SOURCE_SELECTION_BARRIER.json');assert source_bar['choices']==cfg['lr']=={'vidstg':.01,'hc2':.01}
    rows={r['cell_key']:r for r in read(PUB/'ROWS.json')};r1rows={r['cell_key']:r for r in read(R1PUB/'ROWS.json')};cohort=read(BASE/'COHORT.json')['cells']
    support=read(BASE/'EXPERT_SUPPORT.json');osel=read(BASE/'ORACLE_SELECTION.json')
    traces={arm:{r['cell_key']:r for r in read(PUB/(arm+'_TRACES_SCORED.json'))} for arm in ARMS}
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
            close(q,a['logq'][len(grids)],'teacher_gaussian',1e-10);close(p,a['logp0'][len(grids)],'frozen_prior',1e-10)
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
        assert a['after']==a['trace'][-1]['prediction'] and a['spatial_changed'] is False and a['episodic_reset'] and isinstance(a['GT_supervised'],bool)
        count['episodic_queries']+=1;count['independent_gradient_steps']+=3

    from scripts.tastvg_correction_views_common_v1 import oldcell
    from vg_tta.tastvg_oracle_event5_v1 import official
    for ds in ['vidstg','hc2']:
        head=load(R1/ds/'HEAD.pt');cp=load(ROOT/cfg['checkpoints'][ds]['path'])['model_ema']
        for k in head:close(head[k],cp['temp_embed.layers.'+k],'checkpoint_head',0)
        del cp
        plan=read(VIEW/ds/'PLAN.json');labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cohort if c['dataset']==ds]:
            k=cellkey(c);r=rows[k];g=labels[c['split']][str(c['parent'])];row=plan['rows'][c['parent']];ids=row['frame_ids'];truth={int(j):v for j,v in g['truth'].items()}
            for name,value in r1rows[k].items():assert r[name]==value,(k,name,'R1 evidence changed');count['R1_preserved_fields']+=1
            old=oldcell(ds,c['split'],c['condition'],c['order'],c['arrival']);assert old['pre_sha']==c['pre_sha'] and old['post_sha']==c['post_sha']
            p=read(PRE/ds/'predictions'/f'{pre(c)}.json');A8=[ids[p['A_indices'][0]],ids[p['A_indices'][1]]+1]
            intervals={'A8':A8}
            if c['scheduled']:
                z=load(LAT/ds/'target_features'/f'{pre(c)}.pt');raw=np.asarray(support[k]['proposals']);confidence=np.asarray(support[k]['confidence'])
                # Reopen the hashed expert cache, not just its copied support summary.
                ep=read(POOL/ds/'experts/temporal'/c['condition']/f"{c['parent']:05}.json");ec=load(POOL/ds/'experts'/ep['cache'])
                close(raw,ec['proposals'],'raw_support',0);close(confidence,ec['proposal_confidence'],'raw_confidence',0)
                ov=np.maximum(0,np.minimum(raw[:,1],g['span'][1])-np.maximum(raw[:,0],g['span'][0]));ts=ov/(raw[:,1]-raw[:,0]+g['span'][1]-g['span'][0]-ov)
                chosen={'EDeploy':int(np.argmax(confidence)),'EOracle':int(np.argmax(ts))}
                assert chosen['EDeploy']==support[k]['deploy']['index'] and chosen['EOracle']==osel[k]['index'];count['independent_teacher_selections']+=2
                rf=read(POOL/ds/'capture'/c['condition']/f"{c['parent']:05}.json");capture=load(POOL/ds/rf['cache']);r1a=load(R1/ds/'target_runs'/f'{pre(c)}.pt')
                for arm in ARMS:
                    f=BASE/ds/('deploy_runs' if arm=='EDeploy' else 'oracle_runs')/f'{pre(c)}.pt';assert sha(f)==bars[arm]['files'][str(f.relative_to(BASE))]
                    a=load(f);span=raw[chosen[arm]].tolist();assert a['teacher_proposal_index']==chosen[arm] and a['teacher_center']==span and a['GT_supervised']==(arm=='EOracle')
                    check_fit(a,z,head,span,.01)
                    assert a['A_state_pre_sha256']==c['pre_sha'] and a['A_state_post_sha256']==c['post_sha'] and a['pixel_sha256']==c['pixel_sha256']
                    assert a['before']==r1a['before'] and a['before']['indices']==capture['prediction']['indices']==p['native_indices']
                    close(a['logits'][0],r1a['logits'][0],'R1_initial_logits',0)
                    for j,at in enumerate([list(range(q,len(ids),2)) for q in [0,1]]):
                        close(a['logp0'][j],r1a['logp0'][j],'R1_prior',0);close(a['logits'][0][at],capture['prediction']['logits'][j].reshape(-1,2),'original_CUDA_CPU_logits',1e-4)
                    teacher=[]
                    for j,at in enumerate([list(range(q,len(ids),2)) for q in [0,1]]):
                        ii,jj=np.triu_indices(len(at),1);q=np.asarray(a['logq'][j]);ix=int(np.argmax(q));teacher.append([at[ii[ix]],at[jj[ix]]])
                    assert a['teacher_MAP']['indices']==[min(i[0] for i in teacher),max(i[1] for i in teacher)];count['teacher_MAP_checks']+=1
                    intervals.update({arm:a['after']['physical_interval'],arm+'_direct':span,arm+'_map':a['teacher_MAP']['physical_interval'],'N':a['before']['physical_interval'],'R1':r1a['after']['physical_interval']})
                    for j,s in enumerate(traces[arm][k]['trace']):
                        internal=a['trace'][j]
                        for name,value in internal.items():
                            if name=='prediction':continue
                            mapped='teacher_KL' if name=='GT_KL' else 'after_teacher_KL' if name=='after_GT_KL' else name
                            if isinstance(value,(int,float)):close(value,s[mapped],'public_trace_scalars',0)
                            else:assert value==s[mapped]
                        m=official(old['slow']['boxes'],row,truth,g['span'],internal['prediction']['physical_interval'],ds);close([m['v'],m['t']],[s['v'],s['t']],'step_dense_metrics')
            else:
                intervals.update({name:A8 for name in ['N','R1']+[arm+suffix for arm in ARMS for suffix in ['','_direct','_map']]})
            for name,interval in intervals.items():
                m=official(old['slow']['boxes'],row,truth,g['span'],interval,ds);close([m['v'],m['t']],[r[name+'_v'],r[name+'_t']],'official_dense_metrics')
            count['A_immutable_rows']+=1
    assert count['episodic_queries']==576 and count['independent_gradient_steps']==1728 and count['A_immutable_rows']==1152 and count['independent_teacher_selections']==576
    out=dict(status='pass',checks=dict(count),maximum_errors=dict(maxerrors),CPU_wall_seconds=time.monotonic()-tick,
        gradient_validation='NumPy joint-marginal gradients, every three-step SGD state/logit/native decode and teacher MAP independently recomputed.',
        evidence_scope='Deploy GT-free sealed first; oracle explicitly supervised; no target retuning or online persistence.',CUDA_initialized=torch.cuda.is_initialized(),time=time.time())
    write(PUB/'ROOT_AUDIT.json',out);write(BASE/'ROOT_AUDIT.json',out);return out

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='root':z=root_audit()
    else:
        d=Path(sys.argv[1]) if len(sys.argv)>1 else PUB;z=public_audit(d);p=d/'PUBLIC_AUDIT.json'
        if p.exists():assert z['checks']==read(p)['checks']
        else:write(p,z)
    print(json.dumps(z,indent=2))
