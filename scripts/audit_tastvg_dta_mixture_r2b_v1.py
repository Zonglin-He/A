"""Independent NumPy mixture/gradient/SGD/dense audit, plus portable scalar audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_tastvg_dta_oracle_r1_v1 import read,write,sha,summary,pre,cellkey
PUB=ROOT/'results/tastvg_dta_mixture_r2b/2026-10-04';BASE=ROOT/'artifacts/tastvg_dta_mixture_r2b_v1'
R2=ROOT/'artifacts/tastvg_dta_expert_r2_v1';R2PUB=ROOT/'results/tastvg_dta_expert_r2/2026-10-04'
R1=ROOT/'artifacts/tastvg_dta_oracle_r1_v1';LAT=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3';PRE=ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1';BASELINES=['N','A8','R1','EDeploy','EOracle']
PREDICATES=dict(expert_corrupt=lambda r:r['expert_scheduled'] and r['condition']!='clean',expert_clean=lambda r:r['expert_scheduled'] and r['condition']=='clean',
    expert_all=lambda r:r['expert_scheduled'],flow_corrupt=lambda r:r['condition']!='clean',flow_clean=lambda r:r['condition']=='clean',nonexpert_corrupt=lambda r:not r['expert_scheduled'] and r['condition']!='clean')

def public_audit(directory=PUB):
    tick=time.monotonic();directory=Path(directory);counter=collections.Counter();maxerr=0.
    cfg=read(directory/'CONFIG.json');rows=read(directory/'ROWS.json');saved=read(directory/'SUMMARY.json')
    bar=read(directory/'MIX_PREDICTION_BARRIER.json');glob=read(directory/'GLOBAL_PREDICTION_BARRIER.json');tr=read(directory/'MIX_TRACES_SCORED.json')
    assert cfg['head_parameters']==514 and cfg['K']==3 and cfg['beta']==1 and cfg['lr']=={'vidstg':.01,'hc2':.01}
    assert cfg['source_lr_inherited'] and not cfg['source_retuning'] and not cfg['target_tuning'] and cfg['spatial_A_fixed']
    assert cfg['mixture'] and cfg['teacher_joint_preserved'] and cfg['raw_proposals_preserved']
    assert not any(cfg[k] for k in ['PoE','confidence_weighting','top_K','deduplication','gate','new_view','product_of_mixed_marginals','R3_started'])
    assert bar['time']<glob['time'] and bar['GT_guard_active'] and not bar['GT_read'] and not glob['mix_GT_read']
    assert glob['all_new_predictions_sealed_before_GT_scoring'] and bar['cells']==glob['new_query_arm_adaptations']==288
    assert bar['backward_calls']==864 and not bar['CUDA_initialized'] and bar['spatial_unchanged']
    assert sha(directory/'MIX_TRACES.json')==bar['trace_sha256']
    assert len(rows)==1152 and sum(r['expert_scheduled'] for r in rows)==len(tr)==288
    index={r['cell_key']:r for r in rows};assert len(index)==1152
    def close(a,b,tol=1e-10):
        nonlocal maxerr
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        d=float(np.max(np.abs(a-b))) if a.size else 0.;assert d<=tol,(d,tol)
        maxerr=max(maxerr,d);counter['numeric_scalars']+=a.size
    for r in rows:
        assert r['EMix_GT_supervised'] is False and r['A_spatial_unchanged']
        for b in BASELINES:
            for m in ['v','t']:close(r['EMix_minus_'+b+'_'+m],r['EMix_'+m]-r[b+'_'+m])
            if b in ['N','A8','EDeploy']:
                d=r['EMix_minus_'+b+'_v'];close(r['EMix_gross_gain_'+b],max(d,0));close(r['EMix_gross_loss_'+b],max(-d,0))
        if not r['expert_scheduled']:
            for name in ['EMix','EMix_map']:close([r[name+'_v'],r[name+'_t']],[r['A8_v'],r['A8_t']])
    unscored={q['cell_key']:q for q in read(directory/'MIX_TRACES.json')}
    for q in tr:
        r=index[q['cell_key']];assert q['arm']=='EMix' and q['lr']==.01 and q['episodic_reset'] and not q['GT_supervised'] and r['expert_scheduled']
        assert q['A_state_pre_sha256']==r['A_state_pre_sha256'] and q['A_state_post_sha256']==r['A_state_post_sha256'] and q['pixel_sha256']==r['pixel_sha256']
        assert q['proposal_count']==r['proposal_count'] and len(q['trace'])==3
        close(q['teacher_mass'],[1,1],1e-12);assert min(q['teacher_joint_entropy'])>=0
        for j,s in enumerate(q['trace']):
            assert s['step']==j+1;close(s['loss'],s['teacher_KL']+s['prior_KL']);close(s['after_loss'],s['after_teacher_KL']+s['after_prior_KL'])
            close(s['step_displacement'],.01*s['gradient_norm'],2e-6);assert s['bias_gradient_norm']<1e-5
            if j:close(s['loss'],q['trace'][j-1]['after_loss'])
            assert {k:v for k,v in s.items() if k not in ['v','t']}==unscored[q['cell_key']]['trace'][j]
        close([q['trace'][-1]['v'],q['trace'][-1]['t']],[r['EMix_v'],r['EMix_t']])
    td=read(directory/'TEACHER_DIAGNOSTICS.json');assert len(td)==288
    for q in td:
        r=index[q['cell_key']];assert r['expert_scheduled'] and q['raw_rows']==r['proposal_count'] and 0<q['raw_unique_intervals']<=q['raw_rows']
        assert 0<=q['raw_mean_tIoU']<=q['raw_max_tIoU']<=1 and 0<=q['raw_fraction_tIoU_ge_05']<=1
        close([s['mass'] for s in q['offsets']],[1,1],1e-12)
        for s in q['offsets']:assert 0<=s['expected_joint_tIoU']<=1 and s['joint_entropy']<=np.log(s['legal_spans'])+1e-12
    for ds in saved:
        for sp,panels in saved[ds].items():
            rr=[r for r in rows if r['dataset']==ds and r['split']==sp]
            for name,z in panels.items():
                q=[r for r in rr if PREDICATES[name](r)];fields=list(z['metrics']);v=summary(q,fields)
                assert v['sources']==z['sources'] and v['cells']==z['cells']
                for k in fields:
                    for f in ['mean','ci95','cell_macro','source_positive','source_negative']:close(v['metrics'][k][f],z['metrics'][k][f])
                    for s in v['metrics'][k]['source_values']:close(v['metrics'][k]['source_values'][s],z['metrics'][k]['source_values'][s])
                for b,t in z['negative_tails'].items():
                    expected=dict(severe_harm=sum(r['EMix_minus_'+b+'_v']<-.05 for r in q),harm=sum(r['EMix_minus_'+b+'_v']<-1e-12 for r in q),gain=sum(r['EMix_minus_'+b+'_v']>1e-12 for r in q),
                        baseline_good_destroyed_at_03=sum(r[b+'_v']>=.3 and r['EMix_v']<.3 for r in q),baseline_bad_rescued_at_03=sum(r[b+'_v']<.3 and r['EMix_v']>=.3 for r in q))
                    assert expected==t;counter['tail_counts']+=len(t)
                assert z['loss_down_task_down']==sum(r.get('EMix_loss_after',0)<r.get('EMix_loss_before',0) and r['EMix_minus_N_v']<-1e-12 for r in q)
                for o,t in z['orders'].items():
                    v=summary([r for r in q if r['order']==o],fields);assert v['cells']==t['cells'] and v['sources']==t['sources']
                    for k in fields:close(v['metrics'][k]['mean'],t['metrics'][k]['mean']);close(v['metrics'][k]['ci95'],t['metrics'][k]['ci95'])
    return dict(status='pass',checks=dict(counter),max_absolute_error=maxerr,CPU_wall_seconds=time.monotonic()-tick,
        scope='Anonymous mixture timing/normalized mass, paired arithmetic, reset traces, tails, orders and independently resampled 10000 source bootstrap. No private inputs.',time=time.time())

def root_audit():
    import torch
    from scipy.special import logsumexp
    torch.set_num_threads(2);tick=time.monotonic();count=collections.Counter();maxerrors=collections.defaultdict(float)
    cfg=read(PUB/'CONFIG.json');lock=read(BASE/'RUNTIME_LOCK.json');bar=read(BASE/'MIX_PREDICTION_BARRIER.json');glob=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    for f,h in {**lock['code'],**lock['inputs'],**lock['post_seal_label_inputs'],**lock['post_seal_baselines']}.items():assert sha(ROOT/f)==h,f;count['immutable_hashes']+=1
    scored=read(BASE/'SCORE_COMPLETION.json');assert bar['time']<glob['time']<scored['time'] and bar['pid']!=scored['score_pid'] and bar['GT_guard_active'] and not bar['GT_read']
    assert cfg['lr']==read(R1/'SOURCE_SELECTION_BARRIER.json')['choices']=={'vidstg':.01,'hc2':.01}
    cohort=read(BASE/'COHORT.json')['cells'];assert cohort==read(R2/'COHORT.json')['cells']
    rows={r['cell_key']:r for r in read(PUB/'ROWS.json')};oldrows={r['cell_key']:r for r in read(R2PUB/'ROWS.json')};support=read(R2/'EXPERT_SUPPORT.json')
    traces={r['cell_key']:r for r in read(PUB/'MIX_TRACES_SCORED.json')};diagnostics={r['cell_key']:r for r in read(PUB/'TEACHER_DIAGNOSTICS.json')}
    def close(a,b,kind,tol=1e-9):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        d=float(np.max(np.abs(a-b))) if a.size else 0.;assert d<=tol,(kind,d,tol)
        count[kind+'_scalars']+=a.size;maxerrors[kind]=max(maxerrors[kind],d)
    def load(f):return torch.load(f,map_location='cpu',weights_only=False,mmap=True)
    def decode(z,ids):
        out=[]
        for at in [list(range(j,len(ids),2)) for j in [0,1]]:
            a=z[at];s=a[:,0].log_softmax(0);e=a[:,1].log_softmax(0);n=len(at);mask=(torch.ones(n,n)*-1e32).tril(0)
            ij=int((mask+s[:,None]+e[None,:]).flatten().argmax());i,j=divmod(ij,n);assert i<j;out.append([at[i],at[j]])
        pair=[min(x[0] for x in out),max(x[1] for x in out)]
        return dict(indices=pair,physical_interval=[ids[pair[0]],ids[pair[1]]+1],offset_indices=out)
    def check_fit(a,z,head,raw):
        ids=z['frame_ids'];x=torch.relu(torch.nn.functional.linear(z['hidden'].float(),head['0.weight'],head['0.bias']))
        close(a['states'][0]['weight'],head['1.weight'],'head_reset',0);close(a['states'][0]['bias'],head['1.bias'],'head_reset',0)
        w=head['1.weight'].clone();b=head['1.bias'].clone();zz=torch.nn.functional.linear(x,w,b);close(zz,a['logits'][0],'initial_logits',0);assert decode(zz,ids)==a['before']
        grids=[];pairs=[];priors=[];teachers=[]
        for off,at in enumerate([list(range(j,len(ids),2)) for j in [0,1]]):
            f=np.asarray([ids[i] for i in at],float);i,j=np.triu_indices(len(at),1);spacing=np.sort(np.diff(f));sig=float(spacing[(len(spacing)-1)//2])
            # Independent vectorized component-normalize-then-average.
            components=-((f[i][None,:]-raw[:,0,None])**2+(f[j][None,:]+1-raw[:,1,None])**2)/(2*sig**2)
            components-=logsumexp(components,axis=1,keepdims=True);q=logsumexp(components,axis=0)-np.log(len(raw))
            close(q,a['logq'][off],'equal_mixture',1e-9);close(float(np.exp(q).sum()),1.,'teacher_mass',1e-12);close(sig,a['sigma_frames'][off],'sigma',0)
            zv=zz[at].double().numpy();p=zv[i,0]+zv[j,1];p-=logsumexp(p);close(p,a['logp0'][off],'frozen_prior',1e-10)
            grids.append(at);pairs.append((i,j));teachers.append(q);priors.append(p)
        def objective(logits):
            dqs=[];anchors=[];gz=np.zeros((len(ids),2))
            for at,(i,j),q,p0 in zip(grids,pairs,teachers,priors):
                zv=logits[at].double().numpy();p=zv[i,0]+zv[j,1];p-=logsumexp(p)
                dqs.append(float(np.sum(np.exp(q)*(q-p))));anchors.append(float(np.sum(np.exp(p0)*(p0-p))))
                diff=(2*np.exp(p)-np.exp(q)-np.exp(p0))/2;dz=np.zeros_like(zv);np.add.at(dz[:,0],i,diff);np.add.at(dz[:,1],j,diff);gz[at]=dz
            return float(np.mean(dqs)),float(np.mean(anchors)),gz
        for k in range(3):
            q,anchor,gz=objective(zz);s=a['trace'][k];state=a['states'][k+1]
            close([q,anchor,q+anchor],[s['teacher_KL'],s['prior_KL'],s['loss']],'objectives',1e-9)
            gw=gz.T@x.double().numpy();gb=gz.sum(0);close(gw,state['weight_gradient'],'analytic_gradient',2e-5);close(gb,state['bias_gradient'],'analytic_gradient',2e-5)
            ew=a['states'][k]['weight'].clone().add_(state['weight_gradient'],alpha=-.01);eb=a['states'][k]['bias'].clone().add_(state['bias_gradient'],alpha=-.01)
            close(ew,state['weight'],'SGD_arithmetic',0);close(eb,state['bias'],'SGD_arithmetic',0)
            w.add_(torch.from_numpy(gw).float(),alpha=-.01);b.add_(torch.from_numpy(gb).float(),alpha=-.01)
            close(w,state['weight'],'analytic_execution',2e-6);close(b,state['bias'],'analytic_execution',2e-6)
            zz=torch.nn.functional.linear(x,state['weight'],state['bias']);close(zz,a['logits'][k+1],'state_logits',0)
            q,anchor,_=objective(zz);close([q,anchor,q+anchor],[s['after_teacher_KL'],s['after_prior_KL'],s['after_loss']],'post_objectives',1e-9)
            assert decode(zz,ids)==s['prediction'];count['native_decodes']+=1
        assert a['after']==a['trace'][-1]['prediction'] and not a['spatial_changed'] and not a['GT_supervised'] and a['episodic_reset'] and a['proposal_count']==len(raw)
        count['episodic_queries']+=1;count['independent_gradient_steps']+=3
        return grids,pairs,teachers
    from scripts.tastvg_correction_views_common_v1 import oldcell
    from vg_tta.tastvg_oracle_event5_v1 import official
    for ds in ['vidstg','hc2']:
        head=load(R1/ds/'HEAD.pt');cp=load(ROOT/cfg['checkpoints'][ds]['path'])['model_ema']
        for k in head:close(head[k],cp['temp_embed.layers.'+k],'checkpoint_head',0)
        del cp
        plan=read(VIEW/ds/'PLAN.json');labels={sp:read(POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        for c in [c for c in cohort if c['dataset']==ds]:
            k=cellkey(c);r=rows[k];oldr=oldrows[k];g=labels[c['split']][str(c['parent'])];row=plan['rows'][c['parent']];ids=row['frame_ids'];truth={int(j):v for j,v in g['truth'].items()}
            for name in BASELINES+['teacher_MAP','GT_time','EDeploy_map','EOracle_map']:
                for m in ['v','t']:close(r[name+'_'+m],oldr[name+'_'+m],'reused_control_fields',0)
            old=oldcell(ds,c['split'],c['condition'],c['order'],c['arrival']);assert old['pre_sha']==c['pre_sha'] and old['post_sha']==c['post_sha']
            pr=read(PRE/ds/'predictions'/f'{pre(c)}.json');A8=[ids[pr['A_indices'][0]],ids[pr['A_indices'][1]]+1];intervals={'A8':A8}
            if c['scheduled']:
                z=load(LAT/ds/'target_features'/f'{pre(c)}.pt');raw=np.asarray(support[k]['proposals']);ep=read(POOL/ds/'experts/temporal'/c['condition']/f"{c['parent']:05}.json")
                ec=load(POOL/ds/'experts'/ep['cache']);close(raw,ec['proposals'],'raw_support_unchanged',0)
                p=BASE/ds/'mix_runs'/f'{pre(c)}.pt';assert sha(p)==bar['files'][str(p.relative_to(BASE))];a=load(p)
                h=hashlib.sha256(json.dumps(support[k]['proposals'],sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h==a['raw_proposals_sha256']==traces[k]['raw_proposals_sha256']
                grids,pairs,teachers=check_fit(a,z,head,raw)
                assert a['A_state_pre_sha256']==c['pre_sha'] and a['A_state_post_sha256']==c['post_sha'] and a['pixel_sha256']==c['pixel_sha256']
                rf=read(POOL/ds/'capture'/c['condition']/f"{c['parent']:05}.json");capture=load(POOL/ds/rf['cache']);r1a=load(R1/ds/'target_runs'/f'{pre(c)}.pt')
                assert a['before']==r1a['before'] and a['before']['indices']==capture['prediction']['indices']==pr['native_indices'];close(a['logits'][0],r1a['logits'][0],'R1_initial_logits',0)
                teacher=[];dg=diagnostics[k];ov=np.maximum(0,np.minimum(raw[:,1],g['span'][1])-np.maximum(raw[:,0],g['span'][0]));ts=ov/(raw[:,1]-raw[:,0]+g['span'][1]-g['span'][0]-ov)
                close([ts.mean(),ts.max(),np.mean(ts>=.5)],[dg['raw_mean_tIoU'],dg['raw_max_tIoU'],dg['raw_fraction_tIoU_ge_05']],'teacher_postseal_diagnostics')
                assert dg['raw_rows']==len(raw) and dg['raw_unique_intervals']==len(np.unique(raw,axis=0))
                for off,(at,(i,j),q) in enumerate(zip(grids,pairs,teachers)):
                    ix=int(np.argmax(q));teacher.append([at[i[ix]],at[j[ix]]]);close(a['logp0'][off],r1a['logp0'][off],'R1_prior',0)
                    close(a['logits'][0][at],capture['prediction']['logits'][off].reshape(-1,2),'original_CUDA_CPU_logits',1e-4)
                    f=np.asarray(ids)[at];ss=f[i];ee=f[j]+1;gt=g['span'];ov=np.maximum(0,np.minimum(ee,gt[1])-np.maximum(ss,gt[0]));tt=ov/(ee-ss+gt[1]-gt[0]-ov);mass=np.exp(q)
                    close([float(-(mass*q).sum()),mass.sum(),mass@tt],[dg['offsets'][off]['joint_entropy'],dg['offsets'][off]['mass'],dg['offsets'][off]['expected_joint_tIoU']],'mixture_postseal_diagnostics')
                    close(traces[k]['teacher_joint_entropy'][off],float(-(mass*q).sum()),'public_teacher_entropy');assert dg['offsets'][off]['legal_spans']==len(q)
                assert a['teacher_MAP']['indices']==[min(t[0] for t in teacher),max(t[1] for t in teacher)];count['teacher_MAP_checks']+=1
                intervals.update(EMix=a['after']['physical_interval'],EMix_map=a['teacher_MAP']['physical_interval'],N=a['before']['physical_interval'],R1=r1a['after']['physical_interval'])
                for arm,folder in [('EDeploy','deploy_runs'),('EOracle','oracle_runs')]:
                    olda=load(R2/ds/folder/f'{pre(c)}.pt');intervals[arm]=olda['after']['physical_interval'];intervals[arm+'_map']=olda['teacher_MAP']['physical_interval']
                for j,s in enumerate(traces[k]['trace']):
                    internal=a['trace'][j]
                    for name,value in internal.items():
                        if name=='prediction':continue
                        if isinstance(value,(int,float)):close(value,s[name],'public_trace_scalars',0)
                        else:assert value==s[name]
                    m=official(old['slow']['boxes'],row,truth,g['span'],internal['prediction']['physical_interval'],ds);close([m['v'],m['t']],[s['v'],s['t']],'step_dense_metrics')
            else:intervals.update({name:A8 for name in BASELINES+['EMix','EMix_map','EDeploy_map','EOracle_map']})
            for name,iv in intervals.items():
                m=official(old['slow']['boxes'],row,truth,g['span'],iv,ds);close([m['v'],m['t']],[r[name+'_v'],r[name+'_t']],'official_dense_metrics')
            count['A_immutable_rows']+=1
    assert count['episodic_queries']==288 and count['independent_gradient_steps']==864 and count['A_immutable_rows']==1152
    out=dict(status='pass',checks=dict(count),maximum_errors=dict(maxerrors),CPU_wall_seconds=time.monotonic()-tick,
        gradient_validation='Independent NumPy component-normalized equal joint mixture, joint-marginal gradient, all SGD states/logits/native decodes and joint teacher MAP.',
        scope='Full 288 new fits/1152 readouts, unchanged R2 raw support and controls, official dense; no target retuning, persistence or CUDA.',CUDA_initialized=torch.cuda.is_initialized(),time=time.time())
    write(PUB/'ROOT_AUDIT.json',out);write(BASE/'ROOT_AUDIT.json',out);return out

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='root':z=root_audit()
    else:
        d=Path(sys.argv[1]) if len(sys.argv)>1 else PUB;z=public_audit(d);p=d/'PUBLIC_AUDIT.json'
        if p.exists():assert z['checks']==read(p)['checks']
        else:write(p,z)
    print(json.dumps(z,indent=2))
