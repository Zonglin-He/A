"""CPU-only post-seal dense scoring and two-node-held-out utility diagnosis."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_three_scope_common_v1 import *
from scripts.decota_public_result_io_v1 import read as public_read
from vg_tta.decota_three_scope_v1 import boundary,cosine,pair_features,ridge_predict
from scripts.decota_ln_spectrum_math_v1 import weighted_corr,plain
from scripts.score_decota_optimizer_posterior_v1 import stats
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
import numpy as np,torch

def cpu_lock(stage,extra=None):
    verify();pins={f:sha(ROOT/f) for f in ['scripts/score_decota_three_scope_v1.py','vg_tta/decota_three_scope_v1.py',
        'scripts/score_decota_identity_commitment_v1.py','scripts/score_decota_optimizer_posterior_v1.py','vg_tta/tastvg_oracle_event5_v1.py','scripts/decota_ln_spectrum_math_v1.py']}
    labels={str((c1.POOL/ds/f'GT_LABELS_{split}.json').relative_to(ROOT)):sha(c1.POOL/ds/f'GT_LABELS_{split}.json') for ds in DATASETS for split in ['search','confirm']}
    write(BASE/(stage+'_CPU_LOCK.json'),dict(pins=pins,labels=labels,extra=extra or {},time=time.time()))

def verify_cpu(stage):
    verify();z=read(BASE/(stage+'_CPU_LOCK.json'))
    for f,h in {**z['pins'],**z['labels'],**z['extra']}.items():assert sha(ROOT/f)==h,f

def seal_check(name):
    z=read(BASE/name);assert z['status']=='sealed' and not z['GT_read']
    for f,h in z.get('files',{}).items():assert sha(BASE/f)==h,f
    return z

def boundary_prepare():
    verify();bb=read(old.BASE/'ONLINE_GLOBAL_BARRIER.json');assert bb['status']=='sealed' and bb['arrivals']==13824
    rows=[];cache={};tick=time.time()
    for c in read(BASE/'COHORT.json')['cells']:
        ds,cond,parent=c['dataset'],c['condition'],c['parent'];p=read(BASE/ds/'PLAN.json');row=p['rows'][parent];key=(ds,cond,parent)
        oldf=ROOT/c['references']['online100'];x=old.checked(oldf)
        if key not in cache:
            data,rc=c1.cache(ds,row['pool_parent'],cond)
            assert torch.equal(data['prediction']['boxes'],x['native']['boxes'])
            assert all(torch.equal(a,b) for a,b in zip(data['prediction']['logits'],x['native']['logits']))
            q=boundary([a.numpy() for a in x['native']['logits']],data['records'],row['frame_ids'])
            # Independent scalar MAP on all legal original endpoint pairs.
            a,b=q['start'],q['end'];best=max(((np.log(a[i])+np.log(b[j]),-i,-j) for i in range(len(a)) for j in range(i+1,len(a))),key=lambda z:z)
            assert q['indices']==[-best[1],-best[2]]
            cache[key]=q
        rows.append({k:c[k] for k in ['dataset','split','condition','order','arrival','parent']}|dict(q=cache[key],native_interval=x['interval'],reference=str(oldf.relative_to(ROOT)),reference_sha256=sha(oldf),GT_read=False))
    f=BASE/'boundary/PREDICTIONS.json';write(f,rows)
    write(BASE/'BOUNDARY_PREDICTION_BARRIER.json',dict(status='sealed',rows=len(rows),unique_inputs=len(cache),files={str(f.relative_to(BASE)):sha(f)},GT_read=False,time=time.time(),seconds=time.time()-tick))
    cpu_lock('boundary');print('BOUNDARY_LABEL_FREE_SEALED',len(rows),len(cache),flush=True)

def groups(rows,tag,fields):
    out={}
    for ds in DATASETS:
        out[ds]={}
        for split in ['search','confirm']:
            out[ds][split]={}
            for arm in sorted({q[tag] for q in rows}):
                rr=[q for q in rows if q['dataset']==ds and q['split']==split and q[tag]==arm]
                gg={'corruption':[q for q in rr if q['condition']!='clean'],'clean':[q for q in rr if q['condition']=='clean']}
                gg.update({o:[q for q in rr if q['condition']!='clean' and q['order']==o] for o in ['order1','order2']})
                gg.update({co:[q for q in rr if q['condition']==co] for co in sorted({q['condition'] for q in rr})})
                out[ds][split][arm]={k:stats(v,fields) for k,v in gg.items()}
                for k,q in gg.items():
                    ref='vs_uniform_v' if tag=='stream' else 'delta_v'
                    out[ds][split][arm][k]['tails']={f'vs_reference_harm_gt{n}pp':sum(r[ref]<-n/100 for r in q) for n in [5,20]}
                    if tag=='stream':out[ds][split][arm][k]['tails'].update({f'vs_frozen_harm_gt{n}pp':sum(r['vs_frozen_v']<-n/100 for r in q) for n in [5,20]})
    return out

def source_corr(rows,x,y):
    ids=sorted({r['source_id'] for r in rows});idx={s:i for i,s in enumerate(ids)};ix=np.array([idx[r['source_id']] for r in rows]);n=len(ids)
    xx=np.array([r[x] for r in rows]);yy=np.array([r[y] for r in rows]);count=collections.Counter(ix);w=np.array([1/count[i] for i in ix]);point=weighted_corr(xx,yy,w);boot=[];rng=np.random.default_rng(20261005)
    for _ in range(10000):
        c=rng.multinomial(n,np.full(n,1/n));z=weighted_corr(xx,yy,w*c[ix])
        if z is not None:boot.append(z)
    return dict(cells=len(rows),sources=n,mean=point,ci95=np.quantile(boot,[.025,.975]).tolist() if boot else None,valid_draws=len(boot),seed=20261005)

def boundary_score():
    verify_cpu('boundary');bar=seal_check('BOUNDARY_PREDICTION_BARRIER.json');assert bar['rows']==1152
    write(BASE/'boundary_GT_EXPOSURE.json',dict(barrier_sha256=sha(BASE/'BOUNDARY_PREDICTION_BARRIER.json'),time=time.time(),GT_online=False))
    rows=[];checks=0;labels={};plans={};tick=time.time()
    for ds in DATASETS:
        plans[ds]=read(BASE/ds/'PLAN.json')
        for split in ['search','confirm']:labels[ds,split]=read(c1.POOL/ds/f'GT_LABELS_{split}.json')
    for c in read(BASE/'boundary/PREDICTIONS.json'):
        ds,split,parent=c['dataset'],c['split'],c['parent'];row=plans[ds]['rows'][parent];x=old.checked(ROOT/c['reference']);assert sha(ROOT/c['reference'])==c['reference_sha256']
        lab=labels[ds,split][str(parent)];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span'];boxes=x['after'].numpy();dt=DenseTube(boxes,row,truth,span,clip=ds=='hc2')
        native=dt.score(c['native_interval']);cons=dt.score(c['q']['interval'])
        for iv,m in [(c['native_interval'],native),(c['q']['interval'],cons)]:
            off=official(boxes,row,truth,span,iv,ds);assert max(abs(off[k]-m[k]) for k in off)<2e-12;checks+=3
        rows.append({k:c[k] for k in ['dataset','split','condition','order','arrival']}|dict(source_id=parent,arm='consensus',v=cons['v'],t=cons['t'],s=cons['s'],native_v=native['v'],native_t=native['t'],
            delta_v=cons['v']-native['v'],delta_t=cons['t']-native['t'],js=c['q']['js_mean'],native_error=1-native['t'],native_interval=c['native_interval'],consensus_interval=c['q']['interval'],prediction_reference_sha256=c['reference_sha256']))
    sums=groups(rows,'arm',['v','t','s','native_v','native_t','delta_v','delta_t']);go={};js={}
    for ds in DATASETS:
        a=sums[ds]['confirm']['consensus']['corruption']['metrics'];b=sums[ds]['search']['consensus']['corruption']['metrics']
        go[ds]=all(a[k]['ci95'][0]>0 and b[k]['mean']>0 for k in ['delta_t','delta_v'])
        js[ds]={sp:source_corr([r for r in rows if r['dataset']==ds and r['split']==sp and r['condition']!='clean'],'js','native_error') for sp in ['search','confirm']}
    decision=dict(temporal_gradient_qualified=all(go.values()),dataset_gates=go,parameter_update_executed=False,rule_prelocked=True,GT_online=False,production_promoted=False)
    write(PUB/'boundary/ROWS.json',rows);write(PUB/'boundary/SUMMARY.json',sums);write(PUB/'boundary/JS_DIAGNOSTIC.json',js);write(PUB/'boundary/DECISION.json',decision)
    write(PUB/'boundary/ROOT_AUDIT.json',dict(status='pass',rows=1152,checks=checks,official_dense=True,scalar_MAP_independently_verified=True,GT_after_seal=True,new_model_calls=0,time=time.time(),seconds=time.time()-tick))
    write(BASE/'BOUNDARY_DECISION.json',decision);archive('双offset无梯度P0已实际完成1152封存/官方dense评分；Temporal514资格化='+str(decision['temporal_gradient_qualified']))
    print('BOUNDARY_DECISION',decision,flush=True)

def observation_metrics(ex,row,truth,span):
    from scripts.score_decota_actuation_scope_v1 import iou
    vals=[];topvals=[];admitted=valid=scored=event=0;slots=len(ex['positions4']);w,h=row['input']['width'],row['input']['height'];an=[]
    for (_,pos),o in sorted(ex['observations'].items()):
        p=o['probe'];b=np.asarray(p['boxes'],float).reshape(-1,4);s=np.asarray(p['target_scores'],float);mask=(b[:,2:]>0).all(1);b,s=b[mask],s[mask]
        fid=row['frame_ids'][pos];event+=int(span[0]<=fid<span[1]);valid+=int(len(s)>0);admitted+=int(p['accepted'])
        rec=dict(position=pos,frame_id=fid,valid=bool(len(s)),admitted=bool(p['accepted']),event_frame=bool(span[0]<=fid<span[1]),scored_GT=fid in truth,GT_IoU=None)
        if len(s) and fid in truth:
            g=np.asarray(truth[fid],float);g=np.r_[(g[:2]+g[2:])/2/[w,h],(g[2:]-g[:2])/[w,h]];quality=float(iou(b[int(s.argmax())],g));topvals.append(quality);rec['GT_IoU']=quality;scored+=1
            if p['accepted']:vals.append(quality)
        an.append(rec)
    return dict(slots=slots,actual=len(ex['observations']),valid=valid,admitted=admitted,event=event,scored_valid=scored,
        admitted_GT_scored=len(vals),admitted_GT_IoU_sum=sum(vals),top1_GT_IoU_sum=sum(topvals),
        mean_admitted_GT_IoU=float(np.mean(vals)) if vals else None,best=float(max(vals)) if vals else None,worst=float(min(vals)) if vals else None,
        valid_rate=valid/max(slots,1),admission_rate=admitted/max(slots,1),event_rate=event/max(slots,1),admitted_event_GT_mass=sum(vals)/max(slots,1),
        observations=an,identity_accuracy_not_measured=True)

def scope_score():
    from scripts.score_decota_identity_commitment_v1 import audit_fit
    from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import check_state
    from vg_tta.c1_enabling_tricks_v1 import QUERY
    from vg_tta.decota_actuation_scope_v1 import commit_state
    from methods.decota_final_simplified_v1.tensors import state_hash
    if not (BASE/'scope_CPU_LOCK.json').exists():cpu_lock('scope')
    verify_cpu('scope');bar=seal_check('GLOBAL_PREDICTION_BARRIER.json');assert bar['arrivals']==2304
    write(BASE/'scope_GT_EXPOSURE.json',dict(barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time(),GT_online=False))
    torch.set_num_threads(4);rows=[];evidence=[];diag=[];checks=0;tick=time.time();episodes={};oldrows={}
    for r in public_read(old.PUB/'online/ROWS.json'):
        if r['stream'] in ['episodic','online100']:oldrows[(r['dataset'],r['stream'],r['split'],r['condition'],r['order'],r['arrival'])]=r
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');labels={sp:read(c1.POOL/ds/f'GT_LABELS_{sp}.json') for sp in ['search','confirm']}
        ev_done=set()
        for stream in ['episodic','online100']:
            for split,sp in p['splits'].items():
                for cond in p['conditions']:
                    for order,seq in sp['orders'].items():
                        prev=None;prevsha=None
                        for at,parent in enumerate(seq):
                            row=p['rows'][parent];f=BASE/'online'/ds/stream/split/cond/order/f'{at:05}.pt';x=checked(f);assert x['parent']==parent and x['previous_payload_sha256']==prevsha
                            init={n:torch.zeros_like(v) if n==QUERY else v for n,v in (prev if prev is not None else x['source_state']).items()};checks+=check_state(init,x['initial'])
                            assert x['query_reset'] and x['Adam_reset'] and torch.count_nonzero(x['initial'][QUERY])==0
                            ef=ROOT/x['evidence_path'];ex=checked(ef)['expert'];assert sha(ef)==x['evidence_sha256']
                            checks+=audit_fit(x['fit'],x['initial'],x['before'],ex,row,'top1');checks+=check_state(commit_state(x['initial'],x['fit']['state']),x['committed'])
                            assert torch.equal(x['after'],x['fit']['final'])
                            if stream=='online100':prev=x['committed'];prevsha=sha(f)
                            if 'reused_episodic' in x:assert sha(ROOT/x['reused_episodic'])==x['reuse_sha256'] and x['actual_backward_calls']==0
                            lab=labels[split][str(parent)];truth={int(k):v for k,v in lab['truth'].items()};span=lab['span'];iv=x['interval']
                            dt=lambda b:DenseTube(b.numpy(),row,truth,span,clip=ds=='hc2');fr=dt(x['native']['boxes']).score(iv);be=dt(x['before']).score(iv);m=dt(x['after']).score(iv)
                            off=official(x['after'].numpy(),row,truth,span,iv,ds);assert max(abs(off[k]-m[k]) for k in off)<2e-12;checks+=3
                            o=oldrows[(ds,stream,split,cond,order,at)];assert abs(fr['v']-o['frozen_v'])<2e-12
                            key=(ds,split,cond,order,at)
                            if stream=='episodic':episodes[key]=m['v']
                            rr=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,stream=stream,**m,
                                frozen_v=fr['v'],before_v=be['v'],uniform_v=o['v'],uniform_s=o['s'],uniform_before_v=o['before_v'],
                                vs_frozen_v=m['v']-fr['v'],vs_uniform_v=m['v']-o['v'],vs_uniform_s=m['s']-o['s'],current_v=m['v']-be['v'],before_vs_frozen_v=be['v']-fr['v'],
                                net_memory_v=m['v']-episodes[key],uniform_net_memory_v=o['vs_episodic_v'],gross_gain=max(m['v']-fr['v'],0),gross_loss=max(fr['v']-m['v'],0),
                                payload_sha256=sha(f),prestate_sha256=state_hash(x['initial']),committed_state_sha256=state_hash(x['committed']))
                            rows.append(rr);diag.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,stream=stream,
                                selected_step=x['fit']['selected_step'],losses=[h['loss'] for h in x['fit']['path']],step_v=[dt(h['boxes']).score(iv)['v'] for h in x['fit']['path']],
                                seconds=x['seconds'],actual_backward_calls=x['actual_backward_calls'],gradient_norms=[float(h['update']['gradient'].norm()) for h in x['fit']['path'] if 'update' in h]))
                            evkey=(parent,cond)
                            if evkey not in ev_done:
                                oldef=c1.BASE/ds/'evidence'/cond/f'{parent:05}.pt';oldex=c1.checked(oldef)['expert']
                                aa=observation_metrics(oldex,row,truth,span);bb=observation_metrics(ex,row,truth,span)
                                evidence.append(dict(dataset=ds,split=split,condition=cond,source_id=parent,uniform=aa,tts=bb,
                                    delta_valid_rate=bb['valid_rate']-aa['valid_rate'],delta_admission_rate=bb['admission_rate']-aa['admission_rate'],delta_event_rate=bb['event_rate']-aa['event_rate'],
                                    delta_admitted_event_GT_mass=bb['admitted_event_GT_mass']-aa['admitted_event_GT_mass']))
                                ev_done.add(evkey)
                        print('THREE_SCOPE_ROOT',ds,stream,split,cond,order,len(rows),checks,flush=True)
    assert len(rows)==2304 and len(evidence)==576
    fields=['v','s','t','frozen_v','uniform_v','vs_frozen_v','vs_uniform_v','vs_uniform_s','current_v','before_vs_frozen_v','net_memory_v','uniform_net_memory_v','gross_gain','gross_loss']
    sums=groups(rows,'stream',fields);es={}
    for ds in DATASETS:
        es[ds]={sp:{co:stats([e for e in evidence if e['dataset']==ds and e['split']==sp and ((e['condition']=='clean')==(co=='clean'))],['delta_valid_rate','delta_admission_rate','delta_event_rate','delta_admitted_event_GT_mass']) for co in ['clean','corruption']} for sp in ['search','confirm']}
    go={ds:sums[ds]['confirm']['online100']['corruption']['metrics']['vs_uniform_v']['ci95'][0]>0 for ds in DATASETS}
    decision=dict(scope_benefit_established=all(go.values()),dataset_gates=go,production_promoted=False,old_Uniform_baseline_retained=not all(go.values()),method_parameters_changed=False,independent_online_executed=True)
    write(PUB/'scope/ROWS.json',rows);write(PUB/'scope/DIAGNOSTICS.json',diag);write(PUB/'scope/SUMMARY.json',sums);write(PUB/'scope/EVIDENCE_ROWS.json',evidence);write(PUB/'scope/EVIDENCE_SUMMARY.json',es);write(PUB/'scope/DECISION.json',decision)
    cost={ds:{stage:read(BASE/ds/(stage+'_BARRIER.json')) for stage in ['FEATURE','EVIDENCE','ONLINE']} for ds in DATASETS}
    anon={ds:{st:{k:v for k,v in z.items() if k not in ['files','time']} for st,z in dd.items()} for ds,dd in cost.items()}
    write(PUB/'scope/COST.json',anon);write(PUB/'scope/ROOT_AUDIT.json',dict(status='pass',rows=len(rows),checks=checks,source_state_chain=True,independent_numpy_Top1_objective_Adam=True,official_dense=True,GT_after_global_seal=True,decoder_Jacobian_not_independently_replayed=True,time=time.time(),seconds=time.time()-tick))
    write(BASE/'SCOPE_DECISION.json',decision);archive('TTS选帧全部2304独立episodic/online输出封存及根算术/dense评分完成；双集确认增量资格化='+str(decision['scope_benefit_established']))

def memory_prepare():
    verify();files={};features={};metas=[]
    for ds in DATASETS:
        z=read(BASE/ds/'FEATURE_BARRIER.json');assert z['status']=='sealed'
        p=read(BASE/ds/'PLAN.json');splitmap={parent:split for split,sp in p['splits'].items() for parent in sp['orders']['order1']}
        for row in p['rows']:
            for cond in p['conditions']:
                f=BASE/'features'/ds/cond/f"{row['ordinal']:05}.pt";a=checked(f);assert a['recipient_delta_accessed'] is False
                k=f"{ds}:{cond}:{row['ordinal']}";features[k]=a['keys'];metas.append(dict(dataset=ds,split=splitmap[row['ordinal']],condition=cond,source_id=row['ordinal'],feature_payload_sha256=sha(f)))
                files[str(f.relative_to(BASE))]=sha(f)
    # Only label-free membership is read until keys/pairs are fixed.
    f=ROOT/'results/decota_ln_spectrum/2026-10-05/PAIR_SELECTIONS.json'
    assert f.exists();pairs=read(f);assert len(pairs)==939
    for r in pairs:
        for s in [r['donor'],r['recipient']]:assert f"{r['dataset']}:{r['condition']}:{s}" in features
    commit(BASE/'memory/KEYS.pt',dict(features=features,metadata=metas,GT_read=False))
    write(BASE/'memory/PAIR_SELECTIONS.json',pairs);files['memory/KEYS.pt']=sha(BASE/'memory/KEYS.pt');files['memory/PAIR_SELECTIONS.json']=sha(BASE/'memory/PAIR_SELECTIONS.json')
    write(BASE/'MEMORY_KEY_BARRIER.json',dict(status='sealed',files=files,pairs=939,keys=len(features),GT_read=False,time=time.time(),recipient_delta_accessed=False))
    labels=ROOT/'results/decota_ln_spectrum/2026-10-05/PAIR_ROWS.json';cpu_lock('memory',{str(labels.relative_to(ROOT)):sha(labels)})

def node_stats(rows,score):
    from sklearn.metrics import roc_auc_score
    ids=sorted({r[k] for r in rows for k in ['donor','recipient']});ix={s:i for i,s in enumerate(ids)};n=len(ids)
    donor=np.array([ix[r['donor']] for r in rows]);rec=np.array([ix[r['recipient']] for r in rows]);x=np.array([r[score] for r in rows]);y=np.array([r['utility_v'] for r in rows]);count=collections.Counter(rec);base=np.array([1/count[i] for i in rec]);active=y!=0
    def calc(w):
        corr=weighted_corr(x,y,w);wa=w[active];ya=y[active]>0;xa=x[active]
        auc=float(roc_auc_score(ya,xa,sample_weight=wa)) if wa[ya].sum()>0 and wa[~ya].sum()>0 else None
        return corr,auc
    point=calc(base);boots=[[],[]];rng=np.random.default_rng(20261005)
    for _ in range(10000):
        c=rng.multinomial(n,np.full(n,1/n));z=calc(base*c[donor]*c[rec])
        for j,a in enumerate(z):
            if a is not None and np.isfinite(a):boots[j].append(a)
    return dict(pairs=len(rows),source_nodes=n,recipients=len(count),donors=len(set(donor)),zero_utility_pairs=int((y==0).sum()),
        correlation=dict(mean=point[0],ci95=np.quantile(boots[0],[.025,.975]).tolist() if boots[0] else None,valid_draws=len(boots[0])),
        AUC=dict(mean=point[1],ci95=np.quantile(boots[1],[.025,.975]).tolist() if boots[1] else None,valid_draws=len(boots[1])),
        bootstrap='common source-node weights in both roles; recipient-source equal',draws=10000,seed=20261005)

def memory_score():
    verify_cpu('memory');bar=seal_check('MEMORY_KEY_BARRIER.json');assert bar['pairs']==939
    write(BASE/'memory_GT_EXPOSURE.json',dict(barrier_sha256=sha(BASE/'MEMORY_KEY_BARRIER.json'),time=time.time(),only_existing_anonymous_GT_utility_labels=True))
    keys=checked(BASE/'memory/KEYS.pt')['features'];sel=read(BASE/'memory/PAIR_SELECTIONS.json');labels=public_read(ROOT/'results/decota_ln_spectrum/2026-10-05/PAIR_ROWS.json')
    ident=lambda q:(q['dataset'],q['split'],q['condition'],q['donor'],q['recipient'])
    lookup={ident(q):q for q in labels};assert len(lookup)==len(sel)==939
    rows=[];predictors=['query_cos','spatial_cos','ridge','constant'];tick=time.time();fit_checks=0
    for ds in DATASETS:
        pp=[p for p in sel if p['dataset']==ds];xx=[];yy=[]
        for p in pp:
            d=keys[f"{ds}:{p['condition']}:{p['donor']}"];r=keys[f"{ds}:{p['condition']}:{p['recipient']}"]
            xx.append(pair_features(d,r));yy.append(lookup[ident(p)]['utility_v'])
        x=np.array(xx);y=np.array(yy);training=[i for i,p in enumerate(pp) if p['split']=='search'];pred=np.zeros(len(pp));constant=np.zeros(len(pp));training_receipts=[]
        for i,p in enumerate(pp):
            # Exclude BOTH nodes in ALL roles/conditions for honest development.
            banned={p['donor'],p['recipient']}
            train=[j for j in training if banned.isdisjoint({pp[j]['donor'],pp[j]['recipient']})]
            assert train and all(banned.isdisjoint({pp[j]['donor'],pp[j]['recipient']}) for j in train)
            pred[i]=ridge_predict(x[train],y[train],x[i:i+1])[0];constant[i]=y[train].mean();fit_checks+=len(train)
            training_receipts.append(dict(pair=i,training_pairs=len(train),training_source_nodes=sorted({pp[j][k] for j in train for k in ['donor','recipient']}),excluded_test_nodes=sorted(banned)))
            d=keys[f"{ds}:{p['condition']}:{p['donor']}"];r=keys[f"{ds}:{p['condition']}:{p['recipient']}"]
            rows.append(dict(dataset=ds,split=p['split'],condition=p['condition'],donor=p['donor'],recipient=p['recipient'],utility_v=float(y[i]),query_cos=cosine(d['query'],r['query']),spatial_cos=cosine(d['spatial'],r['spatial']),ridge=float(pred[i]),constant=float(constant[i]),training_pairs=len(train),double_role_source_holdout=True))
        write(BASE/f'memory/{ds}_FITTING_RECEIPTS.json',training_receipts)
        print('MEMORY_DOUBLE_SOURCE_HELDOUT_PREDICTED',ds,len(pp),round(time.time()-tick,1),flush=True)
    sums={};gate={a:{} for a in predictors[:-1]}
    for ds in DATASETS:
        sums[ds]={}
        for sp in ['search','confirm']:
            sums[ds][sp]={}
            for co in ['corruption','clean']:
                rr=[r for r in rows if r['dataset']==ds and r['split']==sp and ((r['condition']=='clean')==(co=='clean'))]
                sums[ds][sp][co]={a:node_stats(rr,a) for a in predictors}
                print('MEMORY_SOURCE_BOOTSTRAP',ds,sp,co,flush=True)
        for a in gate:
            z=sums[ds]['confirm']['corruption'][a];cc=z['correlation'];auc=z['AUC']
            gate[a][ds]=bool((cc['ci95'] is not None and cc['ci95'][0]>0) or (auc['mean'] is not None and auc['mean']>.65 and auc['ci95'][0]>.5))
    passing=[a for a,g in gate.items() if all(g.values())]
    decision=dict(memory_qualified=bool(passing),passing_signals=passing,gates=gate,retrieval_executed=False,ridge_alpha=1.,GT_derived_diagnostic_training_disclosed=True,recipient_delta_used=False,confirmation_not_used_to_fit=True,production_promoted=False)
    write(PUB/'memory/ROWS.json',rows);write(PUB/'memory/SUMMARY.json',sums);write(PUB/'memory/DECISION.json',decision)
    write(PUB/'memory/ROOT_AUDIT.json',dict(status='pass',pairs=939,keys=bar['keys'],fitting_node_exclusion_checks=fit_checks,features_sealed_before_label_join=True,recipient_delta_not_used=True,double_role_source_holdout=True,time=time.time(),seconds=time.time()-tick))
    write(BASE/'MEMORY_DECISION.json',decision);archive('到达前key对939精确1/16单write效用的双角色source-heldout诊断完成；conditional retrieval资格化='+str(decision['memory_qualified']))
    print('MEMORY_DECISION',decision,flush=True)

if __name__=='__main__':
    a=sys.argv[1];globals()[a]()
