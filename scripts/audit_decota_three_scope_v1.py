"""Independent input/readout/fitting audit and portable anonymous aggregation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_three_scope_common_v1 import *
from scripts.decota_public_result_io_v1 import read as public_read
import numpy as np

def independent_node_stats(rows,score):
    """Independent weighted Mann-Whitney AUC; no sklearn/producer statistic."""
    ids=sorted({r[k] for r in rows for k in ['donor','recipient']});ix={s:i for i,s in enumerate(ids)};n=len(ids)
    donor=np.array([ix[r['donor']] for r in rows]);rec=np.array([ix[r['recipient']] for r in rows])
    x=np.array([r[score] for r in rows]);y=np.array([r['utility_v'] for r in rows]);count=collections.Counter(rec)
    base=np.array([1/count[i] for i in rec]);active=y!=0;xa=x[active];positive=(y[active]>0).astype(float)
    order=np.argsort(xa,kind='stable');_,starts=np.unique(xa[order],return_index=True)
    counts=np.random.default_rng(20261005).multinomial(n,np.full(n,1/n),size=10000)
    allcorr=[];allauc=[]
    for begin in range(0,10001,500):
        end=min(begin+500,10001)
        if begin==0:w=np.vstack([base,base*counts[:end-1,donor]*counts[:end-1,rec]])
        else:w=base*counts[begin-1:end-1,donor]*counts[begin-1:end-1,rec]
        total=w.sum(1);mx=np.divide(w@x,total,out=np.zeros_like(total),where=total>0);my=np.divide(w@y,total,out=np.zeros_like(total),where=total>0)
        dx=x-mx[:,None];dy=y-my[:,None];vx=(w*dx*dx).sum(1);vy=(w*dy*dy).sum(1)
        corr=np.divide((w*dx*dy).sum(1),np.sqrt(vx*vy),out=np.full_like(total,np.nan),where=(total>0)&(vx>1e-25)&(vy>1e-25))
        allcorr.extend(np.clip(corr,-1,1).tolist())
        wa=w[:,active];wp=wa*positive;wn=wa*(1-positive)
        pg=np.add.reduceat(wp[:,order],starts,axis=1);ng=np.add.reduceat(wn[:,order],starts,axis=1)
        below=np.cumsum(ng,axis=1)-ng;den=wp.sum(1)*wn.sum(1)
        auc=np.divide((pg*(below+.5*ng)).sum(1),den,out=np.full_like(total,np.nan),where=den>0);allauc.extend(auc.tolist())
    def summary(values):
        point=values[0];boot=np.asarray(values[1:]);boot=boot[np.isfinite(boot)]
        return dict(mean=float(point) if np.isfinite(point) else None,ci95=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,valid_draws=len(boot))
    return dict(pairs=len(rows),source_nodes=n,recipients=len(count),donors=len(set(donor)),zero_utility_pairs=int((y==0).sum()),
        correlation=summary(allcorr),AUC=summary(allauc),bootstrap='common source-node weights in both roles; recipient-source equal',draws=10000,seed=20261005)

def numerically_equal(actual,expected):
    if isinstance(expected,dict):return set(actual)==set(expected) and all(numerically_equal(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return isinstance(actual,list) and len(actual)==len(expected) and all(numerically_equal(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,float):return isinstance(actual,(float,int)) and abs(actual-expected)<2e-11
    return actual==expected

def public(folder):
    from scripts.score_decota_three_scope_v1 import groups
    from scripts.score_decota_optimizer_posterior_v1 import stats
    folder=Path(folder);checks=0
    rows=public_read(folder/'boundary/ROWS.json');assert len(rows)==1152
    assert groups(rows,'arm',['v','t','s','native_v','native_t','delta_v','delta_t'])==public_read(folder/'boundary/SUMMARY.json')
    for r in rows:
        assert abs(r['delta_v']-(r['v']-r['native_v']))<1e-12 and abs(r['delta_t']-(r['t']-r['native_t']))<1e-12;checks+=2
    z=public_read(folder/'boundary/DECISION.json');s=public_read(folder/'boundary/SUMMARY.json')
    gates={d:all(s[d]['confirm']['consensus']['corruption']['metrics'][k]['ci95'][0]>0 and s[d]['search']['consensus']['corruption']['metrics'][k]['mean']>0 for k in ['delta_v','delta_t']) for d in DATASETS}
    assert z['dataset_gates']==gates and z['temporal_gradient_qualified']==all(gates.values())
    rr=public_read(folder/'scope/ROWS.json');assert len(rr)==2304
    fields=['v','s','t','frozen_v','uniform_v','vs_frozen_v','vs_uniform_v','vs_uniform_s','current_v','before_vs_frozen_v','net_memory_v','uniform_net_memory_v','gross_gain','gross_loss']
    ss=groups(rr,'stream',fields);assert ss==public_read(folder/'scope/SUMMARY.json')
    scope_gates={d:ss[d]['confirm']['online100']['corruption']['metrics']['vs_uniform_v']['ci95'][0]>0 for d in DATASETS}
    scope_decision=public_read(folder/'scope/DECISION.json')
    assert scope_decision['dataset_gates']==scope_gates and scope_decision['scope_benefit_established']==all(scope_gates.values())
    for r in rr:
        assert abs(r['vs_frozen_v']-r['v']+r['frozen_v'])<1e-12 and abs(r['vs_uniform_v']-r['v']+r['uniform_v'])<1e-12
        assert abs(r['current_v']-r['v']+r['before_v'])<1e-12;assert r['gross_gain']*r['gross_loss']==0;checks+=4
    e=public_read(folder/'scope/EVIDENCE_ROWS.json');assert len(e)==576
    es={ds:{sp:{co:stats([q for q in e if q['dataset']==ds and q['split']==sp and ((q['condition']=='clean')==(co=='clean'))],['delta_valid_rate','delta_admission_rate','delta_event_rate','delta_admitted_event_GT_mass']) for co in ['clean','corruption']} for sp in ['search','confirm']} for ds in DATASETS}
    assert es==public_read(folder/'scope/EVIDENCE_SUMMARY.json')
    for q in e:
        for k in ['uniform','tts']:
            a=q[k];assert a['actual']<=a['slots']<=4 and a['admitted']<=a['valid']<=a['actual'];checks+=2
        assert abs(q['delta_admission_rate']-(q['tts']['admission_rate']-q['uniform']['admission_rate']))<1e-12
    mr=public_read(folder/'memory/ROWS.json');assert len(mr)==939;ms=public_read(folder/'memory/SUMMARY.json');g={a:{} for a in ['query_cos','spatial_cos','ridge']}
    # Recompute all bootstrap CIs directly from anonymous pair predictions.
    for ds in DATASETS:
        for sp in ['search','confirm']:
            for co in ['clean','corruption']:
                aa=[q for q in mr if q['dataset']==ds and q['split']==sp and ((q['condition']=='clean')==(co=='clean'))]
                for a in ['query_cos','spatial_cos','ridge','constant']:
                    calc=independent_node_stats(aa,a);assert numerically_equal(calc,ms[ds][sp][co][a]),(ds,sp,co,a,calc,ms[ds][sp][co][a]);checks+=1
                print('THREE_SCOPE_PUBLIC_BOOTSTRAP_RECHECK',ds,sp,co,flush=True)
        for a in g:
            v=ms[ds]['confirm']['corruption'][a];c,auc=v['correlation'],v['AUC'];g[a][ds]=bool((c['ci95'] is not None and c['ci95'][0]>0) or (auc['mean'] is not None and auc['mean']>.65 and auc['ci95'][0]>.5))
    md=public_read(folder/'memory/DECISION.json');assert md['gates']==g and md['passing_signals']==[a for a,v in g.items() if all(v.values())]
    assert md['memory_qualified']==bool(md['passing_signals'])
    result=dict(status='pass',checks=checks,source_bootstraps_recomputed=True,all_939_pair_predictions_reaggregated=True,
        independent_memory_weighted_AUC_and_correlation=True,independent_memory_float_tolerance=2e-11,
        public_audit_does_not_access_GT_or_private_features=True,private_feature_fitting_verified_by_root_separately=True,time=time.time())
    if not (folder/'PUBLIC_AUDIT.json').exists():write(folder/'PUBLIC_AUDIT.json',result)
    print('THREE_SCOPE_PUBLIC_AUDIT_PASS',checks,flush=True);return result

def root():
    import torch
    from scipy.linalg import cho_factor,cho_solve
    from scripts.score_decota_three_scope_v1 import seal_check,verify_cpu
    from vg_tta.decota_three_scope_v1 import key_vector
    verify();seal_check('GLOBAL_PREDICTION_BARRIER.json');seal_check('MEMORY_KEY_BARRIER.json');seal_check('BOUNDARY_PREDICTION_BARRIER.json')
    checks=0;keys=checked(BASE/'memory/KEYS.pt')['features'];label_free=[];tick=time.time()
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json')
        for row in p['rows']:
            for cond in p['conditions']:
                f=BASE/'features'/ds/cond/f"{row['ordinal']:05}.pt";q=checked(f);ids=np.asarray(row['frame_ids'],float)
                data,rc=c1.cache(ds,row['pool_parent'],cond);a,b=map(int,data['prediction']['indices']);k=min(4,b-a+1);edges=np.linspace(ids[a],ids[b]+1,k+1);chosen=[]
                for j in range(k):
                    pool=np.flatnonzero((ids>=edges[j])&(ids<edges[j+1])&(np.arange(len(ids))>=a)&(np.arange(len(ids))<=b))
                    if len(pool):chosen.append(int(pool[np.argmax(np.asarray(q['tts'])[pool])]))
                while len(chosen)<k:
                    remaining=[i for i in range(a,b+1) if i not in chosen]
                    far=[min(abs(ids[i]-ids[j]) for j in chosen) for i in remaining];chosen.append(remaining[int(np.argmax(far))])
                assert sorted(chosen)==q['positions'] and q['pixel_sha256']==rc['pixel_sha256'];checks+=len(ids)+4
                assert q['keys']['query'].shape==(768,) and q['keys']['spatial'].shape==(256,) and np.isfinite(q['keys']['geometry']).all()
                label_free.append(dict(dataset=ds,condition=cond,source_id=row['ordinal'],positions=q['positions'],tts=[float(x) for x in q['tts']],frame_ids=row['frame_ids'],source_initialized=True,raw_query_or_latent_exported=False))
                ex=checked(BASE/'evidence'/ds/cond/f"{row['ordinal']:05}.pt")['expert'];assert ex['positions4']==q['positions']
                for (_,pos),ob in ex['observations'].items():
                    probe=ob['probe'];scores=np.asarray(probe['target_scores'],float);reason='no_candidates' if not len(scores) else 'low_target_score' if scores.max()<.35 else 'ambiguous_distinct_instances' if probe['margin']<.05 else 'accepted'
                    # Old fallback generic admission has its same saved reason.
                    if ob['receipt']['context_active']:assert probe['accepted']==(reason=='accepted')
                    assert pos in q['positions'] and probe['frame_id']==row['frame_ids'][pos];checks+=2
    # Fit-independent NumPy/Cholesky readback of every prediction and exclusion.
    mr=read(PUB/'memory/ROWS.json')
    for ds in DATASETS:
        pp=[q for q in mr if q['dataset']==ds];xx=[]
        for q in pp:
            d=keys[f"{ds}:{q['condition']}:{q['donor']}"];r=keys[f"{ds}:{q['condition']}:{q['recipient']}"]
            def unit(x):
                x=np.asarray(x,float)
                return x/max(np.linalg.norm(x),1e-12)
            aa=np.r_[unit(d['query']),unit(d['spatial']),d['geometry']];bb=np.r_[unit(r['query']),unit(r['spatial']),r['geometry']]
            cq=float(np.clip(unit(d['query'])@unit(r['query']),-1,1));cs=float(np.clip(unit(d['spatial'])@unit(r['spatial']),-1,1))
            assert abs(cq-q['query_cos'])<1e-12 and abs(cs-q['spatial_cos'])<1e-12
            xx.append(np.r_[aa,bb,np.abs(aa-bb),cq,cs]);checks+=2
        x=np.array(xx);y=np.array([q['utility_v'] for q in pp]);receipts=read(BASE/f'memory/{ds}_FITTING_RECEIPTS.json');search=[i for i,q in enumerate(pp) if q['split']=='search'];memo={}
        for i,q in enumerate(pp):
            banned={q['donor'],q['recipient']};train=[j for j in search if banned.isdisjoint({pp[j]['donor'],pp[j]['recipient']})]
            assert receipts[i]['training_source_nodes']==sorted({pp[j][k] for j in train for k in ['donor','recipient']}) and len(train)==q['training_pairs']
            key=tuple(train)
            if key not in memo:
                a=x[train];mu=np.sum(a,axis=0)/len(a);sd=np.sqrt(np.sum((a-mu)**2,axis=0)/len(a));sd=np.where(sd>1e-12,sd,1.);z=(a-mu)/sd;mean=float(y[train].mean())
                beta=cho_solve(cho_factor(z@z.T+np.eye(len(z)),lower=True),y[train]-mean);memo[key]=(mu,sd,z,beta,mean)
            mu,sd,z,beta,mean=memo[key];expected=mean+((x[i]-mu)/sd)@z.T@beta
            assert abs(expected-q['ridge'])<2e-10 and abs(mean-q['constant'])<1e-12;checks+=len(train)+2
        print('THREE_SCOPE_ROOT_RIDGE_RECHECK',ds,len(pp),len(memo),flush=True)
    write(PUB/'scope/ACQUISITION_READBACK.json',label_free)
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,all_576_physical_frame_selections_independent=True,
        all_939_ridge_predictions_Cholesky_independent=True,double_role_source_exclusion_independent=True,
        per_step_objective_Adam_state_dense_audit=read(PUB/'scope/ROOT_AUDIT.json'),boundary_dense_audit=read(PUB/'boundary/ROOT_AUDIT.json'),
        recipient_delta_not_in_features=True,old_weights_states_not_modified=True,seconds=time.time()-tick,time=time.time()))
    print('THREE_SCOPE_ROOT_AUDIT_PASS',checks,flush=True)

if __name__=='__main__':root() if sys.argv[1]=='root' else public(sys.argv[1])
