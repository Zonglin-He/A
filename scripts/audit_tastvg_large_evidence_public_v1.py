"""Independent arithmetic/decision/source-bootstrap audit of anonymous export."""
import sys,json,hashlib,collections,math
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_summary(rows,fields,ratios=None):
    if not rows:return dict(cells=0,sources=0,metrics={},ratios={})
    sources=sorted({r['source_id'] for r in rows});orders=collections.defaultdict(list);values=[]
    for src in sources:
        ordervalues=[]
        for order in sorted({r['order'] for r in rows}):
            part=[r for r in rows if r['source_id']==src and r['order']==order]
            if not part:continue
            condmeans=[]
            for cond in sorted({r['condition'] for r in part}):
                q=[r for r in part if r['condition']==cond]
                condmeans.append([sum(r[f] for r in q)/len(q) for f in fields])
            v=np.mean(condmeans,0);ordervalues.append(v);orders[order].append(v)
        values.append(np.mean(ordervalues,0))
    mat=np.array(values);rng=np.random.default_rng(20261003);boots=[]
    for _ in range(100):
        draws=rng.integers(0,len(mat),(100,len(mat)))
        boots.append(np.mean(mat[draws],1))
    boot=np.concatenate(boots);q=np.quantile(boot,[.025,.975],0);met={};at={f:j for j,f in enumerate(fields)}
    for j,f in enumerate(fields):
        xs=mat[:,j];loo=[np.mean(np.delete(xs,k)) for k in range(len(xs))] if len(xs)>1 else xs
        met[f]=dict(mean=float(np.mean(xs)),ci95=q[:,j].tolist(),
            source_values={str(s):float(x) for s,x in zip(sources,xs)},
            order_values={o:float(np.mean(v,0)[j]) for o,v in orders.items()},
            leave_one_out_range=[float(min(loo)),float(max(loo))],cell_mean=float(np.mean([r[f] for r in rows])))
    rt={}
    for name,(num,den) in (ratios or {}).items():
        nums=boot[:,at[num]];ds=boot[:,at[den]];keep=ds>1e-12;mden=np.mean(mat[:,at[den]])
        rt[name]=dict(mean=float(np.mean(mat[:,at[num]])/mden) if mden>1e-12 else None,
            ci95=np.quantile(nums[keep]/ds[keep],[.025,.975]).tolist() if keep.any() else None,
            bootstrap_zero_denominator_draws=int((~keep).sum()),valid_draws=int(keep.sum()))
    return dict(cells=len(rows),sources=len(sources),metrics=met,ratios=rt,bootstrap_draws=10000,seed=20261003)

def audit(folder):
    folder=Path(folder);counts=collections.Counter();maxerr=0.;eps=1e-12
    def equal(a,b):
        nonlocal maxerr
        if isinstance(a,dict):
            assert set(a)==set(b),(set(a)^set(b))
            for k in a:equal(a[k],b[k])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            e=abs(float(a)-float(b));assert e<1e-9,(a,b,e);maxerr=max(maxerr,e);counts['numeric_checks']+=1
        else:assert a==b,(a,b)
    cfg=read(folder/'CONFIG.json');assert cfg['bandwidth_seconds']==.5 and cfg['large_radius']==.5 and cfg['dominant_balance']==.25
    seal=read(folder/'SIGNAL_SEAL.json');join=read(folder/'LABEL_JOIN.json')
    assert seal['GT_read'] is False and join['time']>seal['time']==join['signal_seal_time']
    assert sha(folder/'SIGNAL_SEAL.json')==join['signal_seal_sha256'] and sha(folder/'EVIDENCE_ROWS.json')==seal['file_sha256']
    evidence={e['cell_key']:e for e in read(folder/'EVIDENCE_ROWS.json')};derived={}
    assert len(evidence)==288
    for key,e in evidence.items():
        a=e['anchor_index'];xy=e['intervals'];lp=e['native_logprior'];h=e['bandwidth_normalized'];scores={}
        equal(h,.5/e['duration_seconds'])
        assert len(xy)==len(e['candidate_indices'])==32 and 0<=a<8
        for pos in e['native_offset_positions']:
            for col in [0,1]:equal(sum(math.exp(lp[i][col]) for i in pos),.5)
        assert sorted(sum(e['native_offset_positions'],[]))==list(range(len(lp)))
        scores['N']=[lp[i][0]+lp[j][1] for i,j in e['candidate_indices']]
        sides=[];u=[]
        for view in ['view0','view1']:
            pp=e['proposals_'+view];vals=[]
            for x in xy:
                cols=[]
                for col in [0,1]:
                    terms=[-.5*((x[col]-p[col])/h)**2 for p in pp]
                    if terms:
                        mx=max(terms);cols.append(mx+math.log(sum(math.exp(v-mx) for v in terms)/len(terms)))
                vals.append(cols)
            sides.append(vals if pp else None);u.append([sum(v) for v in vals] if pp else [0.]*32)
        scores['U']=u[0];scores['S']=[min(u[0][i]-u[0][a],u[1][i]-u[1][a]) for i in range(32)] if all(v is not None for v in sides) else [0.]*32
        equal(scores,e['scores']);equal(e['details']['view0'],u[0]);equal(e['details']['view1'],u[1])
        equal(e['details']['view0_endpoint_logdensity'],sides[0]);equal(e['details']['view1_endpoint_logdensity'],sides[1])
        assert e['details']['U_available']==(sides[0] is not None) and e['details']['S_available']==all(v is not None for v in sides)
        for s in ['N','U','S']:
            for n in [8,32]:
                mx=max(scores[s][:n]);top=[i for i,v in enumerate(scores[s][:n]) if mx-v<=eps]
                chosen=top[0] if len(top)==1 and mx>scores[s][a]+eps else a
                assert chosen==e['choices'][s+str(n)];counts['decisions']+=1
        geo=[]
        for x in xy:
            ds=x[0]-xy[a][0];de=x[1]-xy[a][1];d=abs(ds)+abs(de);b=2*min(abs(ds),abs(de))/d if d>eps else 0
            if d<=eps:k='same'
            elif b<=.25+eps:
                if abs(ds)>=abs(de):k='trim_start' if ds>0 else 'expand'
                else:k='trim_end' if de<0 else 'expand'
            elif ds<0<de:k='expand'
            elif de<0<ds:k='trim_both'
            else:k='shift'
            radius=d/(xy[a][1]-xy[a][0]);geo.append(dict(ds=ds,de=de,balance=b,radius=radius,kind=k,large=radius>=.5-eps))
        equal(geo,e['geometry']);derived[key]=geo;counts['candidate_geometry']+=32
    allrows=[]
    for sp in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            rows=read(folder/sp/ds/'ROWS.json');br=read(folder/sp/ds/'BINARY_ROWS.json');allrows+=rows;expected=[]
            for r in rows:
                if r['expert_scheduled']:
                    e=evidence[r['evidence_key']];a=e['anchor_index'];v=r['candidate_v'];t=r['candidate_t'];g=derived[e['cell_key']]
                    assert r['A_state_pre_sha256']==e['A_state_pre_sha256'] and r['A_state_post_sha256']==e['A_state_post_sha256']
                    for s in ['N','U','S']:
                        ev=[z-e['scores'][s][a] for z in e['scores'][s]]
                        for group in cfg['groups']:
                            ids=[i for i,z in enumerate(g) if i!=a and z['kind']!='same' and (group=='all_changed' or
                                (z['large'] and (group=='large' or group=='large_'+z['kind']))) and abs(v[i]-v[a])>eps]
                            if not ids:continue
                            pos=[i for i in ids if v[i]>v[a]+eps];neg=[i for i in ids if v[i]<v[a]-eps]
                            tp=sum(ev[i]>eps for i in pos);fp=sum(ev[i]>eps for i in neg);fn=len(pos)-tp;tn=len(neg)-fp;den=len(ids)
                            auc=None
                            if pos and neg:
                                vals=[]
                                for i in pos:
                                    for j in neg:
                                        d=ev[i]-ev[j];vals.append(1 if d>eps else 0 if d< -eps else .5)
                                auc=float(np.mean(vals))
                            expected.append(dict(**{k:r[k] for k in ['dataset','split','source_id','condition','order','arrival']},signal=s,group=group,
                                tp=tp/den,fp=fp/den,fn=fn/den,tn=tn/den,positive=len(pos)/den,negative=len(neg)/den,
                                accepted=(tp+fp)/den,auc=auc,candidates=den,positives=len(pos),negatives=len(neg),TP=tp,FP=fp,FN=fn,TN=tn))
                for n in [8,32]:
                    equal(r[f'O{n}_v'],max(r['candidate_v'][:n]) if r['expert_scheduled'] else r['A8_v'])
                    for s in ['N','U','S']:
                        arm=s+str(n);idx=e['choices'][arm] if r['expert_scheduled'] else None
                        vv=r['candidate_v'][idx] if idx is not None else r['A8_v'];tt=r['candidate_t'][idx] if idx is not None else r['A8_t']
                        equal(vv,r[arm+'_v']);equal(tt,r[arm+'_t']);d=vv-r['A8_v']
                        for f,x in [('gain',d),('t_gain',tt-r['A8_t']),('regret',r[f'O{n}_v']-vv),('gross_gain',max(d,0)),('gross_loss',max(-d,0)),
                                ('changed',int(idx is not None and idx!=a)),('severe',int(d<-.05))]:equal(r[arm+'_'+f],x)
                counts['arrivals']+=1
            equal(expected,br);summary=read(folder/sp/ds/'SUMMARY.json');binarysummary=read(folder/sp/ds/'DISCRIMINATION.json')
            for gr in ['corruption','clean']:
                for sub in ['all','expert','nonexpert']:
                    rr=[r for r in rows if (r['condition']=='clean')==(gr=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))]
                    z=summary[gr][sub];az=independent_summary(rr,list(z['metrics']));equal({k:z[k] for k in az},az)
                    tr={}
                    for s in ['N','U','S']:
                        for n in [8,32]:
                            arm=s+str(n);tr[arm]=dict(improved=sum(r[arm+'_gain']>eps for r in rr),harmed=sum(r[arm+'_gain']< -eps for r in rr),
                                unchanged=sum(abs(r[arm+'_gain'])<=eps for r in rr),severe_harm_gt5pp=sum(r[arm+'_gain']<-.05 for r in rr),
                                correctness={str(q):dict(destroyed=sum(r['A8_v']>q>=r[arm+'_v'] for r in rr),rescued=sum(r[arm+'_v']>q>=r['A8_v'] for r in rr)) for q in [.3,.5]})
                    equal(tr,z['transitions'])
                for group in cfg['groups']:
                    for s in ['N','U','S']:
                        rr=[r for r in expected if r['group']==group and r['signal']==s and (r['condition']=='clean')==(gr=='clean')]
                        z=binarysummary[gr][group][s];az=independent_summary(rr,['tp','fp','fn','tn','positive','negative','accepted'],
                            dict(precision=('tp','accepted'),benefit_recall=('tp','positive'),harm_acceptance=('fp','negative')))
                        ar=[r for r in rr if r['auc'] is not None];az['auc']=independent_summary(ar,['auc'])
                        az['raw_counts']={k:sum(r[k] for r in rr) for k in ['candidates','positives','negatives','TP','FP','FN','TN']}
                        az['eligible_auc_cells']=len(ar);az['undefined_auc_cells']=len(rr)-len(ar);equal(z,az)
            counts['panels']+=1
    assert counts['arrivals']==1152 and counts['decisions']==1728 and counts['candidate_geometry']==9216
    result=dict(status='pass',checks=dict(counts),max_numeric_error=maxerr,
        signal_seal_precedes_cached_GT_join=True,private_assets_required=False,new_model_calls=0,new_expert_calls=0)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':audit(sys.argv[1])
